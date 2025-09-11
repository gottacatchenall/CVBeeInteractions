import os
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics
from torchvision import transforms
import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger
import io
from transformers import AutoModel
import glob
import itertools
import webdataset as wds
import kornia.augmentation as K
from PIL import Image
from src.checkpoints import AsyncTrainableCheckpoint 
import pandas as pd
from torch.utils.data import IterableDataset, DataLoader

torch.set_float32_matmul_precision('high')
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

def model_paths():
    return {      
        "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",     # 87M Params
        "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",    # 303M Params
        "huge": "facebook/dinov3-vith16plus-pretrain-lvd1689m", # 841M Params
    }

def image_embed_dim():
    return {      
        "base": 768,     # 87M Params
        "large": 1024,   # 303M Params
        "huge": 1280,    # 841M Params
    }



def default_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

class SamplerDecoder():
    def __init__(self, transform=None):
        if transform == None:
            self.transform = default_transform()
        else:
            self.transform = transform
        
    def __call__(self, sample):
        img_bytes, meta = sample
        img = Image.open(io.BytesIO(img_bytes))
        img = self.transform(img)
        label = torch.tensor(meta["label"], dtype=torch.long)
        return img, label
    

class PairedIterableDataset(IterableDataset):
    """Yields batches of (plants, pollinators) from two independent datasets."""
    def __init__(self, plant_dataset, poll_dataset, pair_mask):
        self.plant_dataset = plant_dataset
        self.poll_dataset = poll_dataset
        self.pair_mask = pair_mask

    def __iter__(self):
        plant_iter = iter(self.plant_dataset)
        poll_iter = iter(self.poll_dataset)
        for plant_sample, poll_sample in zip(plant_iter, poll_iter):
            plant_img, plant_label = plant_sample
            poll_img, poll_label = poll_sample
            if self.pair_mask[plant_label, poll_label].item():
                yield torch.stack((plant_label, poll_label)), plant_img, poll_img
            else:
                continue

def load_metaweb(path):
    df = pd.read_csv(path)
    mw = torch.zeros(max(df.plant_index)+1, max(df.pollinator_index)+1)
    for _,r in df.iterrows():
        i,j,_,_,int = r
        mw[j,i] = int
    return mw

class MetawebMaskMaker():
    pass

class ZeroShotMaskMaker(MetawebMaskMaker):
    def __init__(self, metaweb, bee_holdouts, plant_holdouts):
        self.metaweb = metaweb
        self.bee_holdouts = bee_holdouts
        self.plant_holdouts = plant_holdouts

    def make_mask(self, metaweb):
        # metaweb is a 2D matrix [num_plants, num_bees] of 0/1 labels
        num_plants, num_bees = metaweb.shape
        train_mask = torch.ones((num_plants, num_bees), dtype=torch.bool)
        # Mask out any pair where either the plant OR the bee is a holdout
        if self.plant_holdouts is not None and len(self.plant_holdouts) > 0:
            train_mask[self.plant_holdouts, :] = False
        if self.bee_holdouts is not None and len(self.bee_holdouts) > 0:
            train_mask[:, self.bee_holdouts] = False
        return train_mask

class PlantPollinatorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        plant_shard_dir,
        poll_shard_dir,
        interactions_path,
        train_pattern = "train-*.tar",
        test_pattern = "test-*.tar",
        val_pattern = "val-*.tar",
        batch_size=32,
        num_workers=4,
        mask_maker: MetawebMaskMaker = None,
    ):
        super().__init__()
        self.plant_shard_dir = plant_shard_dir
        self.poll_shard_dir = poll_shard_dir

        self.metaweb = load_metaweb(interactions_path)
        self.mask_maker = mask_maker

        self.train_pattern = train_pattern
        self.test_pattern = test_pattern
        self.val_pattern = val_pattern

        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def make_dataset(self, shards):
        decoder = SamplerDecoder()
        return (
            wds.WebDataset(
                glob.glob(shards), 
                shardshuffle=True
            )
            .decode()
            .to_tuple("jpg", "json")
        ).map(decoder)

    def setup(self, stage=None):
        # Build masks
        if self.mask_maker is not None:
            self.train_mask = self.mask_maker.make_mask(self.metaweb)
        else:
            self.train_mask = torch.ones_like(self.metaweb, dtype=torch.bool)
        self.val_mask = self.train_mask == False

        self.bee_train_dataset = self.make_dataset(os.path.join(self.poll_shard_dir, self.train_pattern)) #.batched(self.batch_size)
        self.plant_train_dataset = self.make_dataset(os.path.join(self.plant_shard_dir, self.train_pattern)) #.batched(self.batch_size)

        self.bee_test_dataset = self.make_dataset(os.path.join(self.poll_shard_dir, self.test_pattern)) #.batched(self.batch_size)
        self.plant_test_dataset = self.make_dataset(os.path.join(self.plant_shard_dir, self.test_pattern)) #.batched(self.batch_size)


    def train_dataloader(self):
        paired_dataset = PairedIterableDataset(
            self.plant_train_dataset,
            self.bee_train_dataset,
            self.train_mask, 
        )
        return DataLoader(
            paired_dataset,
            batch_size=self.batch_size,  # WebDataset already batches if you use .batched()
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        paired_dataset = PairedIterableDataset(
            self.plant_test_dataset,
            self.bee_test_dataset,
            self.val_mask, 
        )
        return DataLoader(
            paired_dataset,
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )



class VitInteractionClassifier(pl.LightningModule):
    def __init__(
        self, 
        lr = 1e-3, 
        min_crop_size = 0.5,
        num_bees=19,
        num_plants = 158,
        interactions_path = "./data/interactions.csv",
        lambda_bee = 0.5,
        lambda_plant = 0.5,
        lambda_int = 1.0,
        embed_dim = 128,
        num_vit_unfrozen_layers = 2,
        model_type="base"
    ):
        super().__init__()
        self.save_hyperparameters()
        self.metaweb = load_metaweb(interactions_path).long()

        # ---------- Image Model  ----------
        model_name = model_paths()[model_type]
        self.image_model = AutoModel.from_pretrained(
            model_name, 
            local_files_only=True
        )
        # Freeze ViT layers 
        for param in self.image_model.parameters():
            param.requires_grad = False

        # Unfreeze last transformer block
        for param in self.image_model.layer[-1*num_vit_unfrozen_layers:].parameters():
            param.requires_grad = True
        
        # ---------- Shared Embedding Model  ----------
        self.embedding_model = nn.Sequential(
            nn.Linear(image_embed_dim()[model_type], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

        # ---------- Interaction Classification Head  ----------
        self.interaction_classification_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2*embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )


        # ---------- Image Transform  ----------
        self.transform = torch.nn.Sequential(
            K.RandomResizedCrop((224,224), scale=(min_crop_size, 1.0)),
            K.RandomHorizontalFlip(),
            K.ColorJitter(0.4,0.4,0.4,0.1, p=0.8),
            K.RandomGrayscale(p=0.2),
            K.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ).to(device)


        self.criterion_ce = nn.CrossEntropyLoss()

        self.bee_train_metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.classification.Accuracy(
                    task="multiclass", 
                    num_classes=num_bees
                ),
                "MAP": torchmetrics.classification.MulticlassPrecision(num_classes=num_bees, average='macro'),
            },
            prefix="train_bee",
        )
        self.bee_valid_metrics = self.bee_train_metrics.clone(prefix="valid_bee")
       
        self.plant_train_metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.classification.Accuracy(
                    task="multiclass", 
                    num_classes=num_plants
                ),
                "MAP": torchmetrics.classification.MulticlassPrecision(num_classes=num_plants, average='macro'),
            },
            prefix="train_plant",
        )
        self.plant_valid_metrics = self.plant_train_metrics.clone(prefix="valid_plant")

        self.interaction_train_metrics = torchmetrics.MetricCollection(
            {
                "MAP": torchmetrics.classification.MulticlassAveragePrecision(num_classes=2, average='macro'),
                "ROCAUC": torchmetrics.classification.MulticlassAUROC(num_classes=2, average='macro'),
            },
            prefix="train_interaction",
        )
        self.interaction_valid_metrics = self.interaction_train_metrics.clone(prefix="valid_interaction")
        
        

    def forward(self, plant_img, bee_img):
        plant_embed, bee_embed = self.dual_embed(plant_img, bee_img)
        int_e = torch.concat((plant_embed, bee_embed), dim=1)
        logits = self.interaction_classification_head(int_e)
        return logits
    
    def single_embed(self, x):
        x =  self.image_model(x).pooler_output
        x = self.embedding_model(x)
        return x    
    def dual_embed(self, x, y):
        return self.single_embed(x), self.single_embed(y)

    def classify_plant(self, x):
        x = self.single_embed(x)
        return x
    
    def classify_bee(self, x):
        x = self.single_embed(x)
        return x
    def classify_interaction(self, x, y):
        return self.forward(x, y)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y, z = batch
        y = self.transform(y)
        z = self.transform(z)
        return x, y, z
        
    def training_step(self, batch, batch_idx):
        z, plant_img, bee_img = batch        

        pi = z[:,0]
        bi = z[:,1]

        plant_embed, bee_embed = self.dual_embed(plant_img, bee_img)

        int_e = torch.concat((plant_embed, bee_embed), dim=1)

        int_logits = self.interaction_classification_head(int_e)
        int_loss = self.criterion_ce(int_logits, self.metaweb[pi,bi])
        loss = int_loss

        with torch.no_grad():
            probs = F.softmax(int_logits, dim=1)
            batch_value = self.interaction_train_metrics(probs, self.metaweb[pi,bi])
            self.log_dict(batch_value, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)
    
        return loss

    
    def validation_step(self, batch, batch_idx):
        z, plant_img, bee_img = batch        

        pi = z[:,0]
        bi = z[:,1]

        plant_embed, bee_embed = self.dual_embed(plant_img, bee_img)

        int_e = torch.concat((plant_embed, bee_embed), dim=1)
        int_logits = self.interaction_classification_head(int_e)

        int_loss = self.criterion_ce(int_logits, self.metaweb[pi,bi])
        loss = int_loss

        with torch.no_grad():
            probs = F.softmax(int_logits, dim=1)
            batch_value = self.interaction_valid_metrics(probs, self.metaweb[pi,bi])
            self.log_dict(batch_value, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return {
            "optimizer": optimizer,
        }


def predict_interaction(net, plant_img, bee_img):
    plant_embed, bee_embed = net.dual_embed(plant_img, bee_img)
    int_e = torch.concat((plant_embed, bee_embed), dim=1)
    int_logits = net.interaction_classification_head(int_e)
    return int_logits

def predict_interactions(net, loader):
    summed_interactions = torch.zeros(net.metaweb.shape)
    int_counts = torch.zeros(net.metaweb.shape)
    for batch in loader:
        z, plant_img, bee_img = batch
        pi = z[:,0]
        bi = z[:,1]
        int_logits = predict_interaction(net, plant_img, bee_img)
        int_softmax = F.softmax(int_logits, dim=1)
        summed_interactions[pi,bi] += int_softmax
        int_counts[pi,bi] += 1
    return summed_interactions / int_counts
    
    


def main(args):
    base_path = os.path.join("/scratch", "mcatchen", "iNatImages", "data") if args.cluster else "./data"
    log_path = os.path.join(base_path, "logs")
    bee_dir = "bombus_wds" 
    plant_dir = "plant_wds" 

    num_bee_holdouts, num_plant_holdouts = args.num_bee_holdouts, args.num_plant_holdouts

    metaweb_path = os.path.join(base_path, "interactions.csv")
    metaweb = load_metaweb(metaweb_path).long()
    
    # Build random holdouts by index
    plant_indices = torch.arange(metaweb.shape[0])
    bee_indices = torch.arange(metaweb.shape[1])
    if num_plant_holdouts > 0:
        plant_holdouts = plant_indices[torch.randperm(len(plant_indices))[:num_plant_holdouts]]
    else:
        plant_holdouts = torch.tensor([], dtype=torch.long)
    if num_bee_holdouts > 0:
        bee_holdouts = bee_indices[torch.randperm(len(bee_indices))[:num_bee_holdouts]]
    else:
        bee_holdouts = torch.tensor([], dtype=torch.long)

    species_data = PlantPollinatorDataModule(
        plant_shard_dir=os.path.join(base_path, plant_dir),
        poll_shard_dir=os.path.join(base_path, bee_dir),
        interactions_path=metaweb_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mask_maker=ZeroShotMaskMaker(
            metaweb=metaweb,
            bee_holdouts=bee_holdouts,
            plant_holdouts=plant_holdouts
        )
    )

    net = VitInteractionClassifier(
        lr=args.lr,
        model_type = args.model,
        min_crop_size=args.min_crop_size,
        num_vit_unfrozen_layers=args.num_unfrozen,
        interactions_path=os.path.join(base_path, "interactions.csv")
    )

    logger = CSVLogger(log_path, name=os.environ.get("SLURM_JOB_NAME") or "InteractionTest")

    checkpoint_cb = AsyncTrainableCheckpoint(
        dirpath = os.path.join(logger.log_dir, "checkpoints")
    )
    # Configure accelerator/devices
    slurm_nodes = os.environ.get("SLURM_JOB_NUM_NODES")
    num_nodes = int(slurm_nodes) if slurm_nodes is not None else 1
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = torch.cuda.device_count()
        strategy = "ddp" if devices and devices > 1 or num_nodes > 1 else "auto"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
        strategy = "auto"
    else:
        accelerator = "cpu"
        devices = 1
        strategy = "auto"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        profiler="simple",
        logger=logger,
        enable_checkpointing=False,   # Turn off default ckpt
        callbacks=[checkpoint_cb],
        num_nodes=num_nodes, 
        max_epochs=args.max_epochs,
        enable_progress_bar=False
    )

    trainer.fit(net, species_data)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--min_crop_size', type=float, default=0.7)
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_bee_holdouts', type=int, default=2)
    parser.add_argument('--num_plant_holdouts', type=int, default=10)
    parser.add_argument('--num_unfrozen', type=int, default=2)
    parser.add_argument('--contrastive', action='store_true')
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--persistent_workers', action='store_true')
    parser.add_argument('--species', default='bees', choices=['plants', 'bees'])
    parser.add_argument('--model', default='base', choices=['base', 'large', 'huge'])


    args = parser.parse_args()

    base_path = os.path.join("/scratch", "mcatchen", "iNatImages", "data") if args.cluster else "./data"
    data_dir = "bombus_wds" if args.species == "bees" else "plant_wds"

    image_dir = os.path.join(base_path, data_dir)
    log_path = os.path.join(base_path, "logs")
    main(args)


