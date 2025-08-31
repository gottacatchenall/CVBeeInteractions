import os
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics
from torchvision import transforms
import torchvision
import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger
import io
from transformers import AutoModel, AutoImageProcessor
import glob
import itertools
import webdataset as wds
import kornia.augmentation as K
import itertools
from PIL import Image
from src.checkpoints import AsyncTrainableCheckpoint 
import pandas as pd
from torch.utils.data import IterableDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.set_float32_matmul_precision('high')

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

def get_metaweb_train_mask(mw, train_prop=0.8):
    shape = mw.shape
    train_mask = (torch.randperm(shape[0]*shape[1]) < int(train_prop*shape[0]*shape[1])).bool().view(shape)
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
    ):
        super().__init__()
        self.plant_shard_dir = plant_shard_dir
        self.poll_shard_dir = poll_shard_dir

        self.metaweb = load_metaweb(interactions_path)

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
        self.train_mask = get_metaweb_train_mask(self.metaweb)
        self.test_mask = self.train_mask == False

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
    
    def test_dataloader(self):
        paired_dataset = PairedIterableDataset(
            self.plant_test_dataset,
            self.bee_test_dataset,
            self.test_mask, 
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
            embed_dim = 32,
            model_type="base"
    ):
        super().__init__()
        self.save_hyperparameters()
        self.metaweb = load_metaweb(interactions_path).long().cuda()
        self.onehot_metaweb = F.one_hot(self.metaweb).cuda()

        # ---------- Image Model  ----------
        model_name = model_paths()[model_type]
        self.image_model = AutoModel.from_pretrained(
            model_name, 
            local_files_only=True
        )
        # Freeze ViT layers 
        for param in self.image_model.parameters():
            param.requires_grad = False
        
        # ---------- Shared Embedding Model  ----------
        self.embedding_model = nn.Sequential(
            nn.Linear(image_embed_dim()[model_type], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            #nn.ReLU(),
            #nn.Linear(64, 32),
        )

        # ---------- Bee Classification Model  ----------
        self.bee_classification_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embed_dim, num_bees)
        )

        # ---------- Plant Classification Model  ----------
        self.plant_classification_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embed_dim, num_plants)
        )

        # ---------- Interaction Classification Head  ----------
        self.interaction_classification_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2*embed_dim, 2)
        )


        # ---------- Image Transform  ----------
        self.transform = torch.nn.Sequential(
            K.RandomResizedCrop((224,224), scale=(min_crop_size, 1.0)),
            K.RandomHorizontalFlip(),
            K.ColorJitter(0.4,0.4,0.4,0.1, p=0.8),
            K.RandomGrayscale(p=0.2),
            K.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ).to("cuda")


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
                "MAP": torchmetrics.classification.BinaryAveragePrecision(),
                "ROCAUC": torchmetrics.classification.BinaryAUROC(),
            },
            prefix="train_interaction",
        )
        self.interaction_valid_metrics = self.interaction_train_metrics.clone(prefix="valid_interaction")
        
        

    def forward(self, x):
        with torch.no_grad():
            x =  self.image_model(x).pooler_output
        x = self.embedding_model(x)
        x = self.classification_head(x)
        return x
    
    def single_embed(self, x):
        x =  self.image_model(x).pooler_output
        x = self.embedding_model(x)
        return x    
    def dual_embed(self, x, y):
        return self.single_embed(x), self.single_embed(y)

    def classify_plant(self, x):
        x = self.embed(x)
        x = self.plant_classification_head(x)
        return x
    
    def classify_bee(self, x):
        x = self.embed(x)
        x = self.bee_classification_head(x)
        return x
    def classify_interaction(self, x, y):
        x = self.embed(x)
        x = self.interaction_classification_head(x)
        return x

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
        
        bee_logits = self.bee_classification_head(bee_embed)
        bee_loss = self.criterion_ce(bee_logits, bi)
        

        plant_logits = self.plant_classification_head(plant_embed)
        plant_loss = self.criterion_ce(plant_logits, pi)


        int_e = torch.concat((plant_embed, bee_embed), dim=1)

        int_logits = self.interaction_classification_head(int_e)
        int_loss = self.criterion_ce(int_logits, self.metaweb[pi,bi])


        lb, lp, li = self.hparams.lambda_bee, self.hparams.lambda_plant, self.hparams.lambda_int

        loss = lb*bee_loss + lp*plant_loss + li*int_loss

        with torch.no_grad():
            batch_value = self.plant_train_metrics(plant_logits, pi)
            self.log_dict(batch_value)
            batch_value = self.bee_train_metrics(bee_logits, bi)
            self.log_dict(batch_value)
            batch_value = self.interaction_train_metrics(int_logits, self.onehot_metaweb[pi,bi])
            self.log_dict(batch_value)
            self.log("train_loss", loss, prog_bar=False)
    
        return loss

    
    def validation_step(self, batch, batch_idx):
        z, plant_img, bee_img = batch        

        pi = z[:,0]
        bi = z[:,1]

        plant_embed, bee_embed = self.dual_embed(plant_img, bee_img)
        
        bee_logits = self.bee_classification_head(bee_embed)
        bee_loss = self.criterion_ce(bee_logits, bi)
    
        plant_logits = self.plant_classification_head(plant_embed)
        plant_loss = self.criterion_ce(plant_logits, pi)


        int_e = torch.concat((plant_embed, bee_embed), dim=1)
        int_logits = self.interaction_classification_head(int_e)

        int_loss = self.criterion_ce(int_logits, self.metaweb[pi,bi])

        lb, lp, li = self.hparams.lambda_bee, self.hparams.lambda_plant, self.hparams.lambda_int
        loss = lb*bee_loss + lp*plant_loss + li*int_loss

        with torch.no_grad():
            batch_value = self.interaction_valid_metrics(int_logits, self.onehot_metaweb[pi,bi])
            self.log_dict(batch_value)
            self.log("val_loss", loss, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return {
            "optimizer": optimizer,
        }


def main(args):
    base_path = os.path.join("/scratch", "mcatchen", "iNatImages", "data") if args.cluster else "./data"
    log_path = os.path.join(base_path, "logs")
    bee_dir = "bombus_wds" 
    plant_dir = "plant_wds" 

    species_data = PlantPollinatorDataModule(
        plant_shard_dir=os.path.join(base_path, plant_dir),
        poll_shard_dir=os.path.join(base_path, bee_dir),
        interactions_path=os.path.join(base_path, "interactions.csv"),
        batch_size=128,
        num_workers=args.num_workers,
    )

    net = VitInteractionClassifier(
        lr=args.lr,
        model_type = args.model,
        min_crop_size=args.min_crop_size,
        interactions_path=os.path.join(base_path, "interactions.csv")
    )

    logger = CSVLogger(log_path, name=os.environ.get("SLURM_JOB_NAME"))

    checkpoint_cb = AsyncTrainableCheckpoint(
        dirpath = os.path.join(logger.log_dir, "checkpoints")
    )
    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES"))
    gpus = torch.cuda.device_count()

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        strategy='ddp',
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






