import os, glob, io, itertools
import torch
import pandas as pd
import numpy as np
import argparse
import json 

import torch.nn.functional as F
import pytorch_lightning as pl
import webdataset as wds

from torch import nn
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoModel
from torchvision import transforms
from lightning.pytorch.loggers import CSVLogger
from PIL import Image
from collections import defaultdict

torch.set_float32_matmul_precision('high')

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


# -----------------------
#   Utils
# -----------------------
def default_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])


class SamplerDecoder():
    """Decode WDS samples into (image_tensor, label)."""
    def __init__(self, transform=None):
        self.transform = transform or default_transform()

    def __call__(self, sample):
        img_bytes, meta = sample
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(meta["label"], dtype=torch.long)
        return img, label


def load_metaweb(path):
    df = pd.read_csv(path)
    mw = torch.zeros(max(df.plant_index)+1, max(df.pollinator_index)+1)
    for _, r in df.iterrows():
        i, j, _, _, inter = r
        mw[j, i] = inter
    return mw


# -----------------------
#   Zero-Shot Mask Maker
# -----------------------
class ZeroShotMaskMaker:
    def __init__(self, num_plants, num_bees, holdout_frac=0.2, seed=0):
        rng = np.random.RandomState(seed)
        plant_ids = np.arange(num_plants)
        bee_ids   = np.arange(num_bees)

        self.holdout_plants = rng.choice(plant_ids, int(len(plant_ids)*holdout_frac), replace=False)
        self.holdout_bees   = rng.choice(bee_ids, int(len(bee_ids)*holdout_frac), replace=False)

        self.train_plants = np.setdiff1d(plant_ids, self.holdout_plants)
        self.train_bees   = np.setdiff1d(bee_ids, self.holdout_bees)

    def is_train_pair(self, plant_id, bee_id):
        return (plant_id in self.train_plants) and (bee_id in self.train_bees)

    def is_val_pair(self, plant_id, bee_id):
        return (plant_id in self.holdout_plants) or (bee_id in self.holdout_bees)


# -----------------------
#   Bag Dataset
# -----------------------
class PairedBagDataset(IterableDataset):
    def __init__(self, plant_dataset, bee_dataset, metaweb, mask_maker, split="train", max_instances=8):
        self.plant_dataset = plant_dataset
        self.bee_dataset   = bee_dataset
        self.metaweb       = metaweb
        self.mask_maker    = mask_maker
        self.max_instances = max_instances
        self.split         = split

    def __iter__(self):
        plant_iter = iter(self.plant_dataset)
        bee_iter   = iter(self.bee_dataset)

        for (plant_img, plant_label), (bee_img, bee_label) in zip(plant_iter, bee_iter):
            plant_id, bee_id = plant_label.item(), bee_label.item()

            if self.split == "train" and not self.mask_maker.is_train_pair(plant_id, bee_id):
                continue
            if self.split == "val" and not self.mask_maker.is_val_pair(plant_id, bee_id):
                continue

            bag_size = torch.randint(2, self.max_instances+1, (1,)).item()
            plant_imgs = plant_img.unsqueeze(0).repeat(bag_size, 1, 1, 1)
            bee_imgs   = bee_img.unsqueeze(0).repeat(bag_size, 1, 1, 1)
            bag_label  = self.metaweb[plant_id, bee_id].long()

            yield plant_imgs, bee_imgs, bag_size, bag_label, plant_id, bee_id


def collate_fn_bags(batch):
    plants, bees, bag_sizes, labels, plant_ids, bee_ids = zip(*batch)
    plant_all = torch.cat(plants, dim=0)
    bee_all   = torch.cat(bees, dim=0)
    return (
        plant_all,
        bee_all,
        list(bag_sizes),
        torch.tensor(labels),
        torch.tensor(plant_ids),
        torch.tensor(bee_ids),
    )


# -----------------------
#   Attention Pooling
# -----------------------
class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, bag_sizes):
        bag_reps, offset = [], 0
        for size in bag_sizes:
            bag_slice = x[offset: offset + size]
            attn_weights = torch.softmax(self.attn(bag_slice), dim=0)
            bag_rep = torch.sum(attn_weights * bag_slice, dim=0, keepdim=True)
            bag_reps.append(bag_rep)
            offset += size
        return torch.cat(bag_reps, dim=0)


# -----------------------
#   MIL Model
# -----------------------
class VitInteractionClassifierMIL(pl.LightningModule):
    def __init__(self, bee_names, plant_names, lr=5e-4, embed_dim=32, model_type="base", num_classes=2, num_unfrozen_vit_layers=1):
        super().__init__()
        self.save_hyperparameters()
        self.bee_names = bee_names
        self.plant_names = plant_names

        # ---- Backbone ----
        model_paths = {
            "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "huge": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        }
        embed_dims = {"base": 768, "large": 1024, "huge": 1280}
        self.backbone = AutoModel.from_pretrained(model_paths[model_type], local_files_only=True)

        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.layer[-1*num_unfrozen_vit_layers:].parameters():
            p.requires_grad = True

        # ---- Embedding head ----
        self.embedding_model = nn.Sequential(
            nn.Linear(embed_dims[model_type], 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

        # ---- MIL classifier ----
        self.attn_pool = AttentionPooling(2 * embed_dim)
        self.interaction_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

        self.criterion = nn.CrossEntropyLoss()

        # storage for per-epoch val metrics
        self.val_outputs = []

    def single_embed(self, x):
        feats = self.backbone(x).pooler_output
        return self.embedding_model(feats)

    def forward(self, plant_all, bee_all, bag_sizes):
        plant_embed = self.single_embed(plant_all)
        bee_embed   = self.single_embed(bee_all)
        pair_embed  = torch.cat((plant_embed, bee_embed), dim=1)
        bag_reps = self.attn_pool(pair_embed, bag_sizes)
        logits = self.interaction_head(bag_reps)
        return logits

    def training_step(self, batch, batch_idx):
        plant_all, bee_all, bag_sizes, labels, _, _ = batch
        logits = self(plant_all, bee_all, bag_sizes)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        plant_all, bee_all, bag_sizes, labels, plant_ids, bee_ids = batch
        logits = self(plant_all, bee_all, bag_sizes)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        # store for per-epoch aggregation
        self.val_outputs.append({
            "loss": loss.detach(),
            "preds": preds.cpu(),
            "logits": logits.detach().cpu(),
            "labels": labels.cpu(),
            "plants": plant_ids.cpu(),
            "bees": bee_ids.cpu(),
        })

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss


    def on_validation_epoch_end(self):
        # aggregate per-species accuracy
        plant_correct, plant_total = defaultdict(int), defaultdict(int)
        bee_correct, bee_total = defaultdict(int), defaultdict(int)

        for out in self.val_outputs:
            preds, labels = out["preds"], out["labels"]
            plants, bees = out["plants"], out["bees"]
            for p, b, y, yhat in zip(plants, bees, labels, preds):
                correct = int(y == yhat)
                plant_total[p.item()] += 1
                bee_total[b.item()] += 1
                plant_correct[p.item()] += correct
                bee_correct[b.item()] += correct

        # log average per-species accuracy
        if plant_total:
            accs = [plant_correct[k] / plant_total[k] for k in plant_total]
            self.log("val_plant_acc_mean", np.mean(accs))
        if bee_total:
            accs = [bee_correct[k] / bee_total[k] for k in bee_total]
            self.log("val_bee_acc_mean", np.mean(accs))

        log_dir = os.path.join(self.logger.log_dir, "per_species")
        os.makedirs(log_dir, exist_ok=True)

        plant_df = pd.DataFrame([
            {"plant_id": k, "acc": plant_correct[k] / plant_total[k], "n": plant_total[k]}
            for k in plant_total
        ])
        bee_df = pd.DataFrame([
            {"bee_id": k, "acc": bee_correct[k] / bee_total[k], "n": bee_total[k]}
            for k in bee_total
        ])
        plant_df["plant_name"] = plant_df["plant_id"].map(self.plant_names)
        bee_df["bee_name"] = bee_df["bee_id"].map(self.bee_names)

        plant_df.to_csv(os.path.join(log_dir, f"val_plants_epoch{self.current_epoch}.csv"), index=False)
        bee_df.to_csv(os.path.join(log_dir, f"val_bees_epoch{self.current_epoch}.csv"), index=False)

        # aggregate predictions across validation
        plant_ids_all, bee_ids_all, probs_all = [], [], []
        for out in self.val_outputs:
            logits = F.softmax(out["logits"], dim=1)  # [batch, 2]
            probs = logits[:, 1]  # probability of "interaction"
            plant_ids_all.extend(out["plants"].tolist())
            bee_ids_all.extend(out["bees"].tolist())
            probs_all.extend(probs.cpu().tolist())

        # --- Convert to long-form dataframe ---
        df = pd.DataFrame({
            "plant_id": plant_ids_all,
            "bee_id": bee_ids_all,
            "interaction_prob": probs_all,
        })
        df["plant_name"] = df["plant_id"].map(self.plant_names)
        df["bee_name"] = df["bee_id"].map(self.bee_names)
      

        # Reorder columns
        df = df[["bee_id", "plant_id", "bee_name", "plant_name", "interaction_prob"]]

        # --- Save CSV ---
        log_dir = os.path.join(self.logger.log_dir, "species_probs")
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, f"val_probs_epoch{self.current_epoch}.csv")
        df.to_csv(csv_path, index=False)

        print(f"[Epoch {self.current_epoch}] Wrote interaction probabilities to {csv_path}")

        self.val_outputs.clear()     

    def configure_optimizers(self):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = itertools.chain(
            self.embedding_model.parameters(),
            self.attn_pool.parameters(),
            self.interaction_head.parameters()
        )
        return torch.optim.AdamW([
            {"params": backbone_params, "lr": 1e-5},
            {"params": head_params, "lr": self.hparams.lr},
        ])


def convert_keys_to_int(obj):
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            try:
                new_obj[int(k)] = v
            except ValueError:
                new_obj[k] = v  # Keep as string if not convertible to int
        return new_obj
    return obj

# -----------------------
#   Main Training
# -----------------------
def main(args):
    base_path = os.path.join("/scratch", "mcatchen", "iNatImages", "data") if args.cluster else "./data"
    log_path = os.path.join(base_path, "logs")
    metaweb_path = os.path.join(base_path, "interactions.csv")
    
    metaweb = load_metaweb(metaweb_path).long()


    with open(os.path.join(base_path, 'plant_labels.json')) as json_file:
        plant_names = json.load(json_file, object_hook=convert_keys_to_int) 
    with open(os.path.join(base_path, 'bee_labels.json')) as json_file:
        bee_names = json.load(json_file, object_hook=convert_keys_to_int)

    num_plants, num_bees = metaweb.shape
    mask_maker = ZeroShotMaskMaker(num_plants, num_bees, holdout_frac=0.2, seed=42)

    decoder = SamplerDecoder()
    plant_dataset = (
        wds.WebDataset(glob.glob(os.path.join(base_path, "plant_wds", "train-*.tar")), shardshuffle=True)
        .decode().to_tuple("jpg", "json").map(decoder)
    )
    bee_dataset = (
        wds.WebDataset(glob.glob(os.path.join(base_path, "bombus_wds", "train-*.tar")), shardshuffle=True)
        .decode().to_tuple("jpg", "json").map(decoder)
    )

    train_dataset = PairedBagDataset(plant_dataset, bee_dataset, metaweb, mask_maker, split="train", max_instances=args.max_instances)
    val_dataset   = PairedBagDataset(plant_dataset, bee_dataset, metaweb, mask_maker, split="val", max_instances=args.max_instances)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn_bags)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn_bags)

    model = VitInteractionClassifierMIL(
        bee_names, 
        plant_names, 
        lr=args.lr, 
        model_type=args.model,
        num_unfrozen_vit_layers = args.num_unfrozen_vit_layers
    )

    logger = CSVLogger(log_path, name=os.environ.get("SLURM_JOB_NAME") or "InteractionTest")

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
        accelerator = accelerator,
        devices = devices,
        strategy = strategy,
        logger = logger,
        max_epochs = args.max_epochs,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--max_instances', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_bee_holdouts', type=int, default=2)
    parser.add_argument('--num_plant_holdouts', type=int, default=10)
    parser.add_argument('--num_unfrozen_vit_layers', type=int, default=2)
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--persistent_workers', action='store_true')
    parser.add_argument('--model', default='base', choices=['base', 'large', 'huge'])


    args = parser.parse_args()
    main(args)

i = iter(train_dataset)
x = next(i)