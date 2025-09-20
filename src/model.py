import torch
import os
import argparse

import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import torch.nn.functional as F

from torch.utils.data import DataLoader
from transformers import AutoModel
from collections import defaultdict


class InteractionPredictor(pl.LightningModule):
    def __init__(
        self, 
        lr=3e-4,
        model_type="base",
        embed_dim = 128
    ):
        super().__init__()

        # ---- Setup Backbone --- #
        model_paths = {
            "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "huge": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        }
        self.backbone = AutoModel.from_pretrained(model_paths[model_type], local_files_only=True)
        for p in self.backbone.parameters():
            p.requires_grad = False

        backbone_dims = {"base": 768, "large": 1024, "huge": 1280}
        self.backbone_dim = backbone_dims[model_type]
        self.embed_dim = embed_dim


         # ---- Embedding head ----
        self.embedding_model = nn.Sequential(
            nn.Linear(2*self.backbone_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )
        # ---- Final classifier ----
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 1)
        )

        self.lr = lr
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.val_outputs = []

    def forward(self, plant_bag, bee_bag):
        """
        plant_bag: [N, B, C, H, W]
        bee_bag:   [N, B, C, H, W]
        """
        B, N, C, H, W = plant_bag.shape

        # Merge bags into one batch for the backbone

        b = plant_bag.view(B * N, C, H, W)
        p = bee_bag.view(B * N, C, H, W)

        # ViT forward 
        b_e = self.backbone(b).pooler_output
        p_e = self.backbone(p).pooler_output

        # Reshape back into bags
        b_e = b_e.view(B, N, self.backbone_dim)
        p_e = p_e.view(B, N, self.backbone_dim)

        # Mean pooling 
        pooled_b_e = torch.mean(b_e, dim=1)
        pooled_p_e = torch.mean(p_e, dim=1)

        # Concat bee and plant features
        e = torch.concat([pooled_b_e, pooled_p_e], dim=1)

        # Reduce dim
        e2 = self.embedding_model(e)

        # Final prediction
        logits = self.head(e2).flatten()
        return logits

    def shared_step(self, batch, batch_idx):
        # batch is a list/tuple of N items from PairedBagDataset, but when using DataLoader with batch_size=N,
        # a collated batch will have each field stacked automatically.

        plant_name, bee_name, plant_bags, bee_bags, labels = batch

        logits = self.forward(plant_bags, bee_bags)
        labels = labels.float()
        loss = self.loss_fn(logits, labels)
        preds = torch.sigmoid(logits)
        return plant_name, bee_name, loss, logits, preds, labels

    def training_step(self, batch, batch_idx):
        _, _, loss, _, _, _ = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        plant_name, bee_name, loss, logits, preds, labels = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=False, on_epoch=True)
        self.val_outputs.append({
            "loss": loss.detach(),
            "preds": preds.cpu(),
            "logits": logits.detach().cpu(),
            "labels": labels.cpu(),
            "plants": plant_name,
            "bees": bee_name,
        })

    def on_validation_epoch_end(self):
        # aggregate per-species accuracy
        plant_correct, plant_total = defaultdict(int), defaultdict(int)
        bee_correct, bee_total = defaultdict(int), defaultdict(int)

        for out in self.val_outputs:
            preds, labels = out["preds"], out["labels"]
            plants, bees = out["plants"], out["bees"]
            for p, b, y, yhat in zip(plants, bees, labels, preds):
                correct = int(y == yhat)
                plant_total[p] += 1
                bee_total[b] += 1
                plant_correct[p] += correct
                bee_correct[b] += correct

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

        plant_df.to_csv(os.path.join(log_dir, f"val_plants_epoch{self.current_epoch}.csv"), index=False)
        bee_df.to_csv(os.path.join(log_dir, f"val_bees_epoch{self.current_epoch}.csv"), index=False)

        # aggregate predictions across validation
        plant_ids_all, bee_ids_all, probs_all = [], [], []
        for out in self.val_outputs:
            logits = F.softmax(out["logits"], dim=0)  # [batch, 2]
            probs = logits #logits[:, 1]  # probability of "interaction"
            plant_ids_all.extend(out["plants"])
            bee_ids_all.extend(out["bees"])
            probs_all.extend(probs.cpu().tolist())

        # --- Convert to long-form dataframe ---
        df = pd.DataFrame({
            "plant_id": plant_ids_all,
            "bee_id": bee_ids_all,
            "interaction_prob": probs_all,
        })
      

        # Reorder columns
        df = df[["bee_id", "plant_id", "interaction_prob"]]

        # --- Save CSV ---
        log_dir = os.path.join(self.logger.log_dir, "species_probs")
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, f"val_probs_epoch{self.current_epoch}.csv")
        df.to_csv(csv_path, index=False)

        print(f"[Epoch {self.current_epoch}] Wrote interaction probabilities to {csv_path}")

        self.val_outputs.clear()  

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optim





