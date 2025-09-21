import torch
import os
import argparse

import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import torch.nn.functional as F
import kornia.augmentation as K


from torch.utils.data import DataLoader
from transformers import AutoModel
from collections import defaultdict

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
)

from torchmetrics.functional.classification import binary_auroc, binary_average_precision, binary_accuracy


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        x: [B, N, D]  (batch, num_instances, feature_dim)
        returns: [B, D]
        """
        attn_scores = self.attn(x)                          # [B, N, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)    # [B, N, 1]
        pooled = torch.sum(attn_weights * x, dim=1)         # [B, D]
        return pooled


class InteractionPredictor(pl.LightningModule):
    def __init__(
        self, 
        lr=3e-4,
        model_type="base",
        embed_dim = 128,
        attention_hidden_dim = 128,
        num_vit_unfrozen_layers = 1
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
        # Unfreeze last transformer block
        for param in self.backbone.layer[-1*num_vit_unfrozen_layers:].parameters():
            param.requires_grad = True

        backbone_dims = {"base": 768, "large": 1024, "huge": 1280}
        self.backbone_dim = backbone_dims[model_type]
        self.embed_dim = embed_dim

        # ---- Attention pooling ----
        self.attn_pool_b = AttentionPooling(self.backbone_dim, attention_hidden_dim)
        self.attn_pool_p = AttentionPooling(self.backbone_dim, attention_hidden_dim)

         # ---- Embedding head ----
        self.embedding_model = nn.Sequential(
            nn.Linear(2*self.backbone_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

        # ---- Final classifier ----
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.lr = lr
        self.loss_fn = nn.BCEWithLogitsLoss()

        # ---- TorchMetrics ----
        self.val_acc = BinaryAccuracy()
        self.val_auc = BinaryAUROC()
        self.val_ap = BinaryAveragePrecision()

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
        #pooled_b_e = torch.mean(b_e, dim=1)
        #pooled_p_e = torch.mean(p_e, dim=1)

        # Attention pooling
        pooled_b_e = self.attn_pool_b(b_e)   # [B, D]
        pooled_p_e = self.attn_pool_p(p_e)   # [B, D]

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
        # update torchmetrics
        self.val_acc.update(logits, labels.int())
        self.val_auc.update(logits, labels.int())
        self.val_ap.update(logits, labels.int())

        self.log("val/loss", loss, prog_bar=False, on_epoch=True)

        # keep for per-species logging
        self.val_outputs.append({
            "loss": loss.detach(),
            "preds": preds.cpu(),
            "logits": logits.detach().cpu(),
            "labels": labels.cpu(),
            "plants": plant_name,
            "bees": bee_name,
        })

    def on_validation_epoch_end(self):
        # --- Global metrics ---

        acc = self.val_acc.compute()
        auc = self.val_auc.compute()
        ap = self.val_ap.compute()
        self.log("val/accuracy", acc, prog_bar=True)
        self.log("val/rocauc", auc)
        self.log("val/avg_precision", ap)
        
        # --- Collect per-species data ---
        plant_preds, plant_labels = defaultdict(list), defaultdict(list)
        bee_preds, bee_labels = defaultdict(list), defaultdict(list)

        for out in self.val_outputs:
            preds, labels = out["preds"], out["labels"]
            plants, bees = out["plants"], out["bees"]
            for p, b, y, yhat in zip(plants, bees, labels, preds):
                plant_preds[p].append(yhat.item())
                plant_labels[p].append(y.item())
                bee_preds[b].append(yhat.item())
                bee_labels[b].append(y.item())

        # --- Per-species metrics ---
        plant_df = []
        for k in plant_preds:
            yhat = torch.tensor(plant_preds[k])
            y = torch.tensor(plant_labels[k]).int()
            acc = ((yhat >= 0.5).int() == y).float().mean().item()
            auc = binary_auroc(yhat, y) if len(y.unique()) > 1 else torch.nan
            ap = binary_average_precision(yhat, y) if len(y.unique()) > 1 else torch.nan
            plant_df.append({"plant_id": k, "acc": acc, "auc": auc,
                             "ap": ap, "n": len(y)})

        bee_df = []
        for k in bee_preds:
            yhat = torch.tensor(bee_preds[k])
            y = torch.tensor(bee_labels[k]).int()
            acc = binary_accuracy(yhat, y)
            auc = binary_auroc(yhat, y) if len(y.unique()) > 1 else torch.tensor(torch.nan)
            ap = binary_average_precision(yhat, y) if len(y.unique()) > 1 else torch.tensor(torch.nan)
            bee_df.append({"bee_id": k, "acc": acc.item(), "auc": auc.item(),
                           "ap": ap.item(), "n": len(y)})

        # log means
        if plant_df:
            self.log("val_plant_acc_mean", np.nanmean([d["acc"] for d in plant_df]))
            self.log("val_plant_auc_mean", np.nanmean([d["auc"] for d in plant_df]))
            self.log("val_plant_ap_mean", np.nanmean([d["ap"] for d in plant_df]))
        if bee_df:
            self.log("val_bee_acc_mean", np.nanmean([d["acc"] for d in bee_df]))
            self.log("val_bee_auc_mean", np.nanmean([d["auc"] for d in bee_df]))
            self.log("val_bee_ap_mean", np.nanmean([d["ap"] for d in bee_df]))

        # --- Save per-species CSVs ---
        log_dir = os.path.join(self.logger.log_dir, "per_species")
        os.makedirs(log_dir, exist_ok=True)
        pd.DataFrame(plant_df).to_csv(os.path.join(log_dir, f"val_plants_epoch{self.current_epoch}.csv"), index=False)
        pd.DataFrame(bee_df).to_csv(os.path.join(log_dir, f"val_bees_epoch{self.current_epoch}.csv"), index=False)

        # --- Aggregate prediction probabilities ---
        plant_ids_all, bee_ids_all, probs_all, preds_all = [], [], [], []
        for out in self.val_outputs:
            probs = torch.sigmoid(out["logits"])
            preds = (probs >= 0.5).int()
            plant_ids_all.extend(out["plants"])
            bee_ids_all.extend(out["bees"])
            probs_all.extend(probs.cpu().tolist())
            preds_all.extend(preds.cpu().tolist())

        df = pd.DataFrame({
            "plant_id": plant_ids_all,
            "bee_id": bee_ids_all,
            "interaction_prob": probs_all,
            "interaction_pred": preds_all,
        })
        df = df[["bee_id", "plant_id", "interaction_prob", "interaction_pred"]]

        log_dir = os.path.join(self.logger.log_dir, "species_probs")
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, f"val_probs_epoch{self.current_epoch}.csv")
        df.to_csv(csv_path, index=False)

        print(f"[Epoch {self.current_epoch}] Wrote interaction probabilities to {csv_path}")

        # reset for next epoch
        self.val_acc.reset()
        self.val_auc.reset()
        self.val_ap.reset()

        self.val_outputs.clear()

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optim





