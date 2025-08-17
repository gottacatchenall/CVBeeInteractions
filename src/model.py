import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

import pytorch_lightning as pl
import torchmetrics


# -------------------
# Lightning Module
# -------------------
class VitClassifier(pl.LightningModule):
    def __init__(self, lr = 1e-3, gamma=0.95, num_classes=19):
        super().__init__()
        self.save_hyperparameters()
        self.image_model = AutoModel.from_pretrained(
            "google/vit-base-patch16-224",
            local_files_only=True
        )
        # --- Freeze ViT layers  ---
        for param in self.image_model.embeddings.parameters():
            param.requires_grad = False
        for param in self.image_model.encoder.parameters():
            param.requires_grad = False
        for param in self.image_model.pooler.parameters():
            param.requires_grad = True
        
        self.embedding_model = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(32, num_classes)
        )

        self.train_metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.classification.Accuracy(
                    task="multiclass", 
                    num_classes=num_classes
                ),
                "MAP_macro": torchmetrics.classification.MulticlassPrecision(num_classes=num_classes, average='macro'),
                "MAP_micro": torchmetrics.classification.MulticlassPrecision(num_classes=num_classes, average='micro'),
            },
            prefix="train_",
        )
        self.valid_metrics = self.train_metrics.clone(prefix="valid_")

    def forward(self, x):
        x = self.image_model(x).pooler_output
        x = self.embedding_model(x)
        x = self.classification_head(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=False)
        with torch.no_grad():
            batch_value = self.train_metrics(y_hat, y)
            self.log_dict(batch_value)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        with torch.no_grad():
            batch_value = self.valid_metrics(y_hat, y)
            val_loss = F.cross_entropy(y_hat, y)
            batch_value["valid_loss"] = val_loss
            self.log_dict(batch_value)
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        # ExponentialLR scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.hparams.gamma
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",   # or "step"
                "frequency": 1,
            },
        }


