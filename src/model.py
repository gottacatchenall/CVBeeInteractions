import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

import pytorch_lightning as pl

# -------------------
# Lightning Module
# -------------------
class VitClassifier(pl.LightningModule):
    def __init__(self, lr = 1e-3):
        super().__init__()
        self.lr = lr
        self.image_model = AutoModel.from_pretrained(
            "google/vit-base-patch16-224",
            local_files_only=True
        )
        self.embedding_model = nn.Sequential(
            nn.Linear(768, 64)
        )
        self.classification_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 19)
        )
    def forward(self, x):
        x = self.image_model(x).pooler_output
        x = self.embedding_model(x)
        x = self.classification_head(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

