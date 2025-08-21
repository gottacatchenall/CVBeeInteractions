import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModel
from src.checkpoints import AsyncTrainableCheckpoint 

import pytorch_lightning as pl
import torchmetrics
import kornia.augmentation as K

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

class VitClassifier(pl.LightningModule):
    def __init__(
            self, 
            lr = 1e-3, 
            min_crop_size = 0.5,
            embedding_dim=128,
            num_classes=19,
            augmentation = True, 
            model_type="base"
    ):
        super().__init__()
        self.save_hyperparameters()

        model_name = model_paths()[model_type]
        self.image_model = AutoModel.from_pretrained(
            model_name, 
            local_files_only=True
        )
        # --- Freeze ViT layers  ---
        for param in self.image_model.parameters():
            param.requires_grad = False

        image_model_output_dim = image_embed_dim()[model_type]
        self.embedding_model = nn.Sequential(
            nn.Linear(image_model_output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.hparams.embedding_dim),
        )

        self.classification_head = nn.Sequential(
            nn.Linear(self.hparams.embedding_dim, self.hparams.num_classes)
        )

        self.transform = torch.nn.Sequential(
            K.RandomResizedCrop((224,224), scale=(min_crop_size, 1.0)),
            K.RandomHorizontalFlip(),
            K.ColorJitter(0.4,0.4,0.4,0.1, p=0.8),
            K.RandomGrayscale(p=0.2),
            K.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ).to("cuda")

        self.criterion_ce = nn.CrossEntropyLoss()

        self.train_metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.classification.Accuracy(
                    task="multiclass", 
                    num_classes=self.hparams.num_classes
                ),
                "MAP": torchmetrics.classification.MulticlassPrecision(num_classes=self.hparams.num_classes, average='macro'),
            },
            prefix="train_",
        )
        self.valid_metrics = self.train_metrics.clone(prefix="valid_")

    def forward(self, x):
        with torch.no_grad():
            x =  self.image_model(x).pooler_output
        x = self.embedding_model(x)
        x = self.classification_head(x)
        return x

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        if self.hparams.augmentation:
                x = self.transform(x)
        else:
            x = self.resize(x)
        return x, y
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion_ce(logits, y)
        with torch.no_grad():
            batch_value = self.train_metrics(logits, y)
            self.log_dict(batch_value)
            self.log("train_loss", loss, prog_bar=False)    
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
        return {
            "optimizer": optimizer,
        }

