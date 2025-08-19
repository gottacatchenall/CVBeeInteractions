import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
from torchvision.transforms import v2

from src.supervised_contrastive import SupConLoss
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


# -------------------
# Lightning Module
# -------------------
class VitClassifier(pl.LightningModule):
    def __init__(
            self, 
            lr = 1e-3, 
            gamma=0.95, 
            T_0 = 20,
            T_mult = 1,
            eta_min = 1e-4,
            temperature=0.07,
            use_supcon=True,
            alpha=0.5,  # weight for combining SupCon + CE losses
            num_classes=19, 
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
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(32, num_classes)
        )
        self.transform = torch.nn.Sequential(
            K.RandomResizedCrop((224,224), scale=(0.2,1.0)),
            K.RandomHorizontalFlip(),
            K.ColorJitter(0.4,0.4,0.4,0.1, p=0.8),
            K.RandomGrayscale(p=0.2),
            K.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ).to("cuda")

        # --- Losses ---
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_supcon = SupConLoss(temperature=temperature)

        # --- Training options ---
        self.use_supcon = use_supcon  # whether to use contrastive loss
        self.alpha = alpha            # mixing factor between CE & SupCon


        self.train_metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.classification.Accuracy(
                    task="multiclass", 
                    num_classes=num_classes
                ),
                "MAP": torchmetrics.classification.MulticlassPrecision(num_classes=num_classes, average='macro'),
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
        if self.trainer.training and self.use_supcon:
            x1 = self.transform(x)  # GPU augment
            x2 = self.transform(x)
            x = torch.stack([x1, x2], dim=1)  # [bsz, 2, C, H, W]
        elif self.trainer.training:
            x = self.transform(x)
        return x, y
    
    def training_step(self, batch, batch_idx):
        """
        Training step: supports three modes
        - CE only
        - SupCon only
        - Joint CE + SupCon
        """
        x, y = batch

        if self.use_supcon:
            # --- Case: contrastive training ---
            bsz, n_views, C, H, W = x.shape
            x = x.view(bsz * n_views, C, H, W)

            
            img_embed = self.image_model(x).pooler_output
            
            # TODO: there should be a projective head here 
            embeddings = self.embedding_model(img_embed)

            logits = self.classification_head(x) 
            
            embeddings = embeddings.view(bsz, n_views, -1)  # [bsz, n_views, dim]


            # Contrastive loss
            supcon_loss = self.criterion_supcon(embeddings, labels=y)

            # Optional CE loss (on first view only)
            ce_loss = self.criterion_ce(
                logits[:bsz], y
            )  # take first view’s logits

            # Combine losses
            loss = self.alpha * supcon_loss + (1 - self.alpha) * ce_loss

            self.log("train_supcon_loss", supcon_loss, prog_bar=False)
            self.log("train_ce_loss", ce_loss, prog_bar=False)
            with torch.no_grad():
                batch_value = self.train_metrics(logits[:bsz], y)
                self.log_dict(batch_value)
        else:
            # --- Case: normal crossentropy training only ---
            logits = self(x)
            loss = self.criterion_ce(logits, y)
            with torch.no_grad():
                batch_value = self.train_metrics(loss, y)
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
        """
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = self.hparams.T_0,
            T_mult = self.hparams.T_mult,
            eta_min = self.hparams.eta_min
        )
        
                "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",   # or "step"
                "frequency": 1,
            },
        """
        return {
            "optimizer": optimizer,
        }


"""
from src.model import VitClassifier
from src.dataset import WebDatasetDataModule
import src.dataset
species_data = WebDatasetDataModule(
    data_dir = "./data/bombus_wds",
)
species_data.setup('fit')
dl = species_data.train_dataloader()
net = VitClassifier(
        num_classes=19,
        model_type = "large"
)        
x,y = next(iter(dl))
x = vit.image_model(x).pooler_output
x = vit.embedding_model(x)
x = vit.classification_head(x)
yhat = vit(x)
yhat.shape, y.shape
"""