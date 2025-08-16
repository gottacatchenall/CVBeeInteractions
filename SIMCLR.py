import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

import pytorch_lightning as pl
import torchmetrics
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from src.simclr_transforms import simclr_transforms 
import torchvision.transforms as transforms

# -------------------
# Dataset and Data Module
# -------------------
class TorchSavedDataset(Dataset):
    def __init__(self, file_path, train=True):
        if "dataset.pt" not in file_path:
            file_path = os.path.join(file_path, "dataset.pt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        print(f"Loading dataset from: {file_path}")
        dataset_dict = torch.load(file_path)
        if 'data' not in dataset_dict or 'labels' not in dataset_dict:
            raise ValueError("The .pt file must contain 'data' and 'labels' keys.")
        idx = dataset_dict['train_indices'] if train else dataset_dict['test_indices']
        self.data = dataset_dict['data'].index_select(0, idx)
        self.labels = dataset_dict['labels'].index_select(0, idx)
        self.classes = dataset_dict['classes']
        if self.data.shape[0] != self.labels.shape[0]:
            raise ValueError("Number of samples in data and labels do not match.")
        self.num_samples = self.data.shape[0]
        print(f"Successfully loaded {self.num_samples} samples.")
       
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        return image, label
    
class TransformedDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        view1 = self.transform(img)
        view2 = self.transform(img)
        return (view1, view2), label
    
class SimCLRDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./", batch_size=128, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = simclr_transforms(size=224)
        self.val_transforms = transforms.Compose([])

    def prepare_data(self):
        pass 

    def setup(self, stage):
        if stage == "fit":
            full_dataset = TorchSavedDataset(self.data_dir, train=True)
            self.train_dataset = TransformedDataset(full_dataset, self.train_transforms)
            self.val_dataset = TransformedDataset(full_dataset, self.val_transforms)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True
        )

# -------------------
# SimCLR Lightning Module
# -------------------
class SimCLR(pl.LightningModule):
    def __init__(self, lr=1e-3, temperature=0.5):
        super().__init__()
        self.lr = lr
        self.temperature = temperature
        
        # 1. Base Encoder (ViT)
        self.encoder = AutoModel.from_pretrained(
            "google/vit-base-patch16-224",
            local_files_only=True
        )
        
        # 2. Embedding Layer: New addition for 256-dim embedding
        self.embedding_layer = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        # 3. Projection Head: Now starts with 256 input
        self.projection_head = nn.Sequential(
            nn.Linear(128, 128),
        )
        
    def forward(self, x):
        # Forward pass through the encoder, embedding layer, and projection head
        x = self.encoder(x).pooler_output
        z = self.embedding_layer(x) # Pass through the new embedding layer
        z = self.projection_head(z)
        return z

    def info_nce_loss(self, z1, z2):
        N, D = z1.shape
        # 1. Normalize so cosine similarity = dot product ---------------------------------
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # 2. Concat views 
        features = torch.cat([z1, z2], dim=0) 

        #3. Pairwise similarities  
        sim = features @ features.T  # [2N, 2N]

        # 4. Temperature scaling
        logits = sim / self.temperature  # [2N, 2N]

        # 5. Compute positives
        device = features.device
        idx = torch.arange(2 * N, device=device)
        pos_idx = (idx + N) % (2 * N)
        pos_logits = logits[idx, pos_idx]

        # 6. Exclude self-similarities 
        logits = logits.clone()
        logits.fill_diagonal_(float("-inf"))

        # 7. Log-softmax 
        log_den = torch.logsumexp(logits, dim=1)    

        # 8. InfoNCE loss = -mean(log p(pos | i))
        loss = -(pos_logits - log_den).mean()

        # Debug NaNs
        if torch.isnan(loss):
            raise ValueError(
                f"NaN in loss!"
                f"pos_logits range=({pos_logits.min().item()}, {pos_logits.max().item()}), "
                f"log_den range=({log_den.min().item()}, {log_den.max().item()}"
        )
        return loss


    def training_step(self, batch, batch_idx):
        (x1, x2), _ = batch
        z1 = self(x1)
        z2 = self(x2)
        loss = self.info_nce_loss(z1, z2)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (x1, x2), _ = batch
        z1 = self(x1)
        z2 = self(x2)
        val_loss = self.info_nce_loss(z1, z2)
        self.log("val_loss", val_loss, prog_bar=True)
            
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def configure_gradient_clipping(
        self,
        optimizer, 
        optimizer_idx,
        gradient_clip_val=1.0, 
        gradient_clip_algorithm="norm"
    ):
        self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm)


def main(image_dir, args):
    species_data = SimCLRDataModule(
        data_dir = image_dir,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
    )

    net = SimCLR(
        lr=args.lr
    )

    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    gpus = torch.cuda.device_count()

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        strategy='ddp',
        profiler="simple",
        num_nodes=num_nodes, 
        max_epochs=args.max_epochs,
        enable_progress_bar=False,
    )

    trainer.fit(net, species_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--species', default='bees', choices=['plants', 'bees'])
    args = parser.parse_args()

    base_path = os.path.join("/scratch", "mcatchen", "iNatImages", "data") if args.cluster else "./data"
    data_dir = "binary_bees" if args.species == "bees" else "binary_plants"

    image_dir = os.path.join(base_path, data_dir)

    main(image_dir, args)



