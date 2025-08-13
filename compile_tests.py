import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoFeatureExtractor
from transformers import AutoModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import os
import pandas as pd
import numpy as np
import argparse

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, ResNetForImageClassification
import torchmetrics

import argparse
import time
import os

parser = argparse.ArgumentParser(description='Cropping iNaturalist Images with Zero-Shot Object Detection')
parser.add_argument('--cluster', action='store_true')
parser.add_argument('--num_workers', type=int, default=0, help='')

args = parser.parse_args()
 
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

torch.set_float32_matmul_precision('medium')


class ViTSpeciesEmbeddingModel(nn.Module):
    def __init__(
        self, 
        model,
        num_classes=19,
        species_embedding_dim = 128, 
    ):
        super().__init__()
        self.num_classes = num_classes
        self.species_embedding_dim = species_embedding_dim
        self.image_model = model
        self.embedding_model = nn.Linear(
            768, 
            species_embedding_dim
        )
        self.classification_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(species_embedding_dim, num_classes) 
        )
    def forward(self, x):
        return self.classification_head(self.embedding_model(self.image_model(x).pooler_output))
    


def test_stats(num_classes, all_labels, all_probs, test_loss):
    stats = {
        "test_loss": test_loss,
    }
    metrics = {
        "AUROC": torchmetrics.AUROC(task="multiclass", num_classes=num_classes).to(device),
        "MAP": torchmetrics.AveragePrecision(task="multiclass", num_classes=num_classes).to(device), 
        "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    }

    for k in metrics.keys():
        stats[k] = metrics[k](all_probs, all_labels).item()
    return stats

vit = AutoModel.from_pretrained("google/vit-base-patch16-224", local_files_only=True)


model = ViTSpeciesEmbeddingModel(vit).to(device)
for param in model.image_model.encoder.parameters():
    param.requires_grad = False
for param in model.image_model.embeddings.parameters():
    param.requires_grad = False


starttime = time.time()
model = torch.compile(
    model,
    mode="reduce-overhead",  # minimal fusion, fast compile
    backend="aot_eager"
)

print(f"Compile time: {time.time() - starttime} seconds")


# load data
base_path = os.path.join("/scratch", "mcatchen", "iNatImages") if args.cluster else "./"
img_dir  = os.path.join(base_path, "data", "bombus_img")   
feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224", local_files_only=True, use_fast=True)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=feature_extractor.image_mean, 
        std=feature_extractor.image_std),
])
full_dataset = datasets.ImageFolder(
    img_dir,
    transform=image_transform,
)



learning_rate = 3e-4
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
pin_mem = torch.cuda.is_available()

# dummy batch
starttime = time.time()
dummy = torch.randn(2, 3, 224, 224).to(device)
e = model(dummy)
print(f"Dummy batch time: {time.time() - starttime} seconds")


for bs in [64, 128, 512, 1024]:
    # setup loader
    full_loader = DataLoader(full_dataset, batch_size=bs, shuffle=True, pin_memory=pin_mem)

    starttime = time.time()
    images, labels = next(iter(full_loader))
    images, labels = images.to(device), labels.to(device)
    logits = model(images)
    loss = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    batch_time = time.time() - starttime
    batch_time*len(full_loader)
    print(f"Batch Size {bs} time: {batch_time} seconds -- Est Epoch Time:    {batch_time*len(full_loader)}")


