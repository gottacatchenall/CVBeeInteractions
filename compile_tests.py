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
args = parser.parse_args()
 
class ViTSpeciesEmbeddingModel(nn.Module):
    def __init__(self, 
        num_classes=19,
        species_embedding_dim = 128, 
        batch_size = 256,
    ):

        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.species_embedding_dim = species_embedding_dim

        self.metrics = {
            "AUROC": torchmetrics.AUROC(task="multiclass", num_classes=self.num_classes).to(self.device),
            "MAP": torchmetrics.AveragePrecision(task="multiclass", num_classes=self.num_classes).to(self.device), 
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)
        }

        self.feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224", local_files_only=True, use_fast=True)

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.feature_extractor.image_mean, 
                std=self.feature_extractor.image_std),
        ])

        self.image_model = AutoModel.from_pretrained("google/vit-base-patch16-224", local_files_only=True)
        self.image_model.to(self.device)

        for param in self.image_model.encoder.parameters():
            param.requires_grad = False
        for param in self.image_model.embeddings.parameters():
            param.requires_grad = False
       
        self.embedding_model = nn.Linear(
            768, 
            species_embedding_dim
        )

        self.classification_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(species_embedding_dim, num_classes) 
        )

    def embed_image(self, x):
        return self.image_model(x).pooler_output
    
    def prepare_train(self):
        self.image_model.pooler.train()
        self.embedding_model.train()
        self.classification_head.train()

    def test_stats(self, all_labels, all_probs, test_loss):
        stats = {
            "test_loss": test_loss,
        }
        for k in self.metrics.keys():
            stats[k] = self.metrics[k](all_probs, all_labels).item()
        return stats
    def forward(self, x):
        return self.classification_head(self.embedding_model(x))
    

model = ViTSpeciesEmbeddingModel().cuda()

starttime = time.time()
model = torch.compile(
    model,
    mode="reduce-overhead",  # minimal fusion, fast compile
    backend="aot_eager"
)

print(f"Compile time: {time.time() - starttime} seconds")

starttime = time.time()
dummy = torch.randn(2, 3, 224, 224, device="cuda")
e = model.image_model(dummy).pooler_output
model.forward(e)
print(f"Dummy batch time: {time.time() - starttime} seconds")


base_path = os.path.join("/scratch", "mcatchen", "iNatImages") if args.cluster else "./"
img_dir  = os.path.join(base_path, "data", "bombus_img")   

full_dataset = datasets.ImageFolder(
    img_dir,
    transform=model.image_transform,
)

pin_mem = torch.cuda.is_available()
full_loader = DataLoader(full_dataset, batch_size=model.batch_size, shuffle=True, pin_memory=pin_mem)

# first epoch 
learning_rate = 3e-4
starttime = time.time()
criterion = nn.CrossEntropyLoss().to(model.device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.prepare_train()
for images, labels in full_loader:
    images, labels = images.to(model.device), labels.to(model.device)
    with torch.no_grad():
        outputs = model.embed_image(images) 

    logits = model(outputs)
    loss = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"First epoch time: {time.time() - starttime} seconds")
