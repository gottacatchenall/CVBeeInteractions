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
import time 

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, ResNetForImageClassification
import torchmetrics

torch.set_float32_matmul_precision('high')

class ResNetSpeciesEmbeddingModel(nn.Module):
    def __init__(
        self, 
        model,
        species_embedding_dim=64, 
        num_classes=19,
        batch_size = 128,
    ):
        super().__init__()
        self.backbone = model.resnet
        self.embedding_model = nn.Sequential(
            nn.Linear(2048, species_embedding_dim) 
        )
        self.classification_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(species_embedding_dim, num_classes) 
        )   

        # --- Freeze CNN layers  ---
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.embedding_model.parameters():
            param.requires_grad = True
        for param in self.classification_head.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x, output_hidden_states=True).pooler_output.squeeze()
        x = self.embedding_model(x)
        x = self.classification_head(x)
        return x


parser = argparse.ArgumentParser(description='resnet, single gpu performance test')
parser.add_argument('--lr', default=1e-3, type=float, help='')
parser.add_argument('--batch_size', type=int, default=512, help='')
parser.add_argument('--num_epoch', type=int, default=2, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')
parser.add_argument('--cluster', action='store_true')


def main():
    args = parser.parse_args()
    base_path = os.path.join("/scratch", "mcatchen", "iNatImages") if args.cluster else "./"


    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    model_name = "microsoft/resnet-50"
    processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=True, use_fast=True)
    resnet = ResNetForImageClassification.from_pretrained(model_name, local_files_only=True)


    # Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    batch_size = args.batch_size
    img_dir  = os.path.join(base_path, "data", "bombus_img")
    full_dataset = datasets.ImageFolder(
        img_dir, 
        transform=transform
    )
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])


    # Loaders
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_mem, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=pin_mem, num_workers=args.num_workers)

    # Model init
    rnsem = ResNetSpeciesEmbeddingModel(resnet)

    compilestart = time.time()
    if torch.cuda.is_available():
        rnsem = torch.compile(rnsem)
    compileend = time.time()

    print(f"Compile Time: {compileend - compilestart} seconds")

    rnsem = rnsem.to(device) # Load model on the GPU


    # Loss 
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(rnsem.parameters(), lr=args.lr)

    perf = []

    total_start = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        start = time.time()
        
        inputs = inputs.to(device) 
        targets = targets.to(device) 

        outputs = rnsem(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - start
        images_per_sec = args.batch_size/batch_time
        perf.append(images_per_sec)

    total_time = time.time() - total_start

    print(f"Total time: {total_time}")
    print(f"Mean images per sec first iter: {perf[0]}")
    print(f"Mean images per sec subseq iter: {sum(perf[1:])/len(perf[1:])}")

if __name__=='__main__':
   main()

