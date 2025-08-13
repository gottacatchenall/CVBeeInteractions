import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoModelForImageClassification
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import os
import pandas as pd
import numpy as np
import argparse
import time 

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, AutoModel
import torchmetrics

torch.set_float32_matmul_precision('high')


class SpeciesEmbeddingModel(nn.Module):
    def __init__(
        self, 
        model,
        species_embedding_dim=64, 
        num_classes=19,
        batch_size = 128,
    ):
        super().__init__()
        if model.base_model_prefix == 'vit':
            self.backbone = model
            self.image_dim = 768
        elif model.base_model_prefix == 'resnet':
            self.backbone = model.resnet
            self.image_dim = 2048
        self.embedding_model = nn.Sequential(
            nn.Linear(self.image_dim, species_embedding_dim) 
        )
        self.classification_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(species_embedding_dim, num_classes) 
        )   

        # --- Freeze CNN layers  ---
        if self.backbone.base_model_prefix == 'vit':
            for param in self.backbone.vit.embeddings.parameters():
                param.requires_grad = False
            for param in self.backbone.vit.encoder.parameters():
                param.requires_grad = False
            for param in self.embedding_model.parameters():
                param.requires_grad = True
            for param in self.classification_head.parameters():
                param.requires_grad = True
        elif self.backbone.base_model_prefix == 'resnet':
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.embedding_model.parameters():
                param.requires_grad = True
            for param in self.classification_head.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x).pooler_output.squeeze()
        x = self.embedding_model(x)
        x = self.classification_head(x)
        return x


parser = argparse.ArgumentParser(description='resnet, single gpu performance test')
parser.add_argument('--lr', default=1e-3, type=float, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--num_epoch', type=int, default=2, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')
parser.add_argument('--cluster', action='store_true')
parser.add_argument('--model', choices=['vit', 'resnet'], default='resnet')

def main():
    args = parser.parse_args()
    base_path = os.path.join("/scratch", "mcatchen", "iNatImages") if args.cluster else "./"


    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    
    model_name = "google/vit-base-patch16-224" if args.model == "vit" else "microsoft/resnet-50"
    processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=True, use_fast=True)

    if args.model == "resnet": 
        image_model = AutoModelForImageClassification.from_pretrained(model_name, local_files_only=True)
    else: 
        image_model = AutoModel.from_pretrained(model_name, local_files_only=True)


  
    
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
    imgmodel = SpeciesEmbeddingModel(image_model)

    compilestart = time.time()
    if torch.cuda.is_available():
        imgmodel = torch.compile(imgmodel)
    compileend = time.time()

    print(f"Compile Time: {compileend - compilestart} seconds")

    imgmodel = imgmodel.to(device) # Load model on the GPU


    # Loss 
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(imgmodel.parameters(), lr=args.lr)

    perf = []

    total_start = time.time()
    first_iter_time = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        start = time.time()
        
        inputs = inputs.to(device) 
        targets = targets.to(device) 

        outputs = imgmodel(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - start
        images_per_sec = args.batch_size/batch_time
        perf.append(images_per_sec)
        if batch_idx == 0:
            first_iter_time = time.time() - total_start

    total_time = time.time() - total_start

    print(f"Total time: {total_time}")
    print(f"First Item Time: {first_iter_time}")
    print(f"Mean images per sec first iter: {perf[0]}")
    print(f"Mean images per sec subseq iter: {sum(perf[1:])/len(perf[1:])}")

if __name__=='__main__':
   main()

