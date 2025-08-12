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

torch.set_float32_matmul_precision('high')

class ResNetSpeciesEmbeddingModel(nn.Module):
    def __init__(
        self, 
        species_embedding_dim=64, 
        num_classes=19,
        batch_size = 128,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

        self.metrics = {
            "AUROC": torchmetrics.AUROC(task="multiclass", num_classes=self.num_classes).to(self.device),
            "MAP": torchmetrics.AveragePrecision(task="multiclass", num_classes=self.num_classes).to(self.device), 
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)
        }

        model_name = "microsoft/resnet-50"
        self.image_processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=True, use_fast=True)
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.image_processor.image_mean, std=self.image_processor.image_std),
        ])

        self.image_model = ResNetForImageClassification.from_pretrained(model_name, local_files_only=True)

        self.embedding_model = nn.Sequential(
            # Final ResNet pooling is [n_batches, 2048]
            nn.Linear(2048, species_embedding_dim) 
        )

        self.classification_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(species_embedding_dim, num_classes) 
        )

        # --- Freeze CNN layers  ---
        for param in self.image_model.parameters():
            param.requires_grad = False
        for param in self.embedding_model.parameters():
            param.requires_grad = True
        for param in self.classification_head.parameters():
            param.requires_grad = True
            
    def embed_image(self, x):
        features = self.image_model.resnet(x, output_hidden_states=True)
        return features.pooler_output.squeeze()
    
    def prepare_train(self):
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


class ViTSpeciesEmbeddingModel(nn.Module):
    def __init__(self, 
        num_classes=19,
        species_embedding_dim = 64, 
        batch_size = 512,
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


        """
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.image_model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.image_model.to(self.device)
        self.image_model.eval() 
        for param in self.image_model.parameters():
            param.requires_grad = False
        """

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
        self.image_model.to(self.device)

        self.classification_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(species_embedding_dim, num_classes) 
        )
        self.image_model.to(self.device)

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


def setup_data(
    data_dir,
    model,
):
    full_dataset = datasets.ImageFolder(
        data_dir, 
        transform=model.image_transform,
    )
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=model.batch_size, shuffle=True, pin_memory=pin_mem)
    test_loader = DataLoader(test_dataset, batch_size=model.batch_size, pin_memory=pin_mem)
    return train_loader, test_loader 


def train_loop(model, optimizer, criterion, train_loader):
    model.prepare_train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(model.device), labels.to(model.device)
        with torch.no_grad():
            outputs = model.embed_image(images) 

        logits = model(outputs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def test_loop(model, criterion, test_loader):
    all_probs = []
    all_labels = []

    running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(model.device), labels.to(model.device)
            vit_outputs = model.embed_image(images)
            logits = model(vit_outputs)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs)
            all_labels.append(labels)
            running_loss += criterion(logits, labels.to(model.device)).item()

    test_loss = running_loss / len(test_loader)
    all_probs = torch.concatenate(all_probs)
    all_labels = torch.concatenate(all_labels)
    return model.test_stats(all_labels, all_probs, test_loss)


def train(model, img_dir, n_epochs, learning_rate):
    model.to(model.device)
    train_loader, test_loader = setup_data(img_dir, model)

    criterion = nn.CrossEntropyLoss().to(model.device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    stat_dicts = []

    for epoch in range(n_epochs):
        train_loss = train_loop(model, optimizer, criterion, train_loader)
        test_stats = test_loop(model, criterion, test_loader)

        test_stats["train_loss"] = train_loss
        test_stats["epoch"] = epoch


        print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} ||| Test Loss: {test_stats["test_loss"]:.4f} | Test MAP: {test_stats["MAP"]:.4f} | Test Accuracy: {test_stats["accuracy"]:.4f}")
        stat_dicts.append(test_stats)

    return pd.DataFrame(stat_dicts)


def main(parser):
    args = parser.parse_args()
    base_path = os.path.join("/scratch", "mcatchen", "iNatImages") if args.cluster else "./"

    model_name = args.model
    
    model = ViTSpeciesEmbeddingModel(batch_size=args.batchsize,species_embedding_dim=args.embeddim) if args.model == "vit" else ResNetSpeciesEmbeddingModel(batch_size=args.batchsize, species_embedding_dim=args.embeddim)

    if torch.cuda.is_available():
        model = torch.compile(model)

    img_dir  = os.path.join(base_path, "data", "bombus_img")    
    print(f"Starting training on {model.device} with dir {img_dir}")

    df = train(model, img_dir, args.nepoch, args.lr)

    csv_path = os.path.join(base_path, model_name+".csv")
    df.to_csv(csv_path, index=False)
 

if __name__=='__main__':   
    parser = argparse.ArgumentParser(description='Cropping iNaturalist Images with Zero-Shot Object Detection')
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--model', choices=['vit', 'resnet'])
    parser.add_argument('--nepoch', default=5, type=int)
    parser.add_argument('--lr', default=5e-3, type=float)
    parser.add_argument('--batchsize', default=256, type=int)
    parser.add_argument('--embeddim', default=64, type=int)

    main(parser)



