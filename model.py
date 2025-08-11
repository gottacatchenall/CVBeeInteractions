import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoFeatureExtractor
from transformers import AutoModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import os
import sklearn
import numpy as np


class SpeciesEmbeddingModel(nn.Module):
    def __init__(self, 
            num_classes,
            species_embedding_dim = 128, 
            batch_size = 256,
        ):

        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.species_embedding_dim = species_embedding_dim

        self.feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.feature_extractor.image_mean, 
                std=self.feature_extractor.image_std),
        ])

        self.image_model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.image_model.to(self.device)
        self.image_model.eval() 
        for param in self.image_model.parameters():
            param.requires_grad = False


        self.embedding_model = nn.Linear(
            768, 
            species_embedding_dim
        )
        self.classification_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(species_embedding_dim, num_classes) 
        )

    def forward(self, x):
        return self.classification_head(self.embedding_model(x))


def setup_data(
    data_dir,
    model,
):
    full_dataset = datasets.ImageFolder(
        data_dir, 
        transform=model.image_transform
    )
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=model.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=model.batch_size)
    return train_loader, test_loader 


def train_loop(model, optimizer, criterion, train_loader):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(model.device), labels.to(model.device)
        with torch.no_grad():
            outputs = model.image_model(pixel_values=images).pooler_output 

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
            images = images.to(model.device)
            vit_outputs = model.image_model(pixel_values=images).pooler_output
            logits = model(vit_outputs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
            running_loss += criterion(logits, labels.to(model.device))

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    _one_hot_labels = np.eye(model.num_classes)[all_labels]

    
    test_map = sklearn.metrics.average_precision_score(
            y_true=_one_hot_labels,  # one-hot encoding
            y_score=all_probs,
            average='macro'
    )
    test_loss = running_loss / len(test_loader)

    return test_loss, test_map

def train(n_epochs, learning_rate):
    img_dir  = os.path.join("data/img_120/")
    model = SpeciesEmbeddingModel(num_classes=18)
    model.to(model.device)

    train_loader, test_loader = setup_data(img_dir, model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        train_loss = train_loop(model, optimizer, criterion, train_loader)
        test_loss, test_map = test_loop(model, criterion, test_loader)
        
        print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} ||| Test Loss: {test_loss:.4f} | Test MAP: {test_map:.4f}" )


train(3, 2e-4)