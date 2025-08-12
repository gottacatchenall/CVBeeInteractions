import torch
import torch.nn as nn
import os
import numpy as np
import sklearn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, ResNetForImageClassification
import matplotlib.pyplot as plt

class ResNetSpecieClassifier(nn.Module):
    def __init__(self, model, species_embedding_dim=16, num_classes=19):
        super().__init__()
        # keep everything except the classifier head
        self.num_classes = num_classes
        self.backbone = model.resnet

        self.species_embedding_mlp = nn.Sequential(
            nn.Linear(2048, species_embedding_dim) 
        )
        self.classification_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(species_embedding_dim, num_classes) 
        )

        # --- Freeze CNN layers  ---
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.species_embedding_mlp.parameters():
            param.requires_grad = True
        for param in self.classification_head.parameters():
            param.requires_grad = True
            
    def resnet_embedding(self, x):
        features = self.backbone(x, output_hidden_states=True)
        return features.pooler_output.squeeze()

    def forward(self, x):
        return self.classification_head(self.species_embedding_mlp(self.resnet_embedding(x)))


# ------------------------
# 1. Load Pretrained ResNet50
# ------------------------
model_name = "microsoft/resnet-50"
processor = AutoImageProcessor.from_pretrained(model_name)
resnet = ResNetForImageClassification.from_pretrained(
    model_name,
)
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# ------------------------
# 2. Prepare Data
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])
batch_size = 256
img_dir  = os.path.join("data/cropped_bombus/")

full_dataset = datasets.ImageFolder(
    img_dir, 
    transform=transform
)
    
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

labels = full_dataset.classes
num_new_classes = len(labels)


# ------------------------
# 3. Fine-Tuning Setup
# ------------------------

model = ResNetSpecieClassifier(resnet)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
loss_fn = nn.CrossEntropyLoss()

for param in model.backbone.parameters():
    param.requires_grad = False
for param in model.species_embedding_mlp.parameters():
    param.requires_grad = True
for param in model.classification_head.parameters():
    param.requires_grad = True

# ------------------------
# 4. Training Loop
# ------------------------
epochs = 25
for epoch in range(epochs):
    _ = model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()


    _ = model.eval()
    correct = 0
    total = 0

    all_probs, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            all_probs.append(torch.softmax(outputs, 0).to('cpu').numpy())
            all_labels.append(labels.to('cpu').numpy())
            total += labels.size(0)
        
    acc = correct / total
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    _one_hot_labels = np.eye(model.num_classes)[all_labels]
    test_map = sklearn.metrics.average_precision_score(
            y_true=_one_hot_labels,  # one-hot encoding
            y_score=all_probs,
            average='macro'
    )
    print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.4f} | Val MAP: {test_map:0.4f}")






# ---- ------- -----------------------------

conf_mat = np.zeros((model.num_classes, model.num_classes))

for inputs, labels in train_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    
    for (i,p) in enumerate(preds):
        conf_mat[labels[i], preds[i]] += 1

conf_mat += 1
fig, ax = plt.subplots()
im = ax.imshow(np.log(conf_mat))
# Loop over data dimensions and create text annotations.
for i in range(model.num_classes):
    for j in range(model.num_classes):
        text = ax.text(j, i, "%i" % (conf_mat[i, j] - 1),
                       ha="center", va="center", color="w")

plt.show()