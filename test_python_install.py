import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoFeatureExtractor
from transformers import AutoModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

print("Is cuda available: %s" % torch.cuda.is_available())