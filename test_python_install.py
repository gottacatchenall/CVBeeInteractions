import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoFeatureExtractor
from transformers import AutoModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import os
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

print("Is cuda available: %s" % torch.cuda.is_available())



bombus_dir = os.path.join("data", "bombus_img")

img_per_species = {x: len(os.listdir(os.path.join(bombus_dir, x))) for x in os.listdir(bombus_dir) if os.path.isdir(os.path.join(bombus_dir, x)) }


prop_img_per_species = {x: img_per_species[x]/sum(img_per_species.values())
for x in img_per_species.keys() }


prop_img_per_species

