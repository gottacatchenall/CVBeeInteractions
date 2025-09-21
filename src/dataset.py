import torch
import webdataset as wds
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.v2 as v2
import random
from torchvision.datasets import ImageFolder
from itertools import cycle, islice

import pytorch_lightning as pl

from torchvision.io import decode_image

import pandas as pd
import numpy as np
import io
import os
import json
import glob

def load_json(path):
    with open(path, 'r') as f:
        dict = json.load(f)
    return dict


class SamplerDecoder():
    def __init__(self, transform=None):
        if transform == None:
            self.transform = v2.Compose([
            v2.ToTensor(),
            v2.Resize((224, 224)),
#            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])
        else:
            self.transform = transform
        
    def __call__(self, sample):
        meta, img_bytes = sample
        img = Image.open(io.BytesIO(img_bytes))
        img = self.transform(img)
        label = torch.tensor(meta["label"], dtype=torch.long)
        return img, label


class ZeroShotMaskMaker:
    def __init__(
        self, 
        plant_names,
        bee_names, 
        holdout_plants=0, 
        holdout_bees=0, 
        seed=0, 
        strict=False
    ):
        """
        Hold out a fixed number of species of each type.
        If holdout_* is a fraction (0 < x < 1), interpret as proportion.
        """
        rng = np.random.RandomState(seed)
        self.strict = strict

        # Handle integers vs fractions
        if 0 < holdout_plants < 1:
            holdout_plants = int(len(plant_names) * holdout_plants)
        if 0 < holdout_bees < 1:
            holdout_bees = int(len(bee_names) * holdout_bees)

        self.holdout_plants = rng.choice(plant_names, holdout_plants, replace=False) if holdout_plants > 0 else []
        self.holdout_bees   = rng.choice(bee_names, holdout_bees, replace=False) if holdout_bees > 0 else []

        self.train_plants = np.setdiff1d(plant_names, self.holdout_plants)
        self.train_bees   = np.setdiff1d(bee_names, self.holdout_bees)

    def is_train_pair(self, plant_id, bee_id):
        return (plant_id in self.train_plants) and (bee_id in self.train_bees)

    def is_val_pair(self, plant_name, bee_name):
        if self.strict:
            return (plant_name in self.holdout_plants and bee_name in self.holdout_bees)
        else:
            # Any pair with a held-out species
            return (plant_name in self.holdout_plants) or (bee_name in self.holdout_bees)

class PairInteractionDataset(Dataset):
    """
    A Dataset for widget/sprocket pair interactions that loads N_IMGS images
    for each on the fly.
    """
    def __init__(self, 
        pairs_data, 
        plant_dir, 
        bee_dir, 
        mask, 
        n_imgs = 16,
        split = "train", 
        transform=None
    ):
        # pairs_data should be a list of tuples: 
        # [(bee_id_1, plant_id_1, label_1), ...]
        self.n_imgs = n_imgs
        self.plant_dir = plant_dir
        self.bee_dir = bee_dir
        self.mask = mask
        self.split = split
        self.transform = transform

        check_fcn = mask.is_train_pair if split == "train" else mask.is_val_pair    
        self.pairs_data = [(p,b,k) for p,b,k in pairs_data if check_fcn(p,b)]

        # Example: {bee_id: [path_to_img_1, path_to_img_2, ...]}
        self.plant_file_maps = self._map_image_files(plant_dir)
        self.bee_file_maps = self._map_image_files(bee_dir)


    def _map_image_files(self, base_dir):
        """Helper to map IDs to lists of image paths."""
        file_map = {}
        for item_id in os.listdir(base_dir): 
            item_path = os.path.join(base_dir, item_id)
            if os.path.isdir(item_path):
                # Assumes images for an ID are in a subfolder named by the ID
                images = [os.path.join(item_path, f) for f in os.listdir(item_path) if f.endswith(('.jpg', '.png'))]
                file_map[item_id] = images
        return file_map
        

    def __len__(self):
        return len(self.pairs_data)

    def __getitem__(self, idx):
        plant_name, bee_name, label = self.pairs_data[idx]
        
        # Get all image paths for the current widget/sprocket
        plant_paths = self.plant_file_maps[plant_name]
        bee_paths = self.bee_file_maps[bee_name]
        
        # Sample N_IMGS paths with replacement (safer if fewer than N_IMGS exist)
        plant_samples = random.choices(plant_paths, k=self.n_imgs)
        bee_samples = random.choices(bee_paths, k=self.n_imgs)
        
        # 2. Load and Transform the sampled images
    
        def load_images(paths):
            imgs = []
            for path in paths:
                img = Image.open(path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
            # Stack the N_IMGS images into a single tensor of shape [n_imgs, C, H, W]
            return torch.stack(imgs) 

        plant_tensors = load_images(plant_samples)
        bee_tensors = load_images(bee_samples)
        
        # NOTE: The tensors are still on the CPU at this point.
        return plant_name, bee_name, plant_tensors, bee_tensors, torch.tensor(label, dtype=torch.long)

class PlantPollinatorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        plant_dir,
        bee_dir,
        interaction_path,
        args
    ):
        super().__init__()

        self.plant_dir = plant_dir
        self.bee_dir = bee_dir

        self.image_transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        df = pd.read_csv(interaction_path)

        self.plant_names = df.plant.unique()
        self.bee_names = df.bee.unique()
        self.all_pairs = [(r.plant, r.bee, r.interaction) for _,r in df.iterrows()]

        self.mask = ZeroShotMaskMaker(
            self.plant_names,
            self.bee_names,
            holdout_bees=args.holdout_bees,
            holdout_plants=args.holdout_plants
        )        
        
        self.batch_size = args.batch_size
        self.imgs_per_species = args.imgs_per_species
        self.num_workers = args.num_workers
        self.prefetch_factor = args.prefetch_factor
        self.persistent_workers = args.persistent_workers
        
    def setup(self, stage=None):
        self.train_dataset = PairInteractionDataset(
            self.all_pairs,
            self.plant_dir,
            self.bee_dir,
            self.mask,
            n_imgs = self.imgs_per_species,
            transform= self.image_transform,
            split = "train"
        )

        self.val_dataset = PairInteractionDataset(
            self.all_pairs,
            self.plant_dir,
            self.bee_dir,
            self.mask,
            n_imgs = self.imgs_per_species,
            transform= self.image_transform,
            split = "val"
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size = self.batch_size, 
            num_workers = self.num_workers,
            pin_memory=True, 
            shuffle=True,
            prefetch_factor = self.prefetch_factor,
            persistent_workers = self.persistent_workers
        )
        return train_loader
  
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory=True,
            shuffle=True,
            prefetch_factor = self.prefetch_factor,
            persistent_workers = self.persistent_workers
        )
        return val_loader