import torch
import webdataset as wds
from torch.utils.data import IterableDataset, DataLoader
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
        plant_labels_path,
        bee_labels_path, 
        holdout_plants=0, 
        holdout_bees=0, 
        seed=0, 
        strict=True
    ):
        """
        Hold out a fixed number of species of each type.
        If holdout_* is a fraction (0 < x < 1), interpret as proportion.
        """
        self.bee_name2label   = load_json(bee_labels_path)
        self.plant_name2label = load_json(plant_labels_path)

        self.num_plants = len(self.plant_name2label)
        self.num_bees = len(self.bee_name2label)

        rng = np.random.RandomState(seed)
        plant_ids = np.arange(self.num_plants)
        bee_ids   = np.arange(self.num_bees)

        self.strict = strict

        # Handle integers vs fractions
        if 0 < holdout_plants < 1:
            holdout_plants = int(len(plant_ids) * holdout_plants)
        if 0 < holdout_bees < 1:
            holdout_bees = int(len(bee_ids) * holdout_bees)

        self.holdout_plants = rng.choice(plant_ids, holdout_plants, replace=False) if holdout_plants > 0 else []
        self.holdout_bees   = rng.choice(bee_ids, holdout_bees, replace=False) if holdout_bees > 0 else []

        self.train_plants = np.setdiff1d(plant_ids, self.holdout_plants)
        self.train_bees   = np.setdiff1d(bee_ids, self.holdout_bees)

    def is_train_pair(self, plant_id, bee_id):
        return (plant_id in self.train_plants) and (bee_id in self.train_bees)

    def is_val_pair(self, plant_id, bee_id):
        if self.strict:
            return (plant_id in self.holdout_plants and bee_id in self.holdout_bees)
        else:
            # Any pair with a held-out species
            return (plant_id in self.holdout_plants) or (bee_id in self.holdout_bees)

class InteractionDataset(IterableDataset):
    def __init__(
        self,
        plant_dir,
        bee_dir,
        plant_labels_path,
        bee_labels_path,
        interaction_path,
        mask,
        split="train",
        n_per_pair=16,   # <-- sample N images per species per pair
        steps_per_epoch = 100,
    ):
        super().__init__()
        self.plant_dir = plant_dir
        self.bee_dir = bee_dir
        self.plant_name2label = load_json(plant_labels_path)
        self.bee_name2label   = load_json(bee_labels_path)
        self.bee_labels = [x for x in self.bee_name2label.values()]
        self.plant_labels = [x for x in self.plant_name2label.values()]
        self.steps_per_epoch = steps_per_epoch
        self.metaweb = self.load_metaweb(interaction_path)

        self.plant_ids = list(self.plant_name2label.values())
        self.bee_ids   = list(self.bee_name2label.values())

        self.split = split
        self.n_per_pair = n_per_pair
        self.mask = mask

        self.plant_datasets = self.get_loaders(plant_dir, self.plant_name2label)
        self.bee_datasets   = self.get_loaders(bee_dir, self.bee_name2label)

        all_pairs = [(p, b) for p in self.plant_labels for b in self.bee_labels]
        if split == "train":
            self.pairs = [(p, b) for (p, b) in all_pairs if self.mask.is_train_pair(p, b)]
        elif split == "val":
            self.pairs = [(p, b) for (p, b) in all_pairs if self.mask.is_val_pair(p, b)]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # --- Split pairs into pos and negative ---
        self.pos_pair = [(p,b) for (p,b) in self.pairs if self.metaweb[p,b] == 1]
        self.neg_pair = [(p,b) for (p,b) in self.pairs if self.metaweb[p,b] == 0]

    def get_loaders(self, dir, name2labels):
        # Use a simpler DataLoader for the WebDataset iteration inside the worker
        decoder = SamplerDecoder()
        datasets = {
            name2labels[name]: (
                wds.WebDataset(
                    os.path.join(dir, f"{name}.tar"),
                    repeat=True # Important for infinite iteration
                )
                .decode()
                .split_by_worker()
                .to_tuple("json", "jpg")
                .map(decoder) # -> (img_tensor, label)
                .batched(self.n_per_pair)
                .repeat() 
            )
            for name in name2labels.keys()
        }
        return datasets 

    """
    def get_loaders(self, dir, name2labels):
        decoder = SamplerDecoder()
        datasets = {
            name2labels[name]: (
                wds.WebDataset(
                    os.path.join(dir, f"{name}.tar"),
                    shardshuffle=True,
                    nodesplitter=wds.split_by_node,
                    repeat=True
                )
                .decode()
                .to_tuple("json", "jpg")
                .map(decoder) # -> (img_tensor, label)
            )
            for name in name2labels.keys()
        }
        loaders = {k: wds.WebLoader(v, batch_size=self.n_per_pair, drop_last=True) for k, v in datasets.items()}
        return loaders
    """
    def load_metaweb(self, interaction_path):
        int_df = pd.read_csv(interaction_path)
        metaweb = torch.zeros(len(self.plant_name2label), len(self.bee_name2label))
        for _, r in int_df.iterrows():
            plant, bee, int_bit = r
            pi, bi = self.plant_name2label[plant], self.bee_name2label[bee]
            metaweb[pi, bi] = int(int_bit)
        return metaweb
    
    # In InteractionDataset.__iter__():
    def __iter__(self):
        # FIX: Use cycling iterators on the WebDataset iterables
        plant_iters = {k: cycle(v) for k, v in self.plant_datasets.items()}
        bee_iters   = {k: cycle(v) for k, v in self.bee_datasets.items()}

        pos = self.pos_pair.copy()
        neg = self.neg_pair.copy()
        random.shuffle(pos)
        random.shuffle(neg)

        pos_iter = cycle(pos) if len(pos) > 0 else None
        neg_iter = cycle(neg) if len(neg) > 0 else None

        # --- Infinite Pair Sampler Generator ---
        def infinite_pair_sampler():
            while True:
                # Logic to select a positive or negative pair
                take_pos = (neg_iter is None) or (pos_iter is not None and torch.rand(1) < 0.5)

                if take_pos and pos_iter is not None:
                    p, b = next(pos_iter)
                elif neg_iter is not None:
                    p, b = next(neg_iter)
                elif pos_iter is not None: # Fallback if only pos_iter remains
                    p, b = next(pos_iter)
                else:
                    # No pairs available (shouldn't happen in a training loop)
                    return 

                # Fetch the image mini-batches (size n_per_pair)
                plant_img, _ = next(plant_iters[p])
                bee_img, _   = next(bee_iters[b])

                # The metaweb value is a scalar; convert to float tensor for consistency
                yield p, b, plant_img, bee_img, self.metaweb[p, b].float()
        # ---------------------------------------

        # Apply islice to limit the number of batches per epoch
        if self.steps_per_epoch is not None and self.steps_per_epoch > 0:
            return islice(infinite_pair_sampler(), self.steps_per_epoch)
        else:
            return infinite_pair_sampler() # If not set, run indefinitely
    """
    def __iter__(self):
        # Ensure all species datasets are accessible as infinitely cycling iterators
        plant_iters = {k: cycle(v) for k, v in self.plant_datasets.items()} # self.plant_datasets from get_loaders
        bee_iters   = {k: cycle(v) for k, v in self.bee_datasets.items()}   # self.bee_datasets from get_loaders

        pos = self.pos_pair.copy()
        neg = self.neg_pair.copy()
        random.shuffle(pos)
        random.shuffle(neg)

        pos_iter = cycle(pos) if len(pos) > 0 else None
        neg_iter = cycle(neg) if len(neg) > 0 else None

        # Use a loop that yields a continuous stream of pairs
        # The total number of iterations should be large/infinite for an IterableDataset
        # Using islice to manage the number of pairs to sample per epoch, or just 'while True'
        while True: # Changed from fixed loop for better IterableDataset behavior
            # Simple alternating selection logic
            pair_to_sample = None
            if pos_iter is not None and (neg_iter is None or torch.rand(1) < 0.5):
                pair_to_sample = next(pos_iter)
            elif neg_iter is not None:
                pair_to_sample = next(neg_iter)
            else:
                # Should not happen if pos/neg are not both None
                break

            p, b = pair_to_sample

            # Use next() on the cycling iterators which now yield the mini-batches
            # plant_img is a batch of (n_per_pair) images and labels are unused
            plant_img, _ = next(plant_iters[p])
            bee_img, _   = next(bee_iters[b])

            # metaweb[p, b] is a scalar, convert to tensor if necessary
            yield p, b, plant_img, bee_img, self.metaweb[p, b].float() 

    def __iter__(self):
        def infinite_loader(loader):
            while True:
                for batch in loader:
                    yield batch

        plant_loaders = {k: infinite_loader(v) for k, v in self.plant_loaders.items()}
        bee_loaders   = {k: infinite_loader(v) for k, v in self.bee_loaders.items()}

        pos = self.pos_pair.copy()
        neg = self.neg_pair.copy()
        random.shuffle(pos)
        random.shuffle(neg)

        pos_iter = cycle(pos) if len(pos) > 0 else None
        neg_iter = cycle(neg) if len(neg) > 0 else None

        total = len(pos) + len(neg)
        take_pos_mask = torch.rand(total) < 0.5

        for take_pos in take_pos_mask:
            if take_pos and pos_iter is not None:
                p, b = next(pos_iter)
            elif neg_iter is not None:
                p, b = next(neg_iter)
            else:
                p, b = next(pos_iter)

            plant_img, _ = next(plant_loaders[p])
            bee_img, _   = next(bee_loaders[b])

            yield p, b, plant_img, bee_img, self.metaweb[p, b]
        """

class PlantPollinatorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        plant_dir,
        bee_dir,
        plant_labels_path,
        bee_labels_path,
        interaction_path,
        args
    ):
        super().__init__()

        self.plant_dir = plant_dir
        self.bee_dir = bee_dir
        self.plant_labels_path = plant_labels_path
        self.bee_labels_path = bee_labels_path
        self.interaction_path = interaction_path
        self.train_steps_per_epoch = args.train_steps_per_epoch

        self.mask = ZeroShotMaskMaker(
            plant_labels_path,
            bee_labels_path,
            holdout_bees=args.holdout_bees,
            holdout_plants=args.holdout_plants
        )
        # self.metaweb = load_metaweb(interactions_path)
        
        
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.prefetch_factor = args.prefetch_factor
        self.persistent_workers = args.persistent_workers
        
    def setup(self, stage=None):
        self.train_dataset = InteractionDataset(
            self.plant_dir,
            self.bee_dir,
            self.plant_labels_path,
            self.bee_labels_path,
            self.interaction_path,
            self.mask,
            steps_per_epoch = self.train_steps_per_epoch,
            split = "train"
        )

        self.val_dataset = InteractionDataset(
            self.plant_dir,
            self.bee_dir,
            self.plant_labels_path,
            self.bee_labels_path,
            self.interaction_path,
            self.mask,
            steps_per_epoch = None,
            split = "val"
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size = self.batch_size, 
            num_workers = self.num_workers,
            #prefetch_factor = self.prefetch_factor,
            #persistent_workers = self.persistent_workers
        )
        return train_loader
  
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            #prefetch_factor = self.prefetch_factor,
            #persistent_workers = self.persistent_workers
        )
        return val_loader