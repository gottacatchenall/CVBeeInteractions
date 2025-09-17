import torch
import webdataset as wds
from torch.utils.data import IterableDataset, DataLoader
from PIL import Image
import torchvision.transforms.v2 as v2
import random
from torchvision.datasets import ImageFolder
from itertools import cycle, islice

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
            # Zero-shot: must mix train × held-out
            return ((plant_id in self.holdout_plants and bee_id in self.train_bees) or
                    (bee_id in self.holdout_bees and plant_id in self.train_plants))
        else:
            # Any pair with a held-out species
            return (plant_id in self.holdout_plants) or (bee_id in self.holdout_bees)


class InteractionDataset(IterableDataset):
    def load_metaweb(self, interaction_path):
        int_df = pd.read_csv(interaction_path)
        metaweb = torch.zeros(len(self.plant_name2label), len(self.bee_name2label), dtype=bool)
        for _, r in int_df.iterrows():
            plant, bee, int_bit = r
            pi, bi = self.plant_name2label[plant], self.bee_name2label[bee]
            metaweb[pi, bi] = bool(int_bit)
        return metaweb
    
    def get_class2imgs(self, img_folder, name2label):
        return {
            v: [x for x in glob.glob(os.path.join(img_folder, k, "*.jpg"))]
            for k,v in name2label.items()
        }
        
    
    def __init__(
        self,
        plant_dir,
        bee_dir,
        plant_labels_path,
        bee_labels_path,
        interaction_path,
        mask,
        num_samples = 16,
        split = "train",
    ):
        super().__init__()   
        
        self.plant_dir = plant_dir
        self.bee_dir = bee_dir

        self.num_samples = num_samples
        self.mask = mask 

        self.bee_name2label   = load_json(bee_labels_path)
        self.plant_name2label = load_json(plant_labels_path)
        self.bee_label2name   = {v:k for k,v in self.bee_name2label.items()}
        self.plant_label2name = {v:k for k,v in self.plant_name2label.items()}

        self.bee_labels = [x for x in self.bee_name2label.values()]
        self.plant_labels = [x for x in self.plant_name2label.values()]

        self.bee_label2image = self.get_class2imgs(self.bee_dir, self.bee_name2label)
        self.plant_label2image = self.get_class2imgs(self.plant_dir, self.plant_name2label)

        self.metaweb = self.load_metaweb(interaction_path)

        self.transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])


        self.split = split
        all_pairs = [(p, b) for p in self.plant_labels for b in self.bee_labels]
        if split == "train":
            self.pairs = [(p, b) for (p, b) in all_pairs if self.mask.is_train_pair(p, b)]
        elif split == "val":
            self.pairs = [(p, b) for (p, b) in all_pairs if self.mask.is_val_pair(p, b)]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # --- Split pairs into pos and negative ---
        self.pos_pair = [(p,b) for (p,b) in self.pairs if self.metaweb[p,b]]
        self.neg_pair = [(p,b) for (p,b) in self.pairs if ~self.metaweb[p,b]]

    def sample_images(self, p,b):
        plant_paths = random.choices(self.plant_label2image[p], k=self.num_samples)
        bee_paths = random.choices(self.bee_label2image[b], k=self.num_samples)

        return torch.stack([self.transform(decode_image(p)) for p in plant_paths]), torch.stack([self.transform(decode_image(b)) for b in bee_paths])

    def __iter__(self):
        # Shuffled copies each epoch
        pos = self.pos_pair.copy()
        neg = self.neg_pair.copy()
        random.shuffle(pos)
        random.shuffle(neg)

        # Cycle within each class; choose class with p=0.5 per sample
        pos_iter = cycle(pos) if len(pos) > 0 else iter(())
        neg_iter = cycle(neg) if len(neg) > 0 else iter(())

        # Define an epoch length; one pass over all unique pairs on average
        total = len(pos) + len(neg)

        for _ in range(total):
            take_pos = random.random() < 0.5
            if take_pos and len(pos) > 0:
                p, b = next(pos_iter)
                p_img, b_img = self.sample_images(p,b)
                yield p, b, p_img, b_img, self.metaweb[p,b]
            elif len(neg) > 0:
                p, b = next(neg_iter)
                p_img, b_img = self.sample_images(p,b)
                yield p, b, p_img, b_img, self.metaweb[p,b]
            else:
                # Fallback if one side is empty
                p, b = next(pos_iter)
                p_img, b_img = self.sample_images(p,b)
                yield p, b, p_img, b_img, self.metaweb[p,b]
            



"""
class PairedDataset(IterableDataset):
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
    ):
        super().__init__()
        self.plant_dir = plant_dir
        self.bee_dir = bee_dir
        self.plant_name2label = load_json(plant_labels_path)
        self.bee_name2label   = load_json(bee_labels_path)
        self.metaweb = self.load_metaweb(interaction_path)

        self.plant_ids = list(self.plant_name2label.values())
        self.bee_ids   = list(self.bee_name2label.values())

        self.split = split
        self.n_per_pair = n_per_pair
        self.mask = mask

        self.plant_loaders = self.get_loaders(plant_dir, self.plant_name2label)
        self.bee_loaders   = self.get_loaders(bee_dir, self.bee_name2label)

    def get_loaders(self, dir, name2labels):
        decoder = SamplerDecoder()
        datasets = {
            name2labels[name]: (
                wds.WebDataset(
                    os.path.join(dir, f"{name}.tar"),
                    shardshuffle=True,
                    nodesplitter=wds.split_by_node,
                )
                .decode()
                .to_tuple("json", "jpg")
                .map(decoder)   # -> (img_tensor, label)
            )
            for name in name2labels.keys()
        }
        loaders = {k: wds.WebLoader(v, batch_size=None) for k, v in datasets.items()}
        return loaders

    def load_metaweb(self, interaction_path):
        int_df = pd.read_csv(interaction_path)
        metaweb = torch.zeros(len(self.plant_name2label), len(self.bee_name2label))
        for _, r in int_df.iterrows():
            plant, bee, int_bit = r
            pi, bi = self.plant_name2label[plant], self.bee_name2label[bee]
            metaweb[pi, bi] = int(int_bit)
        return metaweb

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers

        # shard species pairs
        pairs = [(p, b) for p in self.plant_ids for b in self.bee_ids]
        pairs = pairs[worker_id::num_workers]

        for plant_id, bee_id in pairs:
            if self.split == "train" and not self.mask.is_train_pair(plant_id, bee_id):
                continue
            if self.split == "val" and not self.mask.is_val_pair(plant_id, bee_id):
                continue

            plant_iter = iter(self.plant_loaders[plant_id])
            bee_iter   = iter(self.bee_loaders[bee_id])

            for _ in range(self.n_per_pair):
                try:
                    plant_img, _ = next(plant_iter)
                except StopIteration:
                    plant_iter = iter(self.plant_loaders[plant_id])
                    plant_img, _ = next(plant_iter)

                try:
                    bee_img, _ = next(bee_iter)
                except StopIteration:
                    bee_iter = iter(self.bee_loaders[bee_id])
                    bee_img, _ = next(bee_iter)

                yield plant_id, plant_img, bee_id, bee_img, self.metaweb[plant_id, bee_id]
"""