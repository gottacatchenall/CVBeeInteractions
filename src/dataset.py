import torch
import webdataset as wds
from torch.utils.data import IterableDataset, DataLoader
from PIL import Image
import torchvision.transforms.v2 as v2

import pandas as pd
import numpy as np
import os 
import glob
import io
import json

def load_json(path):
    with open(path, 'r') as f:
        dict = json.load(f)
    return dict

class SamplerDecoder():
    def __init__(self, transform=None):
        if transform == None:
            size = 224
            self.transform = v2.Compose([
                v2.Resize((size, size)),
                v2.ToTensor(),
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
    def __init__(self, num_plants, num_bees, holdout_frac=0.2, seed=0):
        rng = np.random.RandomState(seed)
        plant_ids = np.arange(num_plants)
        bee_ids   = np.arange(num_bees)

        self.holdout_plants = rng.choice(plant_ids, int(len(plant_ids)*holdout_frac), replace=False)
        self.holdout_bees   = rng.choice(bee_ids, int(len(bee_ids)*holdout_frac), replace=False)

        self.train_plants = np.setdiff1d(plant_ids, self.holdout_plants)
        self.train_bees   = np.setdiff1d(bee_ids, self.holdout_bees)

    def is_train_pair(self, plant_id, bee_id):
        return (plant_id in self.train_plants) and (bee_id in self.train_bees)

    def is_val_pair(self, plant_id, bee_id):
        return (plant_id in self.holdout_plants) or (bee_id in self.holdout_bees)


class PairedBagDataset(IterableDataset):
    def __init__(
        self,
        plant_dir,
        bee_dir,
        plant_labels_path,
        bee_labels_path,
        interaction_path,
        mask,
        split="train",
        bag_size=8,
        samples_per_pair=16,  # number of bag-pairs to draw per species pair per epoch
    ):
        super().__init__()
        self.plant_dir = plant_dir
        self.bee_dir = bee_dir
        self.plant_name2label = load_json(plant_labels_path)
        self.bee_name2label = load_json(bee_labels_path)
        self.metaweb = self.load_metaweb(interaction_path)

        self.plant_ids = list(self.plant_name2label.values())
        self.bee_ids = list(self.bee_name2label.values())

        self.split = split
        self.bag_size = bag_size
        self.samples_per_pair = samples_per_pair

        self.bee_loaders = self.get_loaders(bee_dir, self.bee_name2label, bag_size)
        self.plant_loaders = self.get_loaders(plant_dir, self.plant_name2label, bag_size)
        self.mask = mask

    def get_loaders(self, dir, name2labels, bag_size):
        decoder = SamplerDecoder()
        # no outer batching; produce bag tensors of shape [B, C, H, W]
        datasets = {
            name2labels[name]: (
                wds.WebDataset(
                    os.path.join(dir, f"{name}.tar"),
                    shardshuffle=True,
                    nodesplitter=wds.split_by_node,
                )
                .decode()
                .to_tuple("json", "jpg")
                .map(decoder)                      # -> (img_tensor, label)
                .batched(bag_size, partial=False)  # drop incomplete bags
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

    def _stream_next(self, iterator, loader_factory):
        """Helper: get next from iterator, if StopIteration -> re-create iterator via loader_factory and try again."""
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(loader_factory())
            return next(iterator), iterator

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers

        # shard species-pairs across workers deterministically
        pairs = [(p, b) for p in self.plant_ids for b in self.bee_ids]
        pairs = pairs[worker_id::num_workers]

        for plant_id, bee_id in pairs:
            if self.split == "train" and not self.mask.is_train_pair(plant_id, bee_id):
                continue
            if self.split == "val" and not self.mask.is_val_pair(plant_id, bee_id):
                continue

            # create fresh iterators (these are streaming, shuffled by WebDataset)
            plant_loader_factory = lambda: self.plant_loaders[plant_id]
            bee_loader_factory = lambda: self.bee_loaders[bee_id]

            plant_iter = iter(plant_loader_factory())
            bee_iter = iter(bee_loader_factory())

            # Draw `samples_per_pair` bag-pairs for this species-pair per epoch
            drawn = 0
            while drawn < self.samples_per_pair:
                # get next plant bag (reinit iterator if exhausted)
                try:
                    p_item = next(plant_iter)
                except StopIteration:
                    plant_iter = iter(plant_loader_factory())
                    try:
                        p_item = next(plant_iter)
                    except StopIteration:
                        # this species has no bags (possible), break out
                        break

                try:
                    b_item = next(bee_iter)
                except StopIteration:
                    bee_iter = iter(bee_loader_factory())
                    try:
                        b_item = next(bee_iter)
                    except StopIteration:
                        break

                # each item is (bag_imgs, labels) where bag_imgs shape [B, C, H, W]
                plant_bag, _ = p_item
                bee_bag, _ = b_item

                # sanity: ensure shapes match bag_size
                if plant_bag.shape[0] != self.bag_size or bee_bag.shape[0] != self.bag_size:
                    # should not happen because partial=False, but defensive check
                    continue

                yield plant_id, plant_bag, bee_id, bee_bag, self.metaweb[plant_id, bee_id]
                drawn += 1
