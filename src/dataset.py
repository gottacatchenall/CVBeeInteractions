import torch
import os
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


# -------------------
# Dataset Module
# -------------------
class ChunkedData(Dataset):
    def __init__(self, chunk_paths, cache_chunks=False):

        self.chunk_paths = chunk_paths
        self.cache_chunks = cache_chunks
        self.chunk_cache = {}

        # Build global index mapping
        self.index_map = []
        for chunk_id, path in enumerate(chunk_paths):
            with open(path, "rb") as f:
                chunk_data = torch.load(f)
            for i in range(len(chunk_data)):
                self.index_map.append((chunk_id, i))

    def _load_chunk(self, chunk_id):
        if self.cache_chunks and chunk_id in self.chunk_cache:
            return self.chunk_cache[chunk_id]
        with open(self.chunk_paths[chunk_id], "rb") as f:
            data = torch.load(f)
        if self.cache_chunks:
            self.chunk_cache[chunk_id] = data
        return data

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        chunk_id, local_idx = self.index_map[idx]
        chunk = self._load_chunk(chunk_id)
        sample = chunk[local_idx]

        return sample


"""
class SpeciesImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir = "./", batch_size = 128, num_workers=0, transform = None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers

    def prepare_data(self):
        pass 

    def setup(self, stage):
        if stage == "fit":
            full = TorchSavedDataset(self.data_dir, train=True, transform=self.transform)
            self.train, self.val = torch.utils.data.random_split(
                full, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
            )
        if stage == "test":
            self.test = TorchSavedDataset(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.predict = TorchSavedDataset(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)
"""