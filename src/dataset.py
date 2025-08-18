import torch
from torch.utils.data import DataLoader
import webdataset as wds
from PIL import Image
from torchvision import transforms
import pytorch_lightning as pl
import os 
import glob
import io

def default_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

class SamplerDecoder():
    def __init__(self, transform=None):
        if transform == None:
            self.transform = default_transform()
        else:
            self.transform = transform
        
    def __call__(self, sample):
        img_bytes, meta = sample
        img = Image.open(io.BytesIO(img_bytes))
        img = self.transform(img)
        label = torch.tensor(meta["label"], dtype=torch.long)
        return img, label


class WebDatasetDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            data_dir, 
            batch_size=32, 
            num_workers=0, 
            seed = 42,
            prefetch_factor = 2,
            persistent_workers = False,
            train_transform = None,
            test_transform = None,
            train_pattern = "train-*.tar",
            test_pattern = "test-*.tar",
            val_pattern = "val-*.tar"
    ):
        super().__init__()
        self.seed = seed
        self.train_shards = [os.path.join(data_dir, x) for x in glob.glob(train_pattern, root_dir=data_dir)]
        self.test_shards = [os.path.join(data_dir, x) for x in glob.glob(test_pattern, root_dir=data_dir)]
        self.val_shards = [os.path.join(data_dir, x) for x in glob.glob(val_pattern, root_dir=data_dir)]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
    def setup(self, stage=None):
        test_decoder = SamplerDecoder()

        self.train_dataset = (
            wds.WebDataset(self.train_shards, shardshuffle=True)
               .decode()
               .to_tuple("jpg", "json")
        ).map(test_decoder)

        self.test_dataset = (
            wds.WebDataset(self.test_shards, shardshuffle=True)
               .decode()
               .to_tuple("jpg", "json")
        ).map(test_decoder)
        self.val_dataset = (
            wds.WebDataset(self.test_shards, shardshuffle=True)
               .decode()
               .to_tuple("jpg", "json")
        ).map(test_decoder)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers = self.persistent_workers
            prefetch_factor=self.prefetch_factor
            #collate_fn=collate_to_device,
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers = self.persistent_workers
            prefetch_factor=self.prefetch_factor
            #collate_fn=collate_to_device,
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers = self.persistent_workers
            prefetch_factor=self.prefetch_factor
            #collate_fn=collate_to_device,
        )

