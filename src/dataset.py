import torch
from torch.utils.data import DataLoader
import webdataset as wds
from PIL import Image
from torchvision import transforms
import pytorch_lightning as pl
import os 

def decode_sample(sample):
    img_bytes, meta = sample
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(io.BytesIO(img_bytes))
    img = transform(img)
        
    # Load label from json bytes
    label = torch.tensor(meta["label"], dtype=torch.long)
    return img, label

def make_dataset(shard_pattern, shuffle_buffer=1000):
    return (
        wds.WebDataset(shard_pattern, shardshuffle=True)  # shuffle shards
           .decode()                                     # decode bytes
           .to_tuple("jpg", "json")     
           .map(decode_sample)  # transform
           .shuffle(shuffle_buffer)                      # shuffle samples
           .repeat()                                     # repeat indefinitely
    )

class WebDatasetDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            data_dir, 
            batch_size=32, 
            num_workers=0, 
            seed = 42,
            train_pattern = "train-{0000..14}.tar",
            test_pattern = "test-{0000..02}.tar",
            val_pattern = "val-{0000..04}.tar"
    ):
        super().__init__()
        self.seed = seed
        self.train_shards = os.path.join(data_dir, train_pattern)
        self.test_shards = os.path.join(data_dir, test_pattern)
        self.val_shards = os.path.join(data_dir, val_pattern)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = (
            wds.WebDataset(self.train_shards, shardshuffle=True)
               .decode()
               .to_tuple("jpg", "json")
        ).map(decode_sample)

        self.test_dataset = (
            wds.WebDataset(self.test_shards, shardshuffle=True)
               .decode()
               .to_tuple("jpg", "json")
        ).map(decode_sample)
        self.val_dataset = (
            wds.WebDataset(self.test_shards, shardshuffle=True)
               .decode()
               .to_tuple("jpg", "json")
        ).map(decode_sample)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            #collate_fn=collate_to_device,
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            #collate_fn=collate_to_device,
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            #collate_fn=collate_to_device,
        )

#datamodule = WebDatasetDataModule("data/plant_wds", batch_size=64)
#datamodule.setup()
