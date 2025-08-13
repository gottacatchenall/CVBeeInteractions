
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from transformers import AutoFeatureExtractor
from transformers import AutoModel

import pytorch_lightning as pl

import os 
import argparse
import pandas as pd
import statistics


class SpeciesImageDataset(Dataset):
    def __init__(
            self, 
            img_dir, 
            train = True,
            transform=None, 
            target_transform=None
    ):
        df = pd.read_csv(os.path.join(img_dir, "dataset.csv"))

        if train:
            idx = torch.nonzero(torch.tensor(df["train"])).flatten()
        else:
            idx = torch.nonzero(torch.tensor(df["test"])).flatten()

        self.labels = torch.tensor(df["label"])[idx]
        self.img_dir = img_dir
        self.image_relpaths = df.relpath.iloc[idx].to_list()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_relpaths[idx])
        image = torchvision.io.read_image(img_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class SpeciesImagesDataModule(pl.LightningDataModule):
    def __init__(self, data_dir = "./", batch_size = 256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        pass 
    def setup(self, stage):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            full = SpeciesImageDataset(self.data_dir, train=True, transform=self.transform)
            self.train, self.val = torch.utils.data.random_split(
                full, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
            )
        if stage == "test":
            self.test = SpeciesImageDataset(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32)



parser = argparse.ArgumentParser(description='cifar10 classification models, pytorch-lightning parallel test')
parser.add_argument('--lr', default=1e-3, help='')
parser.add_argument('--max_epochs', type=int, default=4, help='')
parser.add_argument('--batch_size', type=int, default=512, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')

def main():
    print("Starting...")

    args = parser.parse_args()

    class ViTSpeciesEmbeddingModel(pl.LightningModule):
        def __init__(
            self, 
            num_classes=19,
            species_embedding_dim = 128, 
        ):
            super(ViTSpeciesEmbeddingModel, self).__init__()
            self.loss_function = nn.CrossEntropyLoss()
            self.num_classes = num_classes
            self.species_embedding_dim = species_embedding_dim

            self.image_model = AutoModel.from_pretrained(
                "google/vit-base-patch16-224", 
                local_files_only=True
            )
            self.image_model.train()
            self.embedding_model = nn.Linear(
                768, 
                species_embedding_dim
            )
            self.classification_head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(species_embedding_dim, num_classes) 
            )

        def forward(self, x):
            return self.classification_head(self.embedding_model(self.image_model(x).pooler_output))
        
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            l =  self.loss_function(y_hat, y)
            self.log_dict({"train_loss": l, "batch_idx": batch_idx}, prog_bar=False)
            return l

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=args.lr)

    net = ViTSpeciesEmbeddingModel()

    class Benchmark(pl.Callback):
        """A callback that measures the median execution time between the start and end of a batch."""
        def __init__(self):
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.times = []

        def median_time(self):
            return statistics.median(self.times)

        def on_train_batch_start(self, trainer, *args, **kwargs):
            self.start.record()

        def on_train_batch_end(self, trainer, *args, **kwargs):
            # Exclude the first iteration to let the model warm up
            if trainer.global_step > 1:
                self.end.record()
                torch.cuda.synchronize()
                self.times.append(self.start.elapsed_time(self.end) / 1000)

    """ 
        Here we initialize a Trainer() explicitly with 1 node and 2 GPUs per node.
        
        To make this script more generic, you can use torch.cuda.device_count() to set the number of GPUs
        and you can use int(os.environ.get("SLURM_JOB_NUM_NODES")) to set the number of nodes. 
        We also set progress_bar_refresh_rate=0 to avoid writing a progress bar to the logs, 
        which can cause issues due to updating logs too frequently.
    """

    num_gpus = torch.cuda.device_count()
    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES"))

    logger = pl.loggers.CSVLogger(os.joinpath("/scratch", "mcatchen", "lightning_logs"), name="my_exp_name")

    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=num_gpus, 
        num_nodes=num_nodes, 
        strategy='ddp', 
        profiler = "simple",
        max_epochs = args.max_epochs, 
        enable_progress_bar=False,
        logger = logger
    ) 

    feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224", local_files_only=True, use_fast=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=feature_extractor.image_mean, 
            std=feature_extractor.image_std),
    ])
    species_data = SpeciesImagesDataModule(
        data_dir = os.path.join("/scratch", "mcatchen", "iNatImages", "data", "bombus_img"),
        batch_size = args.batch_size,
        transform=transform
    )

    model = ViTSpeciesEmbeddingModel()
    compiled_model = torch.compile(model, mode="reduce-overhead")

    # Measure the median iteration time with uncompiled model
    benchmark = Benchmark()
    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=num_gpus, 
        num_nodes=num_nodes, 
        strategy='ddp', 
        #profiler = "simple",
        max_steps=10,
        #max_epochs = args.max_epochs, 
        enable_progress_bar=False,
        #logger = logger
        callbacks=[benchmark]
    ) 
    trainer.fit(model)
    eager_time = benchmark.median_time()

    # Measure the median iteration time with compiled model
    benchmark = Benchmark()
    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=num_gpus, 
        num_nodes=num_nodes, 
        strategy='ddp', 
        #profiler = "simple",
        max_steps=10,
        #max_epochs = args.max_epochs, 
        enable_progress_bar=False,
        #logger = logger
        callbacks=[benchmark]
    ) 
    trainer.fit(compiled_model)
    compile_time = benchmark.median_time()


    trainer.fit(net,species_data)

    speedup = eager_time / compile_time
    print(f"Eager median time: {eager_time:.4f} seconds")
    print(f"Compile median time: {compile_time:.4f} seconds")
    print(f"Speedup: {speedup:.1f}x")


if __name__=='__main__':
   main()