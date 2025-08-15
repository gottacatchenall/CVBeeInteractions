import os
import argparse
import pandas as pd

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from src.model import VitClassifier
from src.dataset import TorchSavedDataset

# -------------------
# Main
# -------------------
def main(image_dir, args):
    dataset_train = TorchSavedDataset(image_dir, train=True)

    # Per-GPU batch size
    gpus = torch.cuda.device_count()

    # DataLoader — fast settings
    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    # Lightning trainer
    net = VitClassifier(
        lr=args.lr
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        strategy='ddp',
        max_epochs=args.max_epochs,
        enable_progress_bar=True
    )

    trainer.fit(net, train_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--max_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--species', default='bees', choices=['plants', 'bees'])
    args = parser.parse_args()

    base_path = os.path.join("/scratch", "mcatchen", "iNatImages", "data") if args.cluster else "./data"
    data_dir = "binary_bees" if args.species == "bees" else "binary_plants"

    image_dir = os.path.join(base_path, data_dir)

    main(image_dir, args)