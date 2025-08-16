import os
import argparse

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from src.model import VitClassifier
from src.dataset import WebDatasetDataModule

torch.set_float32_matmul_precision('high')


# -------------------
# Main
# -------------------
def main(image_dir, args):
    species_data = WebDatasetDataModule(
        data_dir = image_dir,
        batch_size = args.batch_size,
        num_workers= args.num_workers,
    )

    num_classes = 19 if args.species == "Bombus" else 158

    net = VitClassifier(
        lr=args.lr,
        num_classes=num_classes
    )

    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES"))
    gpus = torch.cuda.device_count()

    #trainer = pl.Trainer(accelerator="mps")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        strategy='ddp',
        profiler="simple",
        num_nodes=num_nodes, 
        max_epochs=args.max_epochs,
        enable_progress_bar=False
    )

    trainer.fit(net, species_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--species', default='bees', choices=['plants', 'bees'])
    args = parser.parse_args()

    base_path = os.path.join("/scratch", "mcatchen", "iNatImages", "data") if args.cluster else "./data"
    data_dir = "bombus_wds" if args.species == "bees" else "plant_wds"

    image_dir = os.path.join(base_path, data_dir)

    main(image_dir, args)