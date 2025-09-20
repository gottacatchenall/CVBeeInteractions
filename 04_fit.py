import argparse 
import os
import torch

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from src.dataset import InteractionDataset, ZeroShotMaskMaker, PlantPollinatorDataModule
from src.model import InteractionPredictor
from lightning.pytorch.loggers import CSVLogger

def setup_data(base_path, args, toy=False):
    plant_dir = "toy_plant_wds" if toy else "plant_wds"
    bee_dir = "toy_bee_wds" if toy else "bee_wds"

    plant_dir = os.path.join(base_path, plant_dir)
    bee_dir = os.path.join(base_path, bee_dir)
    plant_labels_path = os.path.join(base_path, "plant_labels.json")
    bee_labels_path = os.path.join(base_path, "bee_labels.json")
    interaction_path = os.path.join(base_path, "interactions.csv")

    dm = PlantPollinatorDataModule(
        plant_dir,
        bee_dir,
        plant_labels_path,
        bee_labels_path,
        interaction_path,
        args
    )
    return dm

def setup_trainer_args():
    # Configure accelerator/devices
    slurm_nodes = os.environ.get("SLURM_JOB_NUM_NODES")
    num_nodes = int(slurm_nodes) if slurm_nodes is not None else 1

    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = torch.cuda.device_count()
        strategy = "ddp" if devices and devices > 1 or num_nodes > 1 else "auto"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
        strategy = "auto"
    else:
        accelerator = "cpu"
        devices = 1
        strategy = "auto"
    return accelerator, devices, strategy

def main(args):
    base_path = os.path.join("/scratch", "mcatchen", "iNatImages", "data") if args.cluster else "./data"

    datamodule = setup_data(base_path, args)

    model = InteractionPredictor(
        lr=args.lr, 
        model_type=args.model,
    )

    accelerator, devices, strategy = setup_trainer_args()

    log_path = os.path.join(base_path, "logs")
    logger = CSVLogger(log_path, name=os.environ.get("SLURM_JOB_NAME") or "InteractionTest")

    trainer = pl.Trainer(
        accelerator = accelerator,
        devices = devices,
        strategy = strategy,
        logger = logger,
        max_epochs = args.max_epochs,
        limit_train_batches = args.train_steps_per_epoch,
        enable_progress_bar=False 
    )
    trainer.fit(model, datamodule)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--samples_per_pair', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--train_steps_per_epoch', type=int, default=1000) 
    parser.add_argument('--holdout_bees', default=0.1)
    parser.add_argument('--holdout_plants', default=0.1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--toy', action='store_true')
    parser.add_argument('--persistent_workers', action='store_true')
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--model', default='base', choices=['base', 'large', 'huge'])

    args = parser.parse_args()
    main(args)

