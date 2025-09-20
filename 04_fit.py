import argparse 
import os
import torch

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from src.dataset import PlantPollinatorDataModule
from src.model import InteractionPredictor
from lightning.pytorch.loggers import CSVLogger


def setup_data(base_path, args, toy=False):
    plant_dir = "toy_plant_wds" if toy else "plant_img"
    bee_dir = "toy_bee_wds" if toy else "bombus_img"

    PLANT_BASE_DIR = os.path.join(base_path, plant_dir)
    BEE_BASE_DIR = os.path.join(base_path, bee_dir)
    METAWEB_PATH = os.path.join(base_path, "interactions.csv")

    dm = PlantPollinatorDataModule(
        PLANT_BASE_DIR,
        BEE_BASE_DIR,
        METAWEB_PATH,
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
        enable_progress_bar=False 
    )
    trainer.fit(model, datamodule)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--imgs_per_species', type=int, default=8) 
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



base_path = os.path.join("/scratch", "mcatchen", "iNatImages", "data") if args.cluster else "./data"

datamodule = setup_data(base_path, args)

model = InteractionPredictor(
    lr=args.lr, 
    model_type=args.model,
)

datamodule.setup()
dl = datamodule.train_dataloader()


batch = next(iter(dl))

model.shared_step(batch, 0)

