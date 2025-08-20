import os
import argparse

import torch
from lightning.pytorch.loggers import CSVLogger

import pytorch_lightning as pl

from src.dataset import WebDatasetDataModule
from src.checkpoints import AsyncTrainableCheckpoint 


torch.set_float32_matmul_precision('high')


# -------------------
# Main
# -------------------
def main(image_dir, log_path, args):
    species_data = WebDatasetDataModule(
        data_dir = image_dir,
        batch_size = args.batch_size,
        num_workers= args.num_workers,
        persistent_workers = args.persistent_workers,
        prefetch_factor = args.prefetch_factor,
        species = args.species
    )

    num_classes = 19 if args.species == "bees" else 158

    net = VitClassifier(
        lr=args.lr,
        num_classes=num_classes,
        model_type = args.model,
        use_supcon = args.contrastive,
        augmentation= args.augmentation,
        min_crop_size=args.min_crop_size
    )

    logger = CSVLogger(log_path, name=os.environ.get("SLURM_JOB_NAME"))

    checkpoint_cb = AsyncTrainableCheckpoint(
        dirpath = os.path.join(logger.log_dir, "checkpoints")
    )
    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES"))
    gpus = torch.cuda.device_count()

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        strategy='ddp',
        profiler="simple",
        logger=logger,
        enable_checkpointing=False,   # Turn off default ckpt
        callbacks=[checkpoint_cb],
        num_nodes=num_nodes, 
        max_epochs=args.max_epochs,
        enable_progress_bar=False
    )

    trainer.fit(net, species_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--min_crop_size', type=float, default=0.7)
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--contrastive', action='store_true')
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--persistent_workers', action='store_true')
    parser.add_argument('--species', default='bees', choices=['plants', 'bees'])
    parser.add_argument('--model', default='base', choices=['base', 'large', 'huge'])


    args = parser.parse_args()

    base_path = os.path.join("/scratch", "mcatchen", "iNatImages", "data") if args.cluster else "./data"
    data_dir = "bombus_wds" if args.species == "bees" else "plant_wds"

    image_dir = os.path.join(base_path, data_dir)
    log_path = os.path.join(base_path, "logs")
    main(image_dir, log_path, args)

