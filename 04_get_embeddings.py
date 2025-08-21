import os
import argparse

import torch
from lightning.pytorch.loggers import CSVLogger

import pytorch_lightning as pl

from src.model import VitClassifier
from src.dataset import WebDatasetDataModule
from src.checkpoints import AsyncTrainableCheckpoint 
from torchvision.datasets import ImageFolder
from collections import Counter

torch.set_float32_matmul_precision('high')


def get_num_images_per_class(data_folder="./data/bombus_img"):
    dataset = ImageFolder(root=data_folder)
    class_counts = Counter(dataset.targets)
    class2imgct =  {dataset.classes[k]: v for k,v in class_counts.items()}
    idx2class = {v: k for k,v in dataset.class_to_idx.items()}
    return idx2class, class2imgct

def load_model(ckpt_path, num_classes, model_type):
    vit = VitClassifier(num_classes=num_classes, model_type = model_type)
    vit = AsyncTrainableCheckpoint().load_trainable_checkpoint(vit, ckpt_path)
    return vit

def main(args):
    base_path = os.path.join("/scratch", "mcatchen", "iNatImages", "data") if args.cluster else "./data"
    data_dir_name = "bombus_wds" if args.species == "bees" else "plant_wds"
    img_dir_name = "bombus_img" if args.species == "bees" else "plant_img"

    wds_dir = os.path.join(base_path, data_dir_name)

    image_dir = os.path.join(base_path, img_dir_name)

    ckpt_filename =  "bombus.ckpt" if args.species == "bees" else "plant.ckpt"
    ckpt_path = os.path.join(base_path, ckpt_filename)

    species_data = WebDatasetDataModule(
        data_dir = wds_dir,
        batch_size = args.batch_size,
        num_workers= args.num_workers,
        persistent_workers = args.persistent_workers,
        prefetch_factor = args.prefetch_factor,
        species = args.species
    )
    species_data.setup('fit')


    num_classes = 19 if args.species == "bees" else 158

    net = load_model(ckpt_path, num_classes, args.model)
    net.to('cuda')

    idx2class, class2imgct = get_num_images_per_class(data_folder=image_dir)
    class2embeddings = {
        k : torch.zeros((v, net.hparams.embedding_dim)) 
        for k,v in class2imgct.items()
    }
    class2cursor = {k: 0 for k,v in class2imgct.items()}

    dl = species_data.train_dataloader()
    for x,y in dl:
        x = x.to('cuda')
        img_embed = net.image_model(x).pooler_output            
        embeddings = net.embedding_model(img_embed) 
        embeddings = embeddings.cpu()

        for (bi, classi) in enumerate(y):
            classname = idx2class[classi.item()]
            classcursor = class2cursor[classname]
            class2embeddings[classname][classcursor,:] = embeddings[bi,:] 
            class2cursor[classname] += 1

        
    torch.save(class2embeddings, os.path.join(base_path, args.species + "_embed.pt"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--persistent_workers', action='store_true')
    parser.add_argument('--species', default='bees', choices=['plants', 'bees'])
    parser.add_argument('--model', default='base', choices=['base', 'large', 'huge'])
    args = parser.parse_args()

    main(args)