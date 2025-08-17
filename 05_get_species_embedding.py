import os
import argparse

import torch
from torch.utils.data import DataLoader

import argparse
from src.model import VitClassifier
from src.dataset import WebDatasetDataModule


import webdataset as wds
import glob 

torch.set_float32_matmul_precision('high')

def get_num_classes(ckpt):
    return ckpt["state_dict"]["classification_head.2.bias"].shape[0]


def load_model(ckpt, num_classes):
    num_classes = get_num_classes(ckpt)
    vit = VitClassifier(num_classes=num_classes)
    vit.load_state_dict(ckpt["state_dict"])
    return vit

def get_mean_embeddings(model, datamodule, num_classes, device):
    model.to(device)
    embed_sum = torch.zeros(num_classes, 128).to(device)
    class_ct = torch.zeros(num_classes).to(device)

    dataloader = datamodule.train_dataloader()

    batchct = 0
    for x,y in dataloader:
        x,y= x.to(device), y.to(device)
        e = model.embedding_model(model.image_model(x).pooler_output)
        for i,s in enumerate(y):
            embed_sum[s.item(),:] += e[i,:]
            class_ct[s.item()] += 1
        batchct += 1

    return (embed_sum.transpose(0,1) / class_ct).transpose(0,1)

def get_class_dict(image_dir, num_classes):
    def preprocess(data):
        img_bytes, meta = data
        class_label = meta["label"]
        class_name = meta["class_name"]
        return class_label, class_name

    train_dataset = (
        wds.WebDataset(glob.glob(f"{image_dir}/train*.tar"))
            .decode()
            .to_tuple("jpg", "json")
    ).map(preprocess)

    label_to_name = {}
    
    it = train_dataset.iterator()
    while len(label_to_name) < num_classes:
        l, n = next(it)
        label_to_name[l] = n
    return label_to_name

def main(args):
    base_path = os.path.join("/scratch", "mcatchen", "iNatImages", "data") if args.cluster else "./data"
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    data_dir = "bombus_wds" if args.species == "bees" else "plant_wds"
    image_dir = os.path.join(base_path, data_dir)

    
    species_data = WebDatasetDataModule(
        data_dir = image_dir
    )
    species_data.setup('fit')

    ckpt_filename =  "bombus.ckpt" if args.species == "bees" else "plant.ckpt"
    ckpt_path = os.path.join(base_path, ckpt_filename)
    ckpt = torch.load(ckpt_path, map_location=device)

    num_classes = get_num_classes(ckpt)
    model = load_model(ckpt, num_classes)
    model.to(device)

    class_dict = get_class_dict(image_dir, num_classes)

    emb = get_mean_embeddings(model, species_data, num_classes, device)
    emb_dict = {}
    for i in range(emb.shape[0]):
        emb_dict[class_dict[i]] = torch.tensor(emb[i,:])


    torch.save(emb_dict, os.path.join(base_path, args.species + "_embed.pt"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--species', default='bees', choices=['plants', 'bees'])
    args = parser.parse_args()
    main(args)

