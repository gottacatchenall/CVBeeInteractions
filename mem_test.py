import torchvision 
import argparse
import os
import random
import torchvision.utils
import torch
import shutil
from torchvision.io import decode_image


def get_dict_size(d):
    GB_PER_BIT = 1.25e-10
    BITS_PER_NUM =32
    return sum([sum([BITS_PER_NUM * torch.prod(torch.tensor(y.shape)) for y in x]) for x in d.values()]).item() *GB_PER_BIT


def copy_images(d, target_dir):
    # d is a dict with pairs sp_name: [path1, path2, ...]
    for sp_name, paths in d.items():
        sp_dir = os.path.join(target_dir, sp_name)
        os.makedirs(sp_dir, exist_ok=True)
        for p in paths:
            shutil.copyfile(p, os.path.join(sp_dir, p.split("/")[-1]))


def select_random_paths_per_class(ds):
    d = {}
    for class_num in range(len(ds.classes)):
        k = random.randint(20, 30)
        sp_name = ds.classes[class_num]

        d[sp_name] = random.choices([x[0] for x in ds.imgs if x[1] == class_num], k=k)
    return d


def create_toy_dataset():
    #base_path = os.path.join("/scratch", "mcatchen", "iNatImages", "data") if args.cluster else "./data"
    base_path = "./data"
    plant_dir, bee_dir = "plant_img", "bombus_img"

    PLANT_BASE_DIR = os.path.join(base_path, plant_dir)
    BEE_BASE_DIR = os.path.join(base_path, bee_dir)


    plant_ds = torchvision.datasets.ImageFolder(PLANT_BASE_DIR)
    bee_ds = torchvision.datasets.ImageFolder(BEE_BASE_DIR)

    toy_bee_folder = os.path.join(base_path, "toy_bee_img")
    os.makedirs(toy_bee_folder, exist_ok=True)
    d = select_random_paths_per_class(bee_ds)
    copy_images(d, toy_bee_folder)

    toy_plant_folder = os.path.join(base_path, "toy_plant_img")
    os.makedirs(toy_plant_folder, exist_ok=True)
    os.makedirs(toy_plant_folder, exist_ok=True)
    d = select_random_paths_per_class(plant_ds)
    copy_images(d, toy_plant_folder)



def read_folder(dir):
    sp_names = os.listdir(dir)
    imgs = {}
    for sp_name in sp_names:
        sp_dir = os.path.join(dir, sp_name)
        imgs[sp_name] = [decode_image(os.path.join(sp_dir, p)) for p in os.listdir(sp_dir)]
    return imgs

def main(args):
    base_path = os.path.join("/scratch", "mcatchen", "iNatImages", "data") if args.cluster else "./data"

    if args.toy:
        bee_dir = os.path.join(base_path, "toy_bee_img")
        plant_dir = os.path.join(base_path, "toy_plant_img")
    else:
        bee_dir = os.path.join(base_path, "bombus_img")
        plant_dir = os.path.join(base_path, "plant_img")

    
    bee_img = read_folder(bee_dir)
    print(f"Bee loaded size: {get_dict_size(bee_img)}")
    plant_img = read_folder(plant_dir)
    print(f"Plant loaded size: {get_dict_size(plant_img)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--toy', action='store_true')
    args = parser.parse_args()
    main(args)

