import argparse
import os
import tarfile
import io
import glob
import argparse
import random
import json 



def write_class_to_tar(class_name, img_paths, outdir, name2label):
    """
    Write all images for a class into WebDataset-style tar shards.
    """
    os.makedirs(outdir, exist_ok=True)
    count = 0
    tar_path = os.path.join(outdir, f"{class_name}.tar")
    tar = tarfile.open(tar_path, "w")

    meta = {
        "label": name2label[class_name],
        "class_name": class_name,
    }

    for i, img_path in enumerate(img_paths):
        key = f"{i:05d}"
        # Add image
        with open(img_path, "rb") as f:
            data = f.read()
        info = tarfile.TarInfo(f"{key}.jpg")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        count += 1

        # JSON metadata
        json_bytes = json.dumps(meta).encode("utf-8")
        json_info = tarfile.TarInfo(name=f"{key}.json")
        json_info.size = len(json_bytes)
        tar.addfile(json_info, fileobj=io.BytesIO(json_bytes))
    tar.close()

def convert_imagefolder_to_wds(img_folder, outdir, name2label, toy=False):
    """
    Convert an ImageFolder-style dataset to per-class WebDataset shards.
    img_folder: path like ./plants or ./pollinators
    outdir: output directory for tar shards
    """
    #classes = [d for d in os.listdir(img_folder) if os.path.isdir(os.path.join(img_folder, d))]

    for cls in name2label.keys():
        cls_path = os.path.join(img_folder, cls)
        img_paths = glob.glob(os.path.join(cls_path, "*.jpg")) + glob.glob(os.path.join(cls_path, "*.png"))

        if toy:
            num_imgs_sampled = random.randint(64, 92)
            img_paths = img_paths[:num_imgs_sampled]

        print(f"Writing {len(img_paths)} images for class {cls} ...")
        write_class_to_tar(cls, img_paths, outdir, name2label)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Converts folders of bee and plant images to Webdataset (wds) format')
    parser.add_argument('--toy', action='store_true')

    args = parser.parse_args()

    with open(os.path.join("data", "plant_labels.json"), 'r') as f:
        plant_name2label = json.load(f)

    with open(os.path.join("data", "bee_labels.json"), 'r') as f:
        bee_name2label = json.load(f)

    if args.toy:
        bee_out_dir = "./data/cluster_toy_bee_wds"
        plant_out_dir = "./data/cluster_toy_plant_wds"
    else:
        bee_out_dir = "./data/bee_wds"
        plant_out_dir = "./data/plant_wds"
    convert_imagefolder_to_wds("./data/plant_img", plant_out_dir, plant_name2label, toy=args.toy)
    convert_imagefolder_to_wds("./data/bombus_img", bee_out_dir, bee_name2label, toy=args.toy)
