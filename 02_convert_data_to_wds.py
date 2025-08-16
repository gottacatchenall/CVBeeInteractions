import os
from pathlib import Path
import tarfile
import json
import io
import random
import argparse

def iter_samples(root, extra_fields=None):
    """
    Yields (img_path, metadata_dict)
    extra_fields: dict of key -> function(img_path, class_name) to generate extra fields
    """
    if extra_fields is None:
        extra_fields = {}

    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    for cls in classes:
        for img_path in (root / cls).glob("*.jpg"):
            meta = {
                "label": class_to_idx[cls],
                "class_name": cls,
                "filename": img_path.name
            }
            for key, func in extra_fields.items():
                meta[key] = func(img_path, cls)
            yield img_path, meta

def _write_single_shard(batch, output_dir, shard_idx, prefix, key_start=0):
    shard_path = output_dir / f"{prefix}-{shard_idx:04d}.tar"
    with tarfile.open(shard_path, "w") as tar:
        for i, (img_path, meta) in enumerate(batch):
            key = f"{key_start + i:04d}"

            # Image
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            img_info = tarfile.TarInfo(name=f"{key}.jpg")
            img_info.size = len(img_bytes)
            tar.addfile(img_info, fileobj=io.BytesIO(img_bytes))

            # JSON metadata
            json_bytes = json.dumps(meta).encode("utf-8")
            json_info = tarfile.TarInfo(name=f"{key}.json")
            json_info.size = len(json_bytes)
            tar.addfile(json_info, fileobj=io.BytesIO(json_bytes))

    print(f"Written shard: {shard_path}")


def write_shards(samples, output_dir, prefix="train", shard_size=1000):
    shard_idx = 0
    batch = []
    for key_idx, (img_path, meta) in enumerate(samples):
        batch.append((img_path, meta))
        if len(batch) >= shard_size:
            _write_single_shard(batch, output_dir, shard_idx, prefix, key_start=key_idx - shard_size + 1)
            batch = []
            shard_idx += 1
    if batch:
        _write_single_shard(batch, output_dir, shard_idx, prefix, key_start=key_idx - len(batch) + 1)

def main(args):
    base_path = os.path.join("/scratch", "mcatchen", "iNatImages", "data") if args.cluster else "./data"

    image_dir = "bombus_img" if args.species == "bees" else "plant_img"
    output_dir = "bombus_wds" if args.species == "bees" else "plant_wds"


    image_dir_path = Path(base_path, image_dir) 
    output_dir_path = Path(base_path, output_dir)  
    output_dir_path.mkdir(exist_ok=True, parents=True)


    shard_size = args.shard_size     
    classes = sorted([d.name for d in image_dir_path.iterdir() if d.is_dir()])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    all_samples = []
    for cls in classes:
        for img_path in (image_dir_path / cls).glob("*.jpg"):
            meta = {
                "label": class_to_idx[cls],
                "class_name": cls,
                "filename": img_path.name
            }
            all_samples.append((img_path, meta))
    
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio              
    random_seed = args.seed

    random.seed(random_seed)
    random.shuffle(all_samples)

    train_split_idx = int(len(all_samples) * train_ratio)
    val_split_idx = int(len(all_samples) * (val_ratio + train_ratio))


    train_samples = all_samples[:train_split_idx]
    val_samples = all_samples[train_split_idx:val_split_idx]
    test_samples = all_samples[val_split_idx:]

    write_shards(train_samples, output_dir_path, prefix="train", shard_size=shard_size)
    write_shards(val_samples, output_dir_path, prefix="val", shard_size=shard_size)
    write_shards(test_samples, output_dir_path, prefix="test", shard_size=shard_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.2)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shard_size', type=int, default=1000)
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--species', default='bees', choices=['plants', 'bees'])
    args = parser.parse_args()
    main(args)





