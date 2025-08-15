import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor

import numpy as np
import os
import argparse
"""
def convert_to_binary_dataset(
    dataset_path, 
    output_dir, 
    img_size=(32, 32),
    test_size = 0.2,
    random_seed=42, 
):
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "google/vit-base-patch16-224",
        local_files_only=True,
        use_fast=True
    )
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
    ])

    # 2. Load the Dataset
    dataset = ImageFolder(root=dataset_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1028, shuffle=False)

    print(f"Found {len(dataset)} images in {dataset_path}")
    print(f"Number of classes: {len(dataset.classes)}")


    idx = 50
    pt = 1

    # 3. Process and Concatenate Tensors
    all_images = []
    all_labels = []

    for i, (images, labels) in enumerate(data_loader):
        if i < idx:
            print(f"Batch {i} of {len(data_loader)}")
            all_images.append(images)
            all_labels.append(labels)


    # Concatenate all batches into single, large tensors
    final_images = torch.cat(all_images, dim=0)
    final_labels = torch.cat(all_labels, dim=0)
    
    # Train/test split
    torch.manual_seed(random_seed)
    indices = torch.randperm(len(dataset))
    # Determine the split point
    split_point = int(len(dataset) * (1 - test_size))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    # 4. Save the Tensors using torch.save
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file_path = os.path.join(output_dir, f'dataset_{pt}.pt')
    
    # Store data and labels in a dictionary for easy access
    dataset_dict = {
        'data': final_images,
        'labels': final_labels,
        'classes': dataset.classes,
        'train_indices': train_indices,
        'test_indices': test_indices
    }
    print(f"Starting saving data")
    torch.save(dataset_dict, output_file_path)

    print(f"Dataset converted successfully! Tensors saved to: {output_file_path}")
    print(f"Data tensor shape: {final_images.shape}")
    print(f"Labels tensor shape: {final_labels.shape}")
"""


# A custom dataset to get a slice of the main dataset
class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def convert_specific_chunk(
    dataset_path, output_dir, chunk_to_save_idx, samples_per_chunk=1028, img_size=(32, 32), test_size = 0.2, random_seed = 42
):
    """
    Converts an image dataset into multiple smaller PyTorch files.

    Args:
        dataset_path (str): Path to the root directory of the image dataset.
        output_dir (str): Directory where the PyTorch files will be saved.
        chunk_size_gb (int): The target size of each saved file in gigabytes.
        img_size (tuple): Desired size of the images (height, width).
    """
    # 1. Define Image Transformations
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # 2. Load the Dataset but don't iterate yet
    full_dataset = ImageFolder(root=dataset_path, transform=transform)
    total_samples = len(full_dataset)
    print(f"Found {total_samples} images in {dataset_path}")

    # 3. Calculate start and end indices for the specific chunk    
    start_idx = chunk_to_save_idx * samples_per_chunk
    end_idx = min((chunk_to_save_idx + 1) * samples_per_chunk, total_samples)
    
    # Get the indices for the specific chunk
    chunk_indices = list(range(start_idx, end_idx))
    
    print(f"Processing chunk {chunk_to_save_idx}: Samples from {start_idx} to {end_idx-1}")
    
    # 4. Create a DataLoader for only this chunk
    chunk_dataset = SubsetDataset(full_dataset, chunk_indices)
    chunk_loader = DataLoader(chunk_dataset, batch_size=256, shuffle=False, num_workers=4)

    # 5. Process and Save the Chunk
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_images = []
    all_labels = []

    for images, labels in chunk_loader:
        all_images.append(images)
        all_labels.append(labels)

    # Concatenate all batches for the chunk
    chunk_images = torch.cat(all_images, dim=0)
    chunk_labels = torch.cat(all_labels, dim=0)

    torch.manual_seed(random_seed)
    indices = torch.randperm(len(all_labels))
    split_point = int(len(all_labels) * (1 - test_size))
    train_onehot = torch.nn.functional.one_hot(indices[:split_point],num_classes=len(all_labels))


    # Define the output file path for the current chunk
    output_file_path = os.path.join(
        output_dir, f"dataset_chunk_{chunk_idx}.pt"
    )
    dataset_dict = {
        "data": chunk_images,
        "labels": chunk_labels,
        "classes": full_dataset.classes,
        "train": train_onehot
    }
    torch.save(dataset_dict, output_file_path)
    print(
        f"Saved chunk {chunk_idx} with {len(chunk_images)} samples to {output_file_path}"
    )
    print("Dataset conversion complete.")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert data to Tensor')
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--species', default='bees', choices=['plants', 'bees'])
    args = parser.parse_args()

    base_path = os.path.join("/scratch", "mcatchen", "iNatImages", "data") if args.cluster else "./data"
    datadir = "bombus_img" if args.species == "bees" else "plant_img"

    dataset_path = os.path.join(base_path, datadir)
    output_directory = os.path.join(base_path, './binary_' + args.species)

    # --- Parameters for your run ---
    TARGET_CHUNK_SIZE_GB = 12
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found. Please provide a valid path.")
    else:
        # Load a dummy dataset to get the total number of samples
        full_dataset_dummy = ImageFolder(root=dataset_path, transform=None)
        total_samples = len(full_dataset_dummy)

        # Calculate samples per chunk and total chunks
        img_size = (224, 224) # Must match the size used in the conversion function
        single_image_size_bytes = img_size[0] * img_size[1] * 3 * 4
        chunk_size_bytes = TARGET_CHUNK_SIZE_GB * (1024**3)
        samples_per_chunk = int(np.floor(chunk_size_bytes / single_image_size_bytes))
        
        if samples_per_chunk == 0:
            raise ValueError("Chunk size is too small to contain even one sample.")

        total_chunks = int(np.ceil(total_samples / samples_per_chunk))
        
        print(f"Total samples: {total_samples}")
        print(f"Samples per chunk: {samples_per_chunk:,}")
        print(f"Total chunks to be created: {total_chunks}")
        
        # Loop through each chunk and process it
        for chunk_idx in range(total_chunks):
            print(f"\n--- Starting conversion for Chunk {chunk_idx}/{total_chunks-1} ---")
            convert_specific_chunk(
                dataset_path, 
                output_directory, 
                chunk_to_save_idx=chunk_idx, 
                samples_per_chunk=samples_per_chunk,
                img_size=img_size
            )

