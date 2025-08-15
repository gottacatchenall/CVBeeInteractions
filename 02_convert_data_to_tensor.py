import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import os
import argparse

def convert_to_binary_dataset(dataset_path, output_dir, img_size=(32, 32)):
    """
    Converts an image dataset to a binary format optimized for GPU loading.

    Args:
        dataset_path (str): Path to the root directory of the image dataset.
                            The structure should be:
                            dataset_path/
                            ├── class_0/
                            │   ├── image1.jpg
                            │   └── image2.png
                            ├── class_1/
                            │   ├── image3.jpg
                            │   └── image4.gif
                            └── ...
        output_dir (str): Directory where the binary files will be saved.
        img_size (tuple): Desired size of the images (width, height).
    """
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(), # Converts to [C, H, W] tensor in [0, 1] range
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 2. Load the Dataset
    dataset = ImageFolder(root=dataset_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1028, shuffle=False)

    print(f"Found {len(dataset)} images in {dataset_path}")
    print(f"Number of classes: {len(dataset.classes)}")

    # 3. Process and Concatenate Tensors
    all_images = []
    all_labels = []

    for i, (images, labels) in enumerate(data_loader):
        print(f"Batch {i} of {len(data_loader)}")
        all_images.append(images)
        all_labels.append(labels)
        
    # Concatenate all batches into single, large tensors
    final_images = torch.cat(all_images, dim=0)
    final_labels = torch.cat(all_labels, dim=0)
    
    # 4. Save the Tensors using torch.save
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file_path = os.path.join(output_dir, 'dataset.pt')
    
    # Store data and labels in a dictionary for easy access
    dataset_dict = {
        'data': final_images,
        'labels': final_labels,
        'classes': dataset.classes,
    }
    print(f"Starting saving data")
    torch.save(dataset_dict, output_file_path)

    print(f"Dataset converted successfully! Tensors saved to: {output_file_path}")
    print(f"Data tensor shape: {final_images.shape}")
    print(f"Labels tensor shape: {final_labels.shape}")

"""
class TorchSavedDataset(Dataset):
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        print(f"Loading dataset from: {file_path}")
        
        # Load the entire dataset dictionary
        dataset_dict = torch.load(file_path)
        
        # Ensure the necessary keys are present
        if 'data' not in dataset_dict or 'labels' not in dataset_dict:
            raise ValueError("The .pt file must contain 'data' and 'labels' keys.")

        self.data = dataset_dict['data']
        self.labels = dataset_dict['labels']
        self.classes = dataset_dict['classes']

        # Ensure data and labels have the same number of samples
        if self.data.shape[0] != self.labels.shape[0]:
            raise ValueError("Number of samples in data and labels do not match.")
        self.num_samples = self.data.shape[0]
        print(f"Successfully loaded {self.num_samples} samples.")
       
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        return image, label
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert data to Tensor')
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--species', default='bees', choices=['plants', 'bees'])
    args = parser.parse_args()

    base_path = os.path.join("/scratch", "mcatchen", "iNatImages", "data") if args.cluster else "./data"
    datadir = "bombus_img" if args.species == "bees" else "plant_img"

    dataset_path = os.path.join(base_path, datadir)
    output_directory = os.path.join(base_path, './binary_' + args.species)

    if os.path.exists(dataset_path):
        convert_to_binary_dataset(dataset_path, output_directory, img_size=(224, 224))
    else:
        print(f"Error: Dataset path '{dataset_path}' not found. Please create a dummy dataset or point to a valid one.")

