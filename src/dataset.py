import torch
import os
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


# -------------------
# Dataset Module
# -------------------
class TorchSavedDataset(Dataset):
    def __init__(self, file_path, train=True):
        if "dataset.pt" not in file_path:
            file_path = os.path.join(file_path, "dataset.pt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        print(f"Loading dataset from: {file_path}")
        
        # Load the entire dataset dictionary
        dataset_dict = torch.load(file_path)
        
        # Ensure the necessary keys are present
        if 'data' not in dataset_dict or 'labels' not in dataset_dict:
            raise ValueError("The .pt file must contain 'data' and 'labels' keys.")

        idx = dataset_dict['train_indices'] if train else dataset_dict['test_indices']


        self.data = dataset_dict['data'].index_select(0, idx)
        self.labels = dataset_dict['labels'].index_select(0, idx)
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

class SpeciesImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir = "./", batch_size = 128, num_workers=0, transform = None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers

    def prepare_data(self):
        pass 

    def setup(self, stage):
        if stage == "fit":
            full = TorchSavedDataset(self.data_dir, train=True, transform=self.transform)
            self.train, self.val = torch.utils.data.random_split(
                full, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
            )
        if stage == "test":
            self.test = TorchSavedDataset(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.predict = TorchSavedDataset(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)