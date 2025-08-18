# new file: simclr_transforms.py
import torchvision.transforms as transforms
import torch 

def simclr_transforms(size=224):
    """
    Returns a data augmentation pipeline for SimCLR.
    
    Args:
        size (int): The size to which images are resized.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])

