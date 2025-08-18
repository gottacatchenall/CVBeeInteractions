# new file: simclr_transforms.py
from torchvision.transforms import v2
import torch 

def simclr_transforms(size=224):
    """
    Returns a data augmentation pipeline for SimCLR.
    
    Args:
        size (int): The size to which images are resized.
    """
    return v2.Compose([
        v2.ToDtype(torch.uint8),
        v2.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        #v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])

