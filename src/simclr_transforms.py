# new file: simclr_transforms.py
import torchvision.transforms as transforms

def simclr_transforms(size=224):
    """
    Returns a data augmentation pipeline for SimCLR.
    
    Args:
        size (int): The size to which images are resized.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])