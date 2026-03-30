"""
dataset.py
Handles data loading, transformations, and attribute filtering for CelebA.
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os

# CelebA has 40 attributes. We map the requested concepts to exact CelebA attribute names.
CONCEPT_NAMES = [
    'Mouth_Slightly_Open', 'High_Cheekbones', 'Chubby', 'Narrow_Eyes', 
    'Bags_Under_Eyes', 'Big_Lips', 'Big_Nose', 'Pointy_Nose', 
    'Bushy_Eyebrows', 'Arched_Eyebrows'
]
TARGET_NAME = 'Smiling'

class CelebACustom(Dataset):
    def __init__(self, root, split='train', transform=None, download=True):
        self.celeba = datasets.CelebA(root=root, split=split, target_type='attr', 
                                        transform=transform, download=download)
        self.attr_names = self.celeba.attr_names
        
        # Find indices for concepts and target
        self.concept_idx = [self.attr_names.index(c) for c in CONCEPT_NAMES]
        self.target_idx = self.attr_names.index(TARGET_NAME)

    def __len__(self):
        return len(self.celeba)

    def __getitem__(self, idx):
        image, attrs = self.celeba[idx]
        concepts = attrs[self.concept_idx].float()
        target = attrs[self.target_idx].float()
        return image, concepts, target

def get_dataloaders(root_dir='./data', batch_size=64, num_workers=2):
    """
    Returns train, val, and test dataloaders for CelebA prepared for ResNet-18.
    """
    # Standard ImageNet normalization for ResNet-18
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
    ])

    # Download datasets
    train_ds = CelebACustom(root=root_dir, split='train', transform=transform)
    val_ds = CelebACustom(root=root_dir, split='valid', transform=transform)
    test_ds = CelebACustom(root=root_dir, split='test', transform=transform)

    # Load data into dataloaders
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, valloader, testloader