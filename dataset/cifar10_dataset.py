import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import CIFAR10_DATA_DIR

def get_cifar10_dataloaders(batch_size=128, test_batch_size=256):
    """
    Load CIFAR-10 dataset and return train/test dataloaders.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10(root=CIFAR10_DATA_DIR, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=CIFAR10_DATA_DIR, train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=False)
    
    return train_loader, test_loader
