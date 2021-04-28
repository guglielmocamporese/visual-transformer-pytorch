##################################################
# Imports
##################################################

import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


# Get datasets
def get_datasets(args):
    ds_args = {
        'root': './data',
        'download': True,
    }
    transform = transforms.ToTensor()
    transform_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dss = {
        'train_aug': CIFAR10(train=True, transform=transform_aug, **ds_args),
        'validation': CIFAR10(train=False, transform=transform_aug, **ds_args)
    }
    return dss

# Get dataloaders
def get_dataloaders(args):

    # Datasets
    dss = get_datasets(args)

    # Dataloaders
    dl_args = {
        'batch_size': args.batch_size,
        'pin_memory': True,
        'num_workers': args.num_workers,
    }
    dls = {
        'train_aug': DataLoader(dss['train_aug'], shuffle=True, **dl_args),
        'validation': DataLoader(dss['validation'], shuffle=False, **dl_args)
    }
    return dls


