from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100

from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    ToTensor,
    Normalize,
    RandomCrop
)


def get_transform(train: bool = True) -> Compose:
    if train:
        return Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
    return Compose([
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])


def get_dataset(num_classes: int, train: bool = True) -> Dataset:
    assert num_classes in [10, 100]
    dataset_class = CIFAR10 if num_classes == 10 else CIFAR100
    transform = get_transform(train)
    return dataset_class(root='../data', download=True, train=train, transform=transform)


def get_loader(num_classes: int, train: bool = True, **loader_kwargs) -> DataLoader:
    return DataLoader(get_dataset(num_classes, train), **loader_kwargs)
