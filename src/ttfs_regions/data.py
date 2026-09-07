from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .utils import cpu_generator


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    raw_dim: int
    num_classes: int = 10


def dataset_info(name: str) -> DatasetInfo:
    key = name.lower().replace("-", "").replace("_", "")
    if key in {"mnist", "fakemnist"}:
        return DatasetInfo("mnist" if key == "mnist" else "fake_mnist", 28 * 28)
    if key in {"cifar10", "fakecifar10"}:
        return DatasetInfo("cifar10" if key == "cifar10" else "fake_cifar10", 3 * 32 * 32)
    raise ValueError(f"Unknown dataset: {name}")


def _fake_dataset(raw_kind: str, train: bool, size: int = 256):
    if raw_kind == "mnist":
        image_size = (1, 28, 28)
    else:
        image_size = (3, 32, 32)
    return datasets.FakeData(
        size=size,
        image_size=image_size,
        num_classes=10,
        transform=transforms.ToTensor(),
        random_offset=0 if train else 10_000,
    )


def load_region_dataset(name: str, root: str | Path, train: bool = True, download: bool = True) -> Dataset:
    """Dataset convention for region experiments: ToTensor only."""
    info = dataset_info(name)
    if info.name == "fake_mnist":
        return _fake_dataset("mnist", train)
    if info.name == "fake_cifar10":
        return _fake_dataset("cifar10", train)
    root = str(Path(root).expanduser())
    transform = transforms.ToTensor()
    if info.name == "mnist":
        return datasets.MNIST(root=root, train=train, download=download, transform=transform)
    return datasets.CIFAR10(root=root, train=train, download=download, transform=transform)


def load_training_oriented_datasets(name: str, root: str | Path, download: bool = True):
    """Datasets used by the auxiliary training-oriented Init 3 arm.

    MNIST uses fixed normalization. CIFAR-10 additionally uses crop/flip
    augmentation for the training split, matching the attached notebooks.
    """
    info = dataset_info(name)
    if info.name.startswith("fake_"):
        base = "mnist" if "mnist" in info.name else "cifar10"
        ds = _fake_dataset(base, True)
        test = _fake_dataset(base, False)
        return ds, ds, test

    root = str(Path(root).expanduser())
    if info.name == "mnist":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ])
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ])
        cls = datasets.MNIST
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        cls = datasets.CIFAR10
    train_ds = cls(root=root, train=True, download=download, transform=train_transform)
    calib_ds = cls(root=root, train=True, download=download, transform=eval_transform)
    test_ds = cls(root=root, train=False, download=download, transform=eval_transform)
    return train_ds, calib_ds, test_ds


def sample_pairs(dataset: Dataset, num_pairs: int, seed: int = 2026) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [tuple(rng.sample(range(len(dataset)), 2)) for _ in range(int(num_pairs))]


def calibration_batch(dataset: Dataset, n: int, seed: int, device: str | torch.device | None = None, flatten: bool = False) -> torch.Tensor:
    n = min(int(n), len(dataset))
    idx = torch.randperm(len(dataset), generator=cpu_generator(seed))[:n]
    xs = []
    for i in idx.tolist():
        x, _ = dataset[i]
        xs.append(x.view(-1) if flatten else x)
    out = torch.stack(xs, dim=0)
    if device is not None:
        out = out.to(device)
    return out
