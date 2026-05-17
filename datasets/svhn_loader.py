"""
SVHN dataset loaders for OOD evaluation.

Loads SVHN .mat files (``test_32x32.mat`` or ``selected_test_32x32.mat``)
and returns PIL images compatible with torchvision transforms.
"""

import os

import numpy as np
import torch.utils.data as data
from PIL import Image


class SVHN_test(data.Dataset):
    """Load a pre-selected SVHN test split from ``selected_test_32x32.mat``."""

    def __init__(self, root, split="test", transform=None, target_transform=None,
                 download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        import scipy.io as sio
        loaded = sio.loadmat(os.path.join(root, "selected_test_32x32.mat"))
        self.data = loaded["X"]
        self.targets = (loaded["y"] % 10).squeeze()
        self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index):
        img = Image.fromarray(np.transpose(self.data[index], (1, 2, 0)))
        target = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)


class SVHN(data.Dataset):
    """Standard SVHN loader (train / test / extra splits)."""

    split_list = {
        "train": ["train_32x32.mat"],
        "test": ["test_32x32.mat"],
        "extra": ["extra_32x32.mat"],
        "train_and_extra": ["train_32x32.mat", "extra_32x32.mat"],
    }

    def __init__(self, root, split="train", transform=None, target_transform=None,
                 download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

        if split not in self.split_list:
            raise ValueError(f"Invalid split '{split}'. Use one of {list(self.split_list.keys())}")

        import scipy.io as sio

        mat_files = self.split_list[split]
        data_parts, target_parts = [], []
        for fname in mat_files:
            loaded = sio.loadmat(os.path.join(root, fname))
            data_parts.append(loaded["X"])
            target_parts.append(loaded["y"])

        self.data = np.concatenate(data_parts, axis=3) if len(data_parts) > 1 else data_parts[0]
        self.targets = np.concatenate(target_parts, axis=0) if len(target_parts) > 1 else target_parts[0]
        self.targets = (self.targets % 10).squeeze()
        self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index):
        img = Image.fromarray(np.transpose(self.data[index], (1, 2, 0)))
        target = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)
