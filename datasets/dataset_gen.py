"""
Dataset loader for GOOD-generated OOD images stored in a nested directory tree.

Expected layout:
    root_path/
        <param_folder_1>/
            <class_folder_a>/
                img_001.png
                img_002.png
            <class_folder_b>/
                ...
        <param_folder_2>/
            ...
"""

import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MultiRootImageDataset(Dataset):
    """Load generated OOD images from a two-level directory hierarchy.

    Args:
        root_path: Top-level directory containing parameter sub-folders,
                   each of which contains class sub-folders with .png images.
        transform: Torchvision transform applied to each image.
        num: If provided and smaller than total images, randomly sub-sample.
    """

    def __init__(self, root_path: str, transform=None, num: int = None):
        self.root_path = root_path
        self.transform = transform or transforms.ToTensor()
        self.num = num
        self.samples = []
        self.class_to_idx = {}
        self.offset = 0  # used by OE training for randomised starting point
        self._load_dataset()

    def _load_dataset(self):
        class_names = set()

        for subdir in sorted(os.listdir(self.root_path)):
            subdir_path = os.path.join(self.root_path, subdir)
            if not os.path.isdir(subdir_path):
                continue

            for class_dir in sorted(os.listdir(subdir_path)):
                class_path = os.path.join(subdir_path, class_dir)
                if not os.path.isdir(class_path):
                    continue
                class_names.add(class_dir)

                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(class_path, img_name)
                        self.samples.append((img_path, class_dir))

        # Build class -> index mapping
        class_names = sorted(class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        # Convert class names to integer labels
        self.samples = [(path, self.class_to_idx[label]) for path, label in self.samples]

        # Random sub-sampling
        if self.num is not None and self.num < len(self.samples):
            self.samples = random.sample(self.samples, self.num)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if len(self.samples) == 0:
            raise IndexError("Dataset is empty — no images found.")
        # Support offset for OE training randomisation
        real_idx = (idx + self.offset) % len(self.samples)
        img_path, label = self.samples[real_idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label
