"""Unified OOD sample generation script for ImageNet-100 and CIFAR-100.

Usage examples:

    # Generate energy-guided OOD samples for ImageNet-100 (classes 0-99)
    python -m generation.generate \\
        --dataset imagenet100 --task image_energy --target OOD \\
        --guide_network resnet34 --num_classes 100 \\
        --load checkpoints/imagenet100_resnet34.pt \\
        --model_name_or_path stable-diffusion-v1-5 \\
        --guidance_name tfg --rho 0.5 --mu 0.5 --sigma 0.1 \\
        --num_samples 100 --per_sample_batch_size 4 \\
        --logging_dir logs_energy_in100

    # Generate KNN-guided OOD samples for CIFAR-100 (classes 0-99)
    python -m generation.generate \\
        --dataset cifar100 --task image_knn --target OOD \\
        --guide_network resnet34 --num_classes 100 \\
        --load checkpoints/cifar100_resnet34.pt \\
        --cache_dir ./cache \\
        --guidance_name tfg --rho 1.0 --mu 1.0 --sigma 0.001 \\
        --num_samples 100 --logging_dir logs_knn_cifar100
"""

import os
import time
import random
import hashlib

import torch
import numpy as np

from generation.utils.configs import get_config
from generation.utils.prompts import get_target_class_list
from generation.pipeline import BasePipeline
from generation.diffusion.stable_diffusion import StableDiffusionSampler
from generation.guidance.base import BaseGuidance
from generation.guidance.tfg import TFGGuidance


def generate_strong_seed() -> int:
    """Generate a high-entropy random seed from system sources."""
    entropy = f"{time.time_ns()}_{os.getpid()}_{os.urandom(16)}"
    return int(hashlib.sha256(entropy.encode()).hexdigest(), 16) % (2 ** 32)


def get_guidance(args, network):
    """Instantiate the appropriate guidance method."""
    noise_fn = getattr(network, 'noise_fn', None)
    if args.guidance_name == 'no':
        return BaseGuidance(args, noise_fn=noise_fn)
    elif 'tfg' in args.guidance_name:
        return TFGGuidance(args, noise_fn=noise_fn)
    else:
        raise NotImplementedError(f"Unknown guidance method: {args.guidance_name}")


def main():
    args = get_config()

    # Determine class range
    if args.classes:
        a, b = args.classes.split(',')
        class_indices = list(range(int(a), int(b)))
    else:
        class_indices = list(range(args.num_classes))

    target_class_list = get_target_class_list(args.dataset)
    print(f"Generating for classes: {class_indices}")

    # Build diffusion network and guidance
    network = StableDiffusionSampler(args)
    guider = get_guidance(args, network)

    for c in class_indices:
        args.target_class = target_class_list[c]
        print(f"\n=== Class {c}: {args.target_class} ===")

        pipeline = BasePipeline(args, network, guider)

        # Set a unique random seed per class
        seed = generate_strong_seed()
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        pipeline.network.generator = torch.manual_seed(seed)
        pipeline.network.seed = seed
        print(f"Seed: {seed}")

        samples = pipeline.sample(args.num_samples)
        if samples is None:
            continue

        # Save images
        folder_path = os.path.join(args.logging_dir, args.target_class)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Saving to: {folder_path}")

        for idx, img in enumerate(samples):
            img_path = os.path.join(
                folder_path, f"{args.target_class}_{idx + 1}.png",
            )
            img.save(img_path)

        torch.cuda.empty_cache()

    print("\nGeneration complete.")


if __name__ == '__main__':
    main()
