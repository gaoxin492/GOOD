"""
Extract features from a trained ID classifier and cache them for KNN guidance.

Saves memory-mapped feature embeddings, logit scores, and labels, plus an L2-
normalised feature tensor for efficient KNN distance computation during generation.

Usage:
  python -m classification.extract_features --dataset imagenet100 --data_root ./data \
      --model_arch resnet34 --checkpoint checkpoints/classifier.pt --cache_dir ./cache
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models.resnet import ResNet_Model


def get_args():
    parser = argparse.ArgumentParser(
        description="Extract and cache features for KNN guidance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="imagenet100",
                        choices=["imagenet100", "cifar100"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--model_arch", type=str, default="resnet34",
                        help="Model architecture matching the checkpoint.")
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained classifier checkpoint.")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Base directory for cached features.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


# Feature dimensions per architecture
FEAT_DIM = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "convnext_base": 1024,
    "vit_b16": 768,
}

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CIFAR normalization
CIFAR_MEAN = [x / 255.0 for x in [125.3, 123.0, 113.9]]
CIFAR_STD = [x / 255.0 for x in [63.0, 62.1, 66.7]]


def build_dataloaders(args):
    """Return (train_loader, val_loader) for feature extraction."""
    if args.dataset == "imagenet100":
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        train_dir = os.path.join(args.data_root, "imagenet100", "train")
        val_dir = os.path.join(args.data_root, "imagenet100", "val")
        train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
        val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform_test)

    elif args.dataset == "cifar100":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
        ])
        data_path = os.path.join(args.data_root, "CIFAR100")
        train_dataset = datasets.CIFAR100(data_path, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR100(data_path, train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def extract_and_cache(model, loader, cache_dir, featdim, num_classes, batch_size, device):
    """Extract features, scores, and labels; save as memory-mapped files."""
    os.makedirs(cache_dir, exist_ok=True)

    n = len(loader.dataset)
    feat_log = np.memmap(os.path.join(cache_dir, "feat.mmap"), dtype=np.float32, mode="w+", shape=(n, featdim))
    score_log = np.memmap(os.path.join(cache_dir, "score.mmap"), dtype=np.float32, mode="w+", shape=(n, num_classes))
    label_log = np.memmap(os.path.join(cache_dir, "label.mmap"), dtype=np.int64, mode="w+", shape=(n,))

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            start = batch_idx * batch_size
            end = min(start + batch_size, n)

            out = model.features(inputs)
            if out.dim() > 2:
                out = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
            score = model.fc(out)

            feat_log[start:end] = out.cpu().numpy()
            score_log[start:end] = score.cpu().numpy()
            label_log[start:end] = targets.numpy()

            if batch_idx % 100 == 0:
                print(f"  [{batch_idx}/{len(loader)}]")

    # Compute and save L2-normalised features for KNN
    feat_tensor = torch.from_numpy(np.array(feat_log)).float()
    feat_norm = F.normalize(feat_tensor, p=2, dim=-1)
    torch.save(feat_norm, os.path.join(cache_dir, "feat_norm.pt"))
    print(f"  Saved feat_norm.pt ({feat_norm.shape})")


def main():
    args = get_args()

    torch.manual_seed(1)
    np.random.seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    featdim = FEAT_DIM[args.model_arch]

    # Load model
    model = ResNet_Model(name=args.model_arch, num_classes=args.num_classes, device=str(device))
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    train_loader, val_loader = build_dataloaders(args)

    for split, loader in [("train", train_loader), ("val", val_loader)]:
        cache_path = os.path.join(args.cache_dir, f"{args.dataset}_{split}_embedding_in_{args.model_arch}")
        print(f"Extracting {split} features -> {cache_path}")
        extract_and_cache(model, loader, cache_path, featdim, args.num_classes, args.batch_size, device)

    print("Done.")


if __name__ == "__main__":
    main()
