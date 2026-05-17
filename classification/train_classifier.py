"""
Train an in-distribution (ID) classifier on ImageNet-100 or CIFAR-100.

The trained classifier is later used for:
  1. Energy-based and KNN-based guidance during OOD sample generation.
  2. Outlier-exposure fine-tuning for OOD detection.

Usage examples:
  # ImageNet-100
  python -m classification.train_classifier --dataset imagenet100 --data_root ./data \
      --res 224 --model_arch resnet34 --epochs 200 --learning_rate 0.001

  # CIFAR-100
  python -m classification.train_classifier --dataset cifar100 --data_root ./data \
      --res 32 --model_arch resnet34 --epochs 500 --learning_rate 0.1
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

from models.resnet import ResNet_Model


def get_args():
    parser = argparse.ArgumentParser(
        description="Train an ID classifier for OOD detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="imagenet100",
                        choices=["imagenet100", "cifar100"])
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory containing dataset folders.")
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--res", type=int, default=None,
                        help="Input resolution. Defaults to 224 (imagenet100) or 32 (cifar100).")
    parser.add_argument("--load", type=str, default="",
                        help="Checkpoint path to resume training from.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=160)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=0.0005)
    parser.add_argument("--model_arch", type=str, default="resnet34",
                        help="Model architecture (resnet34, resnet50, convnext_base, vit_b16).")
    parser.add_argument("--save", type=str, default="checkpoints/classification",
                        help="Directory to save checkpoints.")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--prefetch", type=int, default=4,
                        help="Number of data-loading workers.")
    return parser.parse_args()


# ---- ImageNet normalization ----
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---- CIFAR normalization ----
CIFAR_MEAN = [x / 255.0 for x in [125.3, 123.0, 113.9]]
CIFAR_STD = [x / 255.0 for x in [63.0, 62.1, 66.7]]


def build_dataloaders(args):
    """Return (train_loader, val_loader) for the chosen dataset."""
    if args.dataset == "imagenet100":
        res = args.res or 224
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(res),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(res),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        train_dir = os.path.join(args.data_root, "imagenet100", "train")
        val_dir = os.path.join(args.data_root, "imagenet100", "val")
        train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
        val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform_test)

    elif args.dataset == "cifar100":
        res = args.res or 32
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
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

    print(f"Dataset: {args.dataset}  |  train={len(train_dataset)}  val={len(val_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.prefetch, pin_memory=True,
    )
    return train_loader, val_loader


def train_epoch(classifier, train_loader, optimizer, scheduler, device):
    classifier.train()
    loss_avg, correct, total = 0.0, 0, 0
    for data, target in tqdm(train_loader, desc="  train", leave=False):
        data, target = data.to(device), target.to(device)
        logits = classifier(data)
        loss = F.cross_entropy(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        correct += logits.argmax(1).eq(target).sum().item()
        total += target.size(0)
        loss_avg = loss_avg * 0.8 + loss.item() * 0.2

    return loss_avg, correct / total


@torch.no_grad()
def evaluate(classifier, val_loader, device):
    classifier.eval()
    loss_avg, correct, total = 0.0, 0, 0
    for data, target in tqdm(val_loader, desc="  val", leave=False):
        data, target = data.to(device), target.to(device)
        logits = classifier(data)
        loss = F.cross_entropy(logits, target)

        correct += logits.argmax(1).eq(target).sum().item()
        total += target.size(0)
        loss_avg = loss_avg * 0.8 + loss.item() * 0.2

    return loss_avg, correct / total


def main():
    args = get_args()

    # Resolve default resolution
    if args.res is None:
        args.res = 224 if args.dataset == "imagenet100" else 32

    # Create timestamped save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save, f"log_{timestamp}_{args.model_arch}")
    os.makedirs(save_dir, exist_ok=True)

    # Save config
    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader = build_dataloaders(args)

    # Model
    classifier = ResNet_Model(name=args.model_arch, num_classes=args.num_classes, device=str(device))
    if args.load:
        state = torch.load(args.load, map_location=device, weights_only=True)
        # Remove 'module.' prefix from DataParallel checkpoints
        state = {k.replace("module.", ""): v for k, v in state.items()}
        classifier.load_state_dict(state, strict=True)
    classifier = nn.DataParallel(classifier).to(device)

    # Optimizer
    optimizer = torch.optim.SGD(
        classifier.parameters(), lr=args.learning_rate,
        momentum=args.momentum, weight_decay=args.decay, nesterov=True,
    )

    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step, args.epochs * len(train_loader), 1, 1e-6 / args.learning_rate,
        ),
    )

    # CSV log
    csv_path = os.path.join(save_dir, f"{args.dataset}_training_results.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,time(s),train_loss,train_acc(%),val_loss,val_acc(%)\n")

    # Save interval: every 10 epochs for imagenet100, every 50 for cifar100
    save_interval = 10 if args.dataset == "imagenet100" else 50

    print("Beginning Training\n")
    for epoch in tqdm(range(1, args.epochs + 1), desc="epochs"):
        t0 = time.time()
        train_loss, train_acc = train_epoch(classifier, train_loader, optimizer, scheduler, device)
        val_loss, val_acc = evaluate(classifier, val_loader, device)
        elapsed = int(time.time() - t0)

        with open(csv_path, "a") as f:
            f.write(f"{epoch:03d},{elapsed:05d},{train_loss:.6f},{100*train_acc:.2f},{val_loss:.5f},{100*val_acc:.2f}\n")

        print(f"Epoch {epoch:3d} | Time {elapsed:5d}s | "
              f"Train Loss {train_loss:.4f} Acc {100*train_acc:.2f}% | "
              f"Val Loss {val_loss:.3f} Acc {100*val_acc:.2f}%")

        if epoch % save_interval == 0:
            path = os.path.join(save_dir, f"{args.dataset}_{args.model_arch}_{args.res}_model_{epoch}.pt")
            torch.save(classifier.state_dict(), path)
            print(f"  -> saved {path}")

    # Save final model
    last_path = os.path.join(save_dir, f"{args.dataset}_{args.model_arch}_{args.res}_last_model.pt")
    torch.save(classifier.state_dict(), last_path)
    print(f"Last model saved at {last_path}")


if __name__ == "__main__":
    main()
