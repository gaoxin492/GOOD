"""
Outlier Exposure (OE) fine-tuning: train an OOD detector using GOOD-generated
synthetic outliers alongside in-distribution data.

The training objective combines:
  1. Standard cross-entropy on ID samples.
  2. Energy-based regularisation via a logistic head on logsumexp energies.

Usage:
  python -m detection.train_ood_detector --dataset imagenet100 --data_root ./data \
      --load checkpoints/classifier.pt \
      --ood_data_energy logs_energy/ --ood_data_knn logs_knn/ \
      --ood_data all --energy_weight 1.0 --epochs 200
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
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from models.resnet import ResNet_Model
from datasets.dataset_gen import MultiRootImageDataset


# ---------------------------------------------------------------------------
# Normalization constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR_MEAN = [x / 255.0 for x in [125.3, 123.0, 113.9]]
CIFAR_STD = [x / 255.0 for x in [63.0, 62.1, 66.7]]


def get_args():
    parser = argparse.ArgumentParser(
        description="OE fine-tuning with GOOD-generated OOD samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset
    parser.add_argument("--dataset", type=str, default="imagenet100",
                        choices=["imagenet100", "cifar100"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--res", type=int, default=None,
                        help="Input resolution. Defaults to 224/32.")

    # Classifier
    parser.add_argument("--load", type=str, required=True,
                        help="Pretrained classifier checkpoint to fine-tune.")
    parser.add_argument("--model_arch", type=str, default="resnet34")

    # OOD data
    parser.add_argument("--ood_data_energy", type=str, default="",
                        help="Directory of energy-guided generated OOD images.")
    parser.add_argument("--ood_data_knn", type=str, default="",
                        help="Directory of KNN-guided generated OOD images.")
    parser.add_argument("--ood_data", type=str, default="all",
                        choices=["all", "energy", "knn"],
                        help="Which OOD sources to use.")
    parser.add_argument("--num_ood", type=int, default=None,
                        help="Max number of OOD samples (None = use all).")

    # Training
    parser.add_argument("--energy_weight", type=float, default=1.0,
                        help="Lambda for energy regularisation loss.")
    parser.add_argument("--T", type=float, default=1.0,
                        help="Temperature for energy computation.")
    parser.add_argument("--logistic_hidden", type=int, default=512,
                        help="Hidden size of the logistic regression head.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=0.0005)
    parser.add_argument("--save", type=str, default="checkpoints/detection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefetch", type=int, default=4)

    return parser.parse_args()


def build_id_loaders(args, transform_train, transform_test):
    """Build in-distribution train and validation loaders."""
    if args.dataset == "imagenet100":
        train_dir = os.path.join(args.data_root, "imagenet100", "train")
        val_dir = os.path.join(args.data_root, "imagenet100", "val")
        train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
        val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform_test)
    elif args.dataset == "cifar100":
        data_path = os.path.join(args.data_root, "CIFAR100")
        train_dataset = datasets.CIFAR100(data_path, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR100(data_path, train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.prefetch, pin_memory=True,
    )
    return train_loader, val_loader


def build_ood_loader(args, transform):
    """Build OOD data loader from generated images."""
    num = args.num_ood // 2 if args.num_ood and args.ood_data == "all" else args.num_ood

    if args.ood_data == "energy":
        ood_dataset = MultiRootImageDataset(args.ood_data_energy, transform=transform, num=num)
    elif args.ood_data == "knn":
        ood_dataset = MultiRootImageDataset(args.ood_data_knn, transform=transform, num=num)
    elif args.ood_data == "all":
        ds_energy = MultiRootImageDataset(args.ood_data_energy, transform=transform, num=num)
        ds_knn = MultiRootImageDataset(args.ood_data_knn, transform=transform, num=num)
        ood_dataset = ConcatDataset([ds_energy, ds_knn])
    else:
        raise ValueError(f"Unsupported ood_data choice: {args.ood_data}")

    print(f"OOD dataset size: {len(ood_dataset)}")
    return torch.utils.data.DataLoader(
        ood_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True,
    )


def get_transforms(args):
    """Return (transform_train, transform_test) appropriate for the dataset."""
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
    else:  # cifar100
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
    return transform_train, transform_test


def load_classifier(args, device):
    """Load the pretrained classifier and strip DataParallel prefix."""
    classifier = ResNet_Model(name=args.model_arch, num_classes=args.num_classes, device=str(device))
    state = torch.load(args.load, map_location=device, weights_only=True)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    classifier.load_state_dict(state, strict=True)
    return classifier


def train_epoch(classifier, logistic_head, train_loader_in, train_loader_out,
                optimizer, scheduler, device, args):
    """One epoch of OE training."""
    classifier.train()
    logistic_head.train()
    loss_avg = 0.0

    out_iter = iter(train_loader_out)

    for in_set in tqdm(train_loader_in, desc="  train", leave=False):
        # Get next OOD batch (cycle if exhausted)
        try:
            out_set = next(out_iter)
        except StopIteration:
            out_iter = iter(train_loader_out)
            out_set = next(out_iter)

        in_data, in_target = in_set[0], in_set[1]
        out_data = out_set[0]

        data = torch.cat([in_data, out_data], dim=0).to(device)
        target = in_target.to(device)

        # Random permutation for mixing
        perm = torch.randperm(len(data))
        binary_labels = torch.ones(len(data), device=device)
        binary_labels[len(in_data):] = 0

        logits = classifier(data[perm])

        # Energy scores
        Ec_in = torch.logsumexp(logits[binary_labels[perm].bool()], dim=1) / args.T
        Ec_out = torch.logsumexp(logits[(1 - binary_labels[perm]).bool()], dim=1) / args.T

        # ID classification loss
        fake_target = torch.cat([target, -torch.ones(len(out_data), device=device)], dim=0)
        loss_ce = F.cross_entropy(
            logits[binary_labels[perm].bool()],
            fake_target[perm][binary_labels[perm].bool()].long(),
        )

        # Energy regularisation
        energy_input = torch.cat([Ec_in, Ec_out], dim=0)
        energy_pred = logistic_head(energy_input.view(-1, 1))
        energy_labels = torch.cat([
            torch.ones(len(Ec_in), device=device),
            torch.zeros(len(Ec_out), device=device),
        ]).long()
        loss_energy = F.cross_entropy(energy_pred, energy_labels)

        loss = loss_ce + args.energy_weight * loss_energy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_avg = loss_avg * 0.8 + loss.item() * 0.2

    return loss_avg


@torch.no_grad()
def evaluate(classifier, val_loader, device):
    classifier.eval()
    correct, total, loss_avg = 0, 0, 0.0
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
    if args.res is None:
        args.res = 224 if args.dataset == "imagenet100" else 32

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save, f"log_{timestamp}_{args.ood_data}_lambda_{args.energy_weight}")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build data loaders
    transform_train, transform_test = get_transforms(args)
    train_loader_in, val_loader = build_id_loaders(args, transform_train, transform_test)
    train_loader_out = build_ood_loader(args, transform_train)

    # Model
    classifier = load_classifier(args, device)
    classifier = nn.DataParallel(classifier).to(device)

    # Logistic regression head for energy regularisation
    logistic_head = nn.Sequential(
        nn.Linear(1, args.logistic_hidden),
        nn.ReLU(),
        nn.Linear(args.logistic_hidden, 2),
    )
    logistic_head = nn.DataParallel(logistic_head).to(device)

    # Optimizer
    optimizer = torch.optim.SGD(
        list(classifier.parameters()) + list(logistic_head.parameters()),
        lr=args.learning_rate, momentum=args.momentum,
        weight_decay=args.decay, nesterov=True,
    )

    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step, args.epochs * len(train_loader_in), 1, 1e-6 / args.learning_rate,
        ),
    )

    # CSV log
    csv_path = os.path.join(save_dir, f"{args.dataset}_training_results.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,time(s),train_loss,val_loss,val_acc(%)\n")

    print("Beginning OE Training\n")
    for epoch in tqdm(range(1, args.epochs + 1), desc="epochs"):
        t0 = time.time()
        train_loss = train_epoch(
            classifier, logistic_head, train_loader_in, train_loader_out,
            optimizer, scheduler, device, args,
        )
        val_loss, val_acc = evaluate(classifier, val_loader, device)
        elapsed = int(time.time() - t0)

        with open(csv_path, "a") as f:
            f.write(f"{epoch:03d},{elapsed:05d},{train_loss:.6f},{val_loss:.5f},{100*val_acc:.2f}\n")

        print(f"Epoch {epoch:3d} | Time {elapsed:5d}s | "
              f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.3f} Acc {100*val_acc:.2f}%")

        if epoch % 10 == 0:
            path = os.path.join(save_dir, f"{args.dataset}_{args.model_arch}_{args.res}_model_{epoch}.pt")
            torch.save(classifier.state_dict(), path)

    last_path = os.path.join(save_dir, f"{args.dataset}_{args.model_arch}_{args.res}_last_model.pt")
    torch.save(classifier.state_dict(), last_path)
    print(f"Last model saved at {last_path}")


if __name__ == "__main__":
    main()
