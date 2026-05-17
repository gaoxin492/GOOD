"""
Evaluate OOD detection performance using the unified scoring from GOOD.

Computes energy-based and KNN-based OOD scores on multiple test OOD datasets,
then fuses them with an adaptive weight derived from KL divergence:

    OOD_score = (1 - w) * normalised_energy + w * knn_distance
    where  w = 1 - exp(-a * KL(OOD_knn || ID_knn))

Reports FPR@95, AUROC per dataset and their averages.

Usage:
  python -m detection.evaluate --dataset imagenet100 --data_root ./data \
      --checkpoint checkpoints/detector.pt --ood_test_root ./test_data_all \
      --cache_dir ./cache --K 100 --a 0.5
"""

import argparse
import math
import os

import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from models.resnet import ResNet_Model
from utils.display_results import get_measures


# ---------------------------------------------------------------------------
# Normalization constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR_MEAN = [x / 255.0 for x in [125.3, 123.0, 113.9]]
CIFAR_STD = [x / 255.0 for x in [63.0, 62.1, 66.7]]


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate OOD detection with GOOD unified scoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="imagenet100",
                        choices=["imagenet100", "cifar100"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--res", type=int, default=None)
    parser.add_argument("--model_arch", type=str, default="resnet34")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ood_test_root", type=str, required=True,
                        help="Root dir containing OOD test dataset folders.")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Directory with cached KNN features.")
    parser.add_argument("--K", type=int, default=100, help="Number of KNN neighbors.")
    parser.add_argument("--a", type=float, default=0.5,
                        help="Adaptive weight parameter for KL-based fusion.")
    parser.add_argument("--T", type=float, default=1.0, help="Temperature for energy.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output", type=str, default="",
                        help="Path to write results (prints to stdout if empty).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def kl_divergence(p_samples: np.ndarray, q_samples: np.ndarray,
                  bins: int = 10, epsilon: float = 1e-9) -> float:
    """Estimate KL(P || Q) from samples via histogram binning."""
    all_samples = np.concatenate([p_samples, q_samples])
    bin_edges = np.histogram_bin_edges(all_samples, bins=bins)
    p_counts, _ = np.histogram(p_samples, bins=bin_edges)
    q_counts, _ = np.histogram(q_samples, bins=bin_edges)
    p_probs = (p_counts + epsilon) / (len(p_samples) + epsilon * bins)
    q_probs = (q_counts + epsilon) / (len(q_samples) + epsilon * bins)
    return float(np.sum(p_probs * np.log(p_probs / q_probs)))


def normalise(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Min-max normalise to [0, 1]."""
    return (x - lo) / (hi - lo + 1e-12)


def compute_knn_distance(features: torch.Tensor, reference: torch.Tensor,
                         K: int = 100) -> torch.Tensor:
    """Compute k-th nearest-neighbour L2 distance in feature space."""
    feat_norm = torch.nn.functional.normalize(features, p=2, dim=-1)
    distances = torch.cdist(feat_norm, reference, p=2)
    knn_dists, _ = torch.topk(distances, k=K, dim=-1, largest=False, sorted=True)
    return knn_dists[:, -1]  # k-th nearest distance


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_id_loader(args, transform):
    """Return a loader for the ID validation set."""
    if args.dataset == "imagenet100":
        val_dir = os.path.join(args.data_root, "imagenet100", "val")
        val_dataset = ImageFolder(val_dir, transform=transform)
    elif args.dataset == "cifar100":
        data_path = os.path.join(args.data_root, "CIFAR100")
        val_dataset = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    return torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )


def get_ood_loaders(args, transform):
    """Return a dict of OOD dataset loaders for evaluation."""
    root = args.ood_test_root
    loaders = {}

    if args.dataset == "imagenet100":
        ood_paths = {
            "texture": os.path.join(root, "dtd", "images"),
            "inat": os.path.join(root, "iNaturalist"),
            "places": os.path.join(root, "Places"),
            "sun": os.path.join(root, "SUN"),
        }
        for name, path in ood_paths.items():
            if os.path.isdir(path):
                loaders[name] = torch.utils.data.DataLoader(
                    ImageFolder(path, transform=transform),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True,
                )

    elif args.dataset == "cifar100":
        # Textures
        tex_path = os.path.join(root, "dtd", "images")
        if os.path.isdir(tex_path):
            loaders["textures"] = torch.utils.data.DataLoader(
                ImageFolder(tex_path, transform=transform),
                batch_size=args.batch_size, shuffle=False,
                num_workers=4, pin_memory=True,
            )
        # iSUN
        isun_path = os.path.join(root, "iSUN")
        if os.path.isdir(isun_path):
            loaders["isun"] = torch.utils.data.DataLoader(
                ImageFolder(isun_path, transform=transform),
                batch_size=args.batch_size, shuffle=False,
                num_workers=4, pin_memory=True,
            )
        # LSUN
        lsun_path = os.path.join(root, "LSUN")
        if os.path.isdir(lsun_path):
            loaders["lsun"] = torch.utils.data.DataLoader(
                ImageFolder(lsun_path, transform=transform),
                batch_size=args.batch_size, shuffle=False,
                num_workers=4, pin_memory=True,
            )
        # Places
        places_path = os.path.join(root, "Places")
        if os.path.isdir(places_path):
            loaders["places"] = torch.utils.data.DataLoader(
                ImageFolder(places_path, transform=transform),
                batch_size=args.batch_size, shuffle=False,
                num_workers=4, pin_memory=True,
            )
        # SVHN
        svhn_path = os.path.join(root, "svhn")
        if os.path.isdir(svhn_path):
            try:
                from datasets.svhn_loader import SVHN_test
                svhn_ds = SVHN_test(svhn_path, split="test", transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
                ]))
                loaders["svhn"] = torch.utils.data.DataLoader(
                    svhn_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True,
                )
            except Exception as e:
                print(f"Warning: could not load SVHN: {e}")

    return loaders


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_scores(model, loader, sampled_features, K, T, device):
    """Extract energy and KNN scores for all samples in a loader."""
    knn_scores, energy_scores = [], []
    for batch in tqdm(loader, leave=False):
        data = batch[0].to(device)
        feat, logits = model.forward_repre(data)
        knn = compute_knn_distance(feat, sampled_features, K)
        energy = T * torch.logsumexp(logits / T, dim=1)
        knn_scores.extend(knn.cpu().numpy())
        energy_scores.extend(energy.cpu().numpy())
    return np.array(knn_scores), np.array(energy_scores)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    if args.res is None:
        args.res = 224 if args.dataset == "imagenet100" else 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ResNet_Model(name=args.model_arch, num_classes=args.num_classes, device=str(device))
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Load KNN reference features
    feat_dim = {"resnet18": 512, "resnet34": 512, "resnet50": 2048,
                "vit_b16": 768, "convnext_base": 1024}[args.model_arch]
    cache_path = os.path.join(
        args.cache_dir,
        f"{args.dataset}_train_embedding_in_{args.model_arch}",
    )
    norm_path = os.path.join(cache_path, "feat_norm.pt")
    feat_norm = torch.load(norm_path, map_location=device, weights_only=True)
    sampled_features = feat_norm.to(device)

    # Build transforms
    if args.dataset == "imagenet100":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(args.res),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
        ])

    # ID validation scores
    id_loader = get_id_loader(args, transform)
    print("Extracting ID scores...")
    id_knn, id_energy = extract_scores(model, id_loader, sampled_features, args.K, args.T, device)
    id_energy_neg = -id_energy  # negate for "higher = more OOD"

    # OOD loaders
    ood_loaders = get_ood_loaders(args, transform)
    if not ood_loaders:
        print("No OOD datasets found. Check --ood_test_root.")
        return

    # Evaluate
    weight_fn = lambda kl: 1.0 - math.exp(-args.a * kl)
    all_fpr, all_auc = [], []

    lines = [f"{'Dataset':<15} {'FPR95':>8} {'AUROC':>8}"]
    lines.append("-" * 35)

    for name, loader in ood_loaders.items():
        print(f"Evaluating {name}...")
        ood_knn, ood_energy = extract_scores(model, loader, sampled_features, args.K, args.T, device)
        ood_energy_neg = -ood_energy

        # Adaptive weight from KL divergence
        kl = kl_divergence(ood_knn, id_knn)
        w = weight_fn(kl)

        # Normalise energy to [0, 1] for fusion
        all_energy = np.concatenate([id_energy_neg, ood_energy_neg])
        e_min, e_max = all_energy.min(), all_energy.max()

        ood_score = (1 - w) * normalise(ood_energy_neg, e_min, e_max) + w * ood_knn
        id_score = (1 - w) * normalise(id_energy_neg, e_min, e_max) + w * id_knn

        auroc, aupr, fpr = get_measures(ood_score, id_score)
        all_fpr.append(fpr)
        all_auc.append(auroc)

        lines.append(f"{name:<15} {100*fpr:>7.2f}% {100*auroc:>7.2f}%")

    avg_fpr = np.mean(all_fpr)
    avg_auc = np.mean(all_auc)
    lines.append("-" * 35)
    lines.append(f"{'Average':<15} {100*avg_fpr:>7.2f}% {100*avg_auc:>7.2f}%")

    result_text = "\n".join(lines)
    print("\n" + result_text)

    if args.output:
        with open(args.output, "a") as f:
            f.write(f"\nCheckpoint: {args.checkpoint}  K={args.K}  a={args.a}\n")
            f.write(result_text + "\n")
        print(f"Results appended to {args.output}")


if __name__ == "__main__":
    main()
