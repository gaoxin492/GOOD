"""Configuration dataclass and helpers for GOOD generation pipeline."""

import os
import torch
from dataclasses import dataclass, field
from typing import Literal, Optional, List
from transformers import HfArgumentParser


@dataclass
class Arguments:
    """Arguments for the GOOD OOD sample generation pipeline."""

    # ---- Data ----
    data_type: Literal['text2image'] = field(
        default='text2image',
        metadata={'help': 'Data modality. Use text2image for Stable Diffusion.'},
    )
    dataset: str = field(
        default='imagenet100',
        metadata={'help': 'Dataset name: imagenet100 or cifar100.'},
    )
    task: str = field(
        default='image_energy',
        metadata={'help': 'Guidance task(s), joined by "+" for combined guidance '
                          '(e.g. "image_energy+image_knn").'},
    )
    K: int = field(
        default=100,
        metadata={'help': 'Number of nearest neighbours for KNN guidance.'},
    )

    # ---- Image ----
    image_size: int = field(default=512)

    # ---- Diffusion model ----
    model_name_or_path: str = field(
        default='stable-diffusion-v1-5',
        metadata={'help': 'Pretrained diffusion model path or HuggingFace ID.'},
    )
    train_steps: int = field(default=1000)
    inference_steps: int = field(default=50)
    eta: float = field(
        default=1.0,
        metadata={'help': 'DDIM eta (0 = deterministic, 1 = full stochastic).'},
    )
    clip_x0: bool = field(default=True)
    clip_sample_range: float = field(default=1.0)
    clip_scale: float = field(default=100.0)

    # ---- Inference ----
    seed: int = field(default=42)
    device: str = field(default='cuda')
    logging_dir: str = field(default='logs')
    per_sample_batch_size: int = field(default=4)
    num_samples: int = field(default=100)
    batch_id: int = field(default=0)

    # ---- Guidance ----
    guidance_name: str = field(
        default='tfg',
        metadata={'help': 'Guidance method name: tfg, no.'},
    )
    target: str = field(
        default='OOD',
        metadata={'help': 'Guidance target(s), joined by "+" for combined guidance.'},
    )
    num_classes: int = field(default=100)
    recur_steps: int = field(default=1)
    iter_steps: int = field(default=1)
    guidance_strength: float = field(default=1.0)
    target_class: str = field(
        default=None,
        metadata={'help': 'Current target class ID for generation.'},
    )
    classes: str = field(
        default=None,
        metadata={'help': 'Class range "start,end" or None for all classes.'},
    )
    load: str = field(
        default=None,
        metadata={'help': 'Path to the pretrained ID classifier checkpoint.'},
    )
    cache_dir: str = field(
        default='./cache',
        metadata={'help': 'Directory for KNN feature cache files.'},
    )

    # ---- Classifier-free guidance ----
    guidance_scale: float = field(
        default=7.5,
        metadata={'help': 'Classifier-free guidance scale for Stable Diffusion.'},
    )

    # ---- TFG hyperparameters ----
    rho: float = field(default=1.0, metadata={'help': 'TFG rho (x_t guidance weight).'})
    mu: float = field(default=1.0, metadata={'help': 'TFG mu (x_0 guidance weight).'})
    sigma: float = field(default=0.01, metadata={'help': 'TFG sigma (noise std for MC).'})
    eps_bsz: int = field(default=4, metadata={'help': 'MC noise batch size.'})
    rho_schedule: str = field(default='increase')
    mu_schedule: str = field(default='increase')
    sigma_schedule: str = field(default='decrease')

    # ---- Classifier / guide network ----
    guide_network: str = field(
        default='resnet34',
        metadata={'help': 'Guide network architecture(s), joined by "+" for combined '
                          'guidance (e.g. "resnet34+resnet34").'},
    )

    # ---- Misc ----
    check_done: bool = field(
        default=True,
        metadata={'help': 'Skip generation if output directory already has results.'},
    )

    # ---- Populated at runtime ----
    tasks: List[str] = field(default_factory=list, init=False, repr=False)
    guide_networks: List[str] = field(default_factory=list, init=False, repr=False)
    targets: List[str] = field(default_factory=list, init=False, repr=False)


def get_logging_dir(arg_dict: dict) -> str:
    """Construct logging subdirectory from key hyperparameters."""
    if arg_dict.get('guidance_name') == 'tfg':
        suffix = f"rho={arg_dict['rho']}+mu={arg_dict['mu']}+sigma={arg_dict['sigma']}"
    else:
        suffix = f"guidance_strength={arg_dict['guidance_strength']}"
    return os.path.join(arg_dict['logging_dir'], suffix)


def get_config() -> Arguments:
    """Parse command-line arguments into an Arguments instance."""
    args = HfArgumentParser([Arguments]).parse_args_into_dataclasses()[0]
    args.device = torch.device(args.device)
    args.logging_dir = get_logging_dir(vars(args))

    # Split combined guidance specifications
    args.tasks = args.task.split('+')
    args.guide_networks = args.guide_network.split('+')
    args.targets = args.target.split('+')
    assert len(args.tasks) == len(args.guide_networks) == len(args.targets), (
        "The number of tasks, guide_networks, and targets must match when using "
        "combined guidance (separated by '+')."
    )

    print(f"Logging to: {args.logging_dir}")
    return args
