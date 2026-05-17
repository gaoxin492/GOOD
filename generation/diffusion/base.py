"""Base sampler interface for diffusion models."""

from abc import abstractmethod
from typing import List
import torch


class BaseSampler:
    """Abstract base class for diffusion model samplers."""

    def __init__(self, args):
        self.device = args.device
        self.seed = args.seed
        self.per_sample_batch_size = args.per_sample_batch_size

    @abstractmethod
    def sample(self, sample_size: int, guidance) -> torch.Tensor:
        """Generate samples with optional guidance.

        Args:
            sample_size: Number of samples to generate.
            guidance: A guidance object whose ``guide_step`` is called each
                denoising step.

        Returns:
            Tensor of generated samples.
        """
        pass

    @staticmethod
    def tensor_to_obj(tensor: torch.Tensor):
        """Convert a batch tensor into a list of domain objects (e.g. PIL images)."""
        pass

    @staticmethod
    def obj_to_tensor(objs: List) -> torch.Tensor:
        """Convert a list of domain objects back into a batch tensor."""
        pass
