"""Shared utility functions for guidance modules."""

import torch


def ban_requires_grad(module: torch.nn.Module):
    """Freeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = False


def check_grad_fn(x: torch.Tensor):
    """Assert that a tensor has ``requires_grad=True``."""
    assert x.requires_grad, "Input tensor must have requires_grad=True"


def rescale_grad(grad: torch.Tensor, clip_scale: float, **kwargs) -> torch.Tensor:
    """Rescale gradients by clipping their per-sample scale.

    For standard image tensors the operation is a no-op unless the mean
    squared gradient exceeds ``clip_scale``.  The ``node_mask`` kwarg is
    kept for compatibility with molecule-level masking but is unused in
    image tasks.
    """
    node_mask = kwargs.get('node_mask', None)

    scale = (grad ** 2).mean(dim=-1)
    if node_mask is not None:
        scale = scale.sum(dim=-1) / node_mask.float().squeeze(-1).sum(dim=-1)
        clipped_scale = torch.clamp(scale, max=clip_scale)
        co_ef = clipped_scale / scale
        grad = grad * co_ef.view(-1, 1, 1)

    return grad
