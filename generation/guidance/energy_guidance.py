"""Image-level energy guidance for OOD sample generation.

Drives the diffusion sampling trajectory toward low-density regions of the
in-distribution by using the negative log-sum-exp (energy) of a pretrained
classifier's logits as the guidance signal.
"""

import torch
from models.resnet import ClassifierEnergy
from .utils import check_grad_fn, rescale_grad, ban_requires_grad


class ImageEnergyGuidance:
    """Energy-based (image-level) guidance using a pretrained ID classifier."""

    def __init__(self, guide_network, target, num_classes, load, dataset, device):
        """
        Args:
            guide_network: Architecture name (e.g. 'resnet34').
            target: 'OOD' to push samples away from ID, 'ID' to pull toward ID.
            num_classes: Number of in-distribution classes.
            load: Path to the pretrained classifier checkpoint.
            dataset: Dataset name ('imagenet100' or 'cifar100').
            device: Torch device.
        """
        self.guide_network = guide_network
        self.target = target
        self.num_classes = num_classes
        self.load = load
        self.dataset = dataset
        self.device = device
        self._load_model()

    def _load_model(self):
        self.model = ClassifierEnergy(
            self.guide_network, self.num_classes, self.load, self.dataset, self.device,
        )
        ban_requires_grad(self.model)

    @torch.enable_grad()
    def get_guidance(
        self, x_need_grad, func=lambda x: x, post_process=lambda x: x,
        return_logp=False, check_grad=True, **kwargs,
    ):
        """Compute energy guidance gradient (or log-probability).

        Args:
            x_need_grad: Input latent requiring gradients.
            func: Optional pre-processing function applied before post_process.
            post_process: Typically the VAE decoder (latent -> pixel).
            return_logp: If True, return the scalar log-probability instead
                of the gradient.
            check_grad: Whether to assert requires_grad on input.

        Returns:
            Gradient tensor (same shape as input) or scalar log-prob.
        """
        if check_grad:
            check_grad_fn(x_need_grad)

        x_need_grad = func(x_need_grad)
        x = post_process(x_need_grad)

        # For OOD generation we negate the energy so the gradient pushes
        # samples *away* from the ID distribution.
        direction = -1 if self.target == 'OOD' else 1
        log_probs = self.model(x) * direction

        if return_logp:
            return log_probs

        grad = torch.autograd.grad(log_probs.sum(), x_need_grad)[0]
        return rescale_grad(grad, clip_scale=1.0, **kwargs)
