"""Simplified sampling pipeline that wires together sampler and guidance."""

import os


class BasePipeline:
    """High-level pipeline: runs guided diffusion and returns PIL images."""

    def __init__(self, args, network, guider, evaluator=None):
        """
        Args:
            args: Parsed ``Arguments`` instance.
            network: A diffusion sampler (e.g. ``StableDiffusionSampler``).
            guider: A guidance object (e.g. ``TFGGuidance``).
            evaluator: Optional evaluator (unused in generation-only mode).
        """
        self.network = network
        self.guider = guider
        self.logging_dir = args.logging_dir
        self.check_done = args.check_done

    def sample(self, sample_size: int):
        """Generate ``sample_size`` images and return as a list of PIL Images.

        If ``check_done`` is True and outputs already exist, skips generation.
        """
        if self.check_done and os.path.exists(
            os.path.join(self.logging_dir, "finished_sampling")
        ):
            print("Found existing samples; skipping generation. "
                  "Set --check_done False to regenerate.")
            return None

        samples = self.network.sample(sample_size=sample_size, guidance=self.guider)
        samples = self.network.tensor_to_obj(samples)
        return samples
