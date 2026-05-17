"""Stable Diffusion sampler with text-conditioned generation for GOOD."""

import math
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List

from diffusers import StableDiffusionPipeline
from torchvision.transforms.functional import to_tensor

from .base import BaseSampler
from generation.guidance.base import BaseGuidance
from generation.utils.configs import Arguments
from generation.utils.prompts import get_prompt


class StableDiffusionSampler(BaseSampler):
    """Wraps a pretrained Stable Diffusion pipeline for guided sampling."""

    def __init__(self, args: Arguments):
        super().__init__(args)
        self.image_size = args.image_size
        self.inference_steps = args.inference_steps
        self.eta = args.eta
        self.generator = torch.manual_seed(self.seed)
        self.args = args
        self._build_diffusion(args)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _build_diffusion(self, args: Arguments):
        """Load the pretrained SD pipeline and prepare scheduler tensors."""
        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
            args.model_name_or_path
        ).to(self.device)

        self.scheduler = self.sd_pipeline.scheduler

        unet = self.sd_pipeline.unet
        unet.eval()
        for param in unet.parameters():
            param.requires_grad = False

        self.scheduler.set_timesteps(args.inference_steps)
        ts = self.scheduler.timesteps
        alpha_prod_ts = self.scheduler.alphas_cumprod[ts]
        alpha_prod_t_prevs = torch.cat(
            [alpha_prod_ts[1:], torch.ones(1) * self.scheduler.final_alpha_cumprod]
        )

        self.height = self.width = (
            self.sd_pipeline.unet.config.sample_size
            * self.sd_pipeline.vae_scale_factor
        )

        # Classifier-free guidance settings
        self.do_classifier_free_guidance = True
        self.guidance_scale = args.guidance_scale

        self.unet = unet
        self.ts = ts
        self.alpha_prod_ts = alpha_prod_ts
        self.alpha_prod_t_prevs = alpha_prod_t_prevs

    # ------------------------------------------------------------------
    # Prompt preparation
    # ------------------------------------------------------------------

    def _prepare_prompts(self, args: Arguments) -> List[str]:
        """Build text prompts for the current generation run."""
        prompt = get_prompt(args.dataset, args.target_class)
        prompts = [prompt] * args.num_samples
        print(f"Prompt: {prompt}")
        return prompts

    # ------------------------------------------------------------------
    # Decoding helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents to pixel-space images."""
        scaling = self.sd_pipeline.vae.config.scaling_factor
        return self.sd_pipeline.vae.decode(
            latents / scaling, return_dict=False, generator=self.generator
        )[0]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, sample_size: int, guidance: BaseGuidance) -> torch.Tensor:
        """Generate ``sample_size`` images using guided diffusion sampling.

        Args:
            sample_size: Total number of images to produce.
            guidance: Guidance object (e.g. TFGGuidance) for each denoising step.

        Returns:
            Tensor of shape ``(sample_size, 3, H, W)`` in ``[-1, 1]``.
        """
        self.prompts = self._prepare_prompts(self.args)

        all_samples = []
        n_batches = math.ceil(sample_size / self.per_sample_batch_size)

        for batch_id in range(n_batches):
            self.args.batch_id = batch_id

            batch_prompts = self.prompts[
                batch_id * self.per_sample_batch_size
                : min((batch_id + 1) * self.per_sample_batch_size, len(self.prompts))
            ]

            prompt_embeds, negative_prompt_embeds = self.sd_pipeline.encode_prompt(
                batch_prompts,
                self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            latents = self.sd_pipeline.prepare_latents(
                len(batch_prompts),
                self.sd_pipeline.unet.config.in_channels,
                self.height,
                self.width,
                prompt_embeds.dtype,
                self.device,
                generator=self.generator,
            )

            # Denoising loop with guidance
            for t in tqdm(range(self.inference_steps), total=self.inference_steps):

                def stable_diffusion_unet(latents_in, timestep):
                    latent_model_input = (
                        torch.cat([latents_in] * 2)
                        if self.do_classifier_free_guidance
                        else latents_in
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, timestep
                    )
                    noise_pred = self.unet(
                        latent_model_input, timestep,
                        encoder_hidden_states=prompt_embeds,
                    )[0]
                    if self.do_classifier_free_guidance:
                        uncond, text = noise_pred.chunk(2)
                        noise_pred = uncond + self.guidance_scale * (text - uncond)
                    return noise_pred

                latents = guidance.guide_step(
                    latents, t, stable_diffusion_unet,
                    self.ts, self.alpha_prod_ts, self.alpha_prod_t_prevs, self.eta,
                )

            image = self.decode(latents)
            all_samples.append(image.cpu())
            torch.cuda.empty_cache()

        return torch.cat(all_samples)

    # ------------------------------------------------------------------
    # Tensor ↔ PIL conversion
    # ------------------------------------------------------------------

    def tensor_to_obj(self, x: torch.Tensor) -> List[Image.Image]:
        """Convert a ``[-1, 1]`` image tensor to a list of PIL Images."""
        images = (x / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            return [Image.fromarray(img.squeeze(), mode="L") for img in images]
        return [Image.fromarray(img) for img in images]

    def obj_to_tensor(self, objs: List[Image.Image]) -> torch.Tensor:
        """Convert a list of PIL Images to a ``[-1, 1]`` tensor."""
        tensors = [to_tensor(img) for img in objs]
        return torch.stack(tensors).to(self.device) * 2 - 1
