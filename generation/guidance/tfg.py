"""TFG (Training-Free Guidance) method for guided diffusion sampling.

Implements the core guidance mechanism from the TFG framework, adapted
for OOD sample generation in GOOD.  At each denoising step the method
applies two guidance signals:

* **Delta_t** -- gradient of the guidance loss w.r.t. x_t (controlled by rho).
* **Delta_0** -- iterative update on the predicted x_0 (controlled by mu).
"""

import math
import torch
from torch.autograd import grad
from functools import partial

from .base import BaseGuidance
from generation.guidance.utils import rescale_grad
from generation.utils.configs import Arguments


class TFGGuidance(BaseGuidance):
    """Training-Free Guidance for diffusion sampling."""

    def __init__(self, args: Arguments, **kwargs):
        super().__init__(args, **kwargs)
        self.device = args.device

    # ------------------------------------------------------------------
    # Monte-Carlo smoothed guidance
    # ------------------------------------------------------------------

    @torch.enable_grad()
    def tilde_get_guidance(self, x0, mc_eps, return_logp=False, **kwargs):
        """Compute guidance on x_0 with Monte-Carlo noise smoothing.

        Averages the guidance log-probability over ``mc_eps`` noise samples
        to estimate the smoothed distribution.
        """
        flat_x0 = (x0[None] + mc_eps).reshape(-1, *x0.shape[1:])
        outs = self.guider.get_guidance(
            flat_x0, return_logp=True, check_grad=False, **kwargs,
        )
        avg_logprobs = (
            torch.logsumexp(outs.reshape(mc_eps.shape[0], x0.shape[0]), dim=0)
            - math.log(mc_eps.shape[0])
        )

        if return_logp:
            return avg_logprobs

        _grad = torch.autograd.grad(avg_logprobs.sum(), x0)[0]
        _grad = rescale_grad(_grad, clip_scale=self.args.clip_scale, **kwargs)
        return _grad

    # ------------------------------------------------------------------
    # Noise / schedule helpers
    # ------------------------------------------------------------------

    def get_noise(self, std, shape, eps_bsz=4, **kwargs):
        """Sample Monte-Carlo noise vectors."""
        if std == 0.0:
            return torch.zeros((1, *shape), device=self.device)
        return torch.stack([
            self.noise_fn(torch.zeros(shape, device=self.device), std, **kwargs)
            for _ in range(eps_bsz)
        ])

    def get_rho(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        """Compute the x_t guidance weight rho at step t."""
        if self.args.rho_schedule == 'decrease':
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'increase':
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'constant':
            scheduler = torch.ones_like(alpha_prod_ts)
        else:
            raise ValueError(f"Unknown rho_schedule: {self.args.rho_schedule}")
        return self.args.rho * scheduler[t] * len(scheduler) / scheduler.sum()

    def get_mu(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        """Compute the x_0 guidance weight mu at step t."""
        if self.args.mu_schedule == 'decrease':
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.mu_schedule == 'increase':
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.mu_schedule == 'constant':
            scheduler = torch.ones_like(alpha_prod_ts)
        else:
            raise ValueError(f"Unknown mu_schedule: {self.args.mu_schedule}")
        return self.args.mu * scheduler[t] * len(scheduler) / scheduler.sum()

    def get_std(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        """Compute the MC noise standard deviation at step t."""
        if self.args.sigma_schedule == 'decrease':
            scheduler = (1 - alpha_prod_ts) ** 0.5
        elif self.args.sigma_schedule == 'constant':
            scheduler = torch.ones_like(alpha_prod_ts)
        else:
            raise ValueError(f"Unknown sigma_schedule: {self.args.sigma_schedule}")
        return self.args.sigma * scheduler[t]

    # ------------------------------------------------------------------
    # Main guided denoising step
    # ------------------------------------------------------------------

    def guide_step(
        self,
        x: torch.Tensor,
        t: int,
        unet: torch.nn.Module,
        ts: torch.LongTensor,
        alpha_prod_ts: torch.Tensor,
        alpha_prod_t_prevs: torch.Tensor,
        eta: float,
        **kwargs,
    ) -> torch.Tensor:
        """TFG guided denoising step combining Delta_t and Delta_0."""
        alpha_prod_t = alpha_prod_ts[t]
        alpha_prod_t_prev = alpha_prod_t_prevs[t]

        rho = self.get_rho(t, alpha_prod_ts, alpha_prod_t_prevs)
        mu = self.get_mu(t, alpha_prod_ts, alpha_prod_t_prevs)
        std = self.get_std(t, alpha_prod_ts, alpha_prod_t_prevs)

        t_tensor = ts[t]

        for _ in range(self.args.recur_steps):
            # Monte-Carlo noise for smoothed guidance
            mc_eps = self.get_noise(std, x.shape, self.args.eps_bsz, **kwargs)

            # --- Delta_t: gradient guidance on x_t ---
            if rho != 0.0:
                with torch.enable_grad():
                    x_g = x.clone().detach().requires_grad_()
                    x0 = self._predict_x0(
                        x_g, unet(x_g, t_tensor), alpha_prod_t, **kwargs,
                    )
                    logprobs = self.tilde_get_guidance(
                        x0, mc_eps, return_logp=True, **kwargs,
                    )
                    Delta_t = grad(logprobs.sum(), x_g)[0]
                    Delta_t = rescale_grad(
                        Delta_t, clip_scale=self.args.clip_scale, **kwargs,
                    )
                    Delta_t = Delta_t * rho
            else:
                Delta_t = torch.zeros_like(x)
                x0 = self._predict_x0(x, unet(x, t_tensor), alpha_prod_t, **kwargs)

            # --- Delta_0: iterative guidance on predicted x_0 ---
            new_x0 = x0.clone().detach()
            for _ in range(self.args.iter_steps):
                if mu != 0.0:
                    new_x0 = new_x0 + mu * self.tilde_get_guidance(
                        new_x0.detach().requires_grad_(), mc_eps, **kwargs,
                    )
            Delta_0 = new_x0 - x0

            # --- Combine and predict x_{t-1} ---
            alpha_t = alpha_prod_t / alpha_prod_t_prev
            x_prev = self._predict_x_prev_from_zero(
                x, x0, alpha_prod_t, alpha_prod_t_prev, eta, t_tensor, **kwargs,
            )
            x_prev += Delta_t / alpha_t ** 0.5 + Delta_0 * alpha_prod_t_prev ** 0.5

            # Re-noise for recurrence
            x = self._predict_xt(
                x_prev, alpha_prod_t, alpha_prod_t_prev, **kwargs,
            ).detach().requires_grad_(False)

        return x_prev
