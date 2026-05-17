"""Base guidance method implementing DDIM sampling without task-specific guidance."""

import torch
from diffusers.utils.torch_utils import randn_tensor

from generation.guidance.guider import BaseGuider
from generation.utils.configs import Arguments


class BaseGuidance:
    """Vanilla DDIM sampling (no guidance) and shared helpers.

    Subclasses (e.g. ``TFGGuidance``) override ``guide_step`` to inject
    task-specific gradient guidance into the denoising trajectory.
    """

    def __init__(self, args: Arguments, noise_fn=None):
        self.args = args
        self.guider = BaseGuider(args)

        if noise_fn is None:
            self.generator = torch.manual_seed(args.seed)

            def _default_noise_fn(x, sigma, **kwargs):
                noise = randn_tensor(
                    x.shape, generator=self.generator,
                    device=args.device, dtype=x.dtype,
                )
                return sigma * noise + x

            self.noise_fn = _default_noise_fn
        else:
            self.noise_fn = noise_fn

    # ------------------------------------------------------------------
    # DDIM one-step update (no guidance)
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
        """Perform a single DDIM denoising step (baseline, no guidance)."""
        alpha_prod_t = alpha_prod_ts[t]
        alpha_prod_t_prev = alpha_prod_t_prevs[t]
        t_tensor = ts[t]

        for _ in range(self.args.recur_steps):
            eps = unet(x, t_tensor)
            x0 = self._predict_x0(x, eps, alpha_prod_t, **kwargs)
            x_prev = self._predict_x_prev_from_zero(
                x, x0, alpha_prod_t, alpha_prod_t_prev, eta, t_tensor, **kwargs,
            )
            x = self._predict_xt(x_prev, alpha_prod_t, alpha_prod_t_prev, **kwargs)

        return x_prev

    # ------------------------------------------------------------------
    # Shared prediction helpers
    # ------------------------------------------------------------------

    def _predict_x_prev_from_zero(
        self, xt, x0, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs,
    ) -> torch.Tensor:
        """Compute x_{t-1} from predicted x_0 via the DDIM update rule."""
        new_epsilon = (xt - alpha_prod_t ** 0.5 * x0) / (1 - alpha_prod_t) ** 0.5
        return self._predict_x_prev_from_eps(
            xt, new_epsilon, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs,
        )

    def _predict_x_prev_from_eps(
        self, xt, eps, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs,
    ) -> torch.Tensor:
        """DDIM Eq. (12): predict x_{t-1} from noise estimate epsilon."""
        sigma = eta * (
            (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
            * (1 - alpha_prod_t / alpha_prod_t_prev)
        ) ** 0.5

        pred_sample_direction = (1 - alpha_prod_t_prev - sigma ** 2) ** 0.5 * eps
        pred_x0_direction = (
            (xt - (1 - alpha_prod_t) ** 0.5 * eps) / alpha_prod_t ** 0.5
        )

        prev_sample = alpha_prod_t_prev ** 0.5 * pred_x0_direction + pred_sample_direction

        if eta > 0 and t.item() > 0:
            prev_sample = self.noise_fn(prev_sample, sigma, **kwargs)

        return prev_sample

    def _predict_xt(
        self, x_prev, alpha_prod_t, alpha_prod_t_prev, **kwargs,
    ) -> torch.Tensor:
        """Re-noise x_{t-1} back to x_t (for recurrence steps)."""
        xt_mean = (alpha_prod_t / alpha_prod_t_prev) ** 0.5 * x_prev
        return self.noise_fn(xt_mean, (1 - alpha_prod_t / alpha_prod_t_prev) ** 0.5, **kwargs)

    def _predict_x0(
        self, xt, eps, alpha_prod_t, **kwargs,
    ) -> torch.Tensor:
        """Predict the clean sample x_0 from x_t and noise estimate."""
        pred_x0 = (xt - (1 - alpha_prod_t) ** 0.5 * eps) / alpha_prod_t ** 0.5
        if self.args.clip_x0:
            pred_x0 = torch.clamp(
                pred_x0, -self.args.clip_sample_range, self.args.clip_sample_range,
            )
        return pred_x0
