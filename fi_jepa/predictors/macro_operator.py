from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fi_jepa.models.operator_modules import StochasticOperatorBlock


class MacroOperator(StochasticOperatorBlock):
    """
    Slow regime transition operator.

    Models long-horizon macroeconomic and regime dynamics
    in latent space. Transitions are intentionally slow
    to mimic structural market shifts.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim or (latent_dim * 4),
            num_layers=3,
            dropout=dropout,
        )

        # Learnable macro transition rate
        self.log_time_scale = nn.Parameter(torch.log(torch.tensor(0.1)))

        # Stabilization
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z: torch.Tensor):

        z = self.norm(z)

        z_next, mu, logvar = super().forward(z)

        # Convert logvar -> sigma for diagnostics
        sigma = torch.exp(0.5 * logvar)

        # Stabilize sigma
        sigma = torch.clamp(sigma, min=1e-4, max=5.0)

        # Slow regime dynamics
        time_scale = torch.exp(self.log_time_scale)

        z_next = z + time_scale * (z_next - z)

        return z_next, mu, sigma