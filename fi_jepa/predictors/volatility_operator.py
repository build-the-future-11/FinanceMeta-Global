from __future__ import annotations

import torch
import torch.nn as nn

from fi_jepa.models.operator_modules import StochasticOperatorBlock


class VolatilityOperator(StochasticOperatorBlock):
    """
    Captures volatility clustering dynamics.

    Amplifies latent changes during high volatility regimes.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim or (latent_dim * 2),
            num_layers=2,
            dropout=dropout,
        )

        # log-scale for stability
        self.log_vol_scale = nn.Parameter(torch.zeros(1))

        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z: torch.Tensor):

        z = self.norm(z)

        z_next, mu, logvar = super().forward(z)

        # convert logvar -> sigma
        sigma = torch.exp(0.5 * logvar)
        sigma = torch.clamp(sigma, min=1e-4, max=5.0)

        # volatility amplification
        vol_scale = torch.exp(self.log_vol_scale)

        z_next = z + vol_scale * (z_next - z)

        return z_next, mu, sigma