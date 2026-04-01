from __future__ import annotations

import torch
import torch.nn as nn

from fi_jepa.models.operator_modules import StochasticOperatorBlock


class LiquidityOperator(StochasticOperatorBlock):
    """
    Captures order flow imbalance and liquidity shocks.

    Uses a gating mechanism to modulate latent transitions
    based on liquidity pressure.
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

        self.norm = nn.LayerNorm(latent_dim)

        # liquidity gating
        self.liquidity_gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor):

        z = self.norm(z)

        gate = self.liquidity_gate(z)

        z_next, mu, logvar = super().forward(z)

        sigma = torch.exp(0.5 * logvar)
        sigma = torch.clamp(sigma, min=1e-4, max=5.0)

        # gated dynamics
        z_next = z + gate * (z_next - z)

        return z_next, mu, sigma