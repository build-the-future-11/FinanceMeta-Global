from __future__ import annotations

import torch
import torch.nn as nn

from fi_jepa.models.operator_modules import StochasticOperatorBlock


class LiquidityOperator(StochasticOperatorBlock):
    """
    Captures order flow and liquidity shocks.
    """

    def __init__(self, latent_dim: int):
        super().__init__(
            latent_dim=latent_dim,
            hidden_dim=latent_dim * 2,
            num_layers=2,
            dropout=0.1,
        )

        self.liquidity_gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):

        gate = self.liquidity_gate(z)

        z_next, mu, logvar = super().forward(z)

        z_next = z + gate * (z_next - z)

        return z_next, mu, logvar