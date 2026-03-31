from __future__ import annotations

import torch
import torch.nn as nn

from fi_jepa.models.operator_modules import StochasticOperatorBlock


class VolatilityOperator(StochasticOperatorBlock):
    """
    Captures volatility clustering dynamics.
    """

    def __init__(self, latent_dim: int):
        super().__init__(
            latent_dim=latent_dim,
            hidden_dim=latent_dim * 2,
            num_layers=2,
            dropout=0.1,
        )

        self.vol_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, z):

        z_next, mu, logvar = super().forward(z)

        volatility_amplifier = torch.exp(self.vol_scale)

        z_next = z + volatility_amplifier * (z_next - z)

        return z_next, mu, logvar