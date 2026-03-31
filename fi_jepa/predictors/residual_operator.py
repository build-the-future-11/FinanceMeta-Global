from __future__ import annotations

import torch
import torch.nn as nn

from fi_jepa.models.operator_modules import StochasticOperatorBlock


class ResidualOperator(StochasticOperatorBlock):
    """
    Residual operator capturing unexplained market microstructure effects.
    """

    def __init__(self, latent_dim: int):
        super().__init__(
            latent_dim=latent_dim,
            hidden_dim=latent_dim * 2,
            num_layers=2,
            dropout=0.1,
        )

        self.residual_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, z):

        z_next, mu, logvar = super().forward(z)

        z_next = z + self.residual_scale * (z_next - z)

        return z_next, mu, logvar