from __future__ import annotations

import torch
import torch.nn as nn

from fi_jepa.models.operator_modules import StochasticOperatorBlock


class MacroOperator(StochasticOperatorBlock):
    """
    Slow regime transition operator.

    Captures macro regime shifts and long-horizon dynamics.
    """

    def __init__(self, latent_dim: int, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim or (latent_dim * 4),
            num_layers=3,
            dropout=dropout,
        )

        self.time_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, z):
        z_next, mu, logvar = super().forward(z)

        # Slow transition scaling
        z_next = z + self.time_scale * (z_next - z)

        return z_next, mu, logvar