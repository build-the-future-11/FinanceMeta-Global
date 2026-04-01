from __future__ import annotations

import torch
import torch.nn as nn

from fi_jepa.models.operator_modules import StochasticOperatorBlock


class ResidualOperator(StochasticOperatorBlock):
    """
    Residual operator capturing unexplained market microstructure effects.

    Enhanced with:
    - adaptive residual scaling
    - uncertainty-aware modulation
    - stability constraints
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

        # -------------------------------------------------
        # Learnable base scale
        # -------------------------------------------------
        self.base_scale = nn.Parameter(torch.tensor(0.3))

        # -------------------------------------------------
        # Adaptive gating (state-dependent residual strength)
        # -------------------------------------------------
        self.adaptive_gate = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),  # keeps scale bounded
        )

        # -------------------------------------------------
        # Uncertainty modulation (from logvar)
        # -------------------------------------------------
        self.uncertainty_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, z):
        """
        Args:
            z: [B, D] or [B, T, D]

        Returns:
            z_next, mu, logvar
        """

        z_next, mu, logvar = super().forward(z)

        # -------------------------------------------------
        # Residual delta
        # -------------------------------------------------
        delta = z_next - z

        # -------------------------------------------------
        # Adaptive gate (state-aware)
        # -------------------------------------------------
        gate = self.adaptive_gate(z)

        # -------------------------------------------------
        # Uncertainty modulation
        # higher variance → stronger residual
        # -------------------------------------------------
        uncertainty = torch.exp(0.5 * logvar)

        # -------------------------------------------------
        # Final scaling
        # -------------------------------------------------
        scale = (
            self.base_scale
            + gate * 0.5
            + self.uncertainty_scale * uncertainty
        )

        # Clamp for stability
        scale = torch.clamp(scale, 0.0, 2.0)

        # -------------------------------------------------
        # Apply residual update
        # -------------------------------------------------
        z_next = z + scale * delta

        return z_next, mu, logvar