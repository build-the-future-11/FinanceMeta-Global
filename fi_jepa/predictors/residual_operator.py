from __future__ import annotations

import torch
import torch.nn as nn

from fi_jepa.models.operator_modules import StochasticOperatorBlock


class ResidualOperator(StochasticOperatorBlock):
    """
    Residual operator capturing unexplained market microstructure effects.

    Enhancements:
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

        # Stabilization
        self.norm = nn.LayerNorm(latent_dim)

        # Base residual strength (log parameterization)
        self.log_base_scale = nn.Parameter(torch.log(torch.tensor(0.3)))

        # State-dependent gating
        self.adaptive_gate = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )

        # Uncertainty influence
        self.log_uncertainty_scale = nn.Parameter(torch.log(torch.tensor(0.1)))

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: [B, D] or [B, T, D]

        Returns:
            z_next, mu, sigma
        """

        z = self.norm(z)

        z_next, mu, logvar = super().forward(z)

        # Convert logvar → sigma
        sigma = torch.exp(0.5 * logvar)
        sigma = torch.clamp(sigma, min=1e-4, max=5.0)

        # Residual delta
        delta = z_next - z

        # State-aware gate
        gate = self.adaptive_gate(z)

        # Stable parameter transforms
        base_scale = torch.exp(self.log_base_scale)
        uncertainty_scale = torch.exp(self.log_uncertainty_scale)

        # Residual scaling
        scale = base_scale + gate * 0.5 + uncertainty_scale * sigma

        # Stabilize scale
        scale = torch.clamp(scale, 0.0, 2.5)

        # Residual update
        z_next = z + scale * delta

        return z_next, mu, sigma