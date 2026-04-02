from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fi_jepa.models.operator_modules import StochasticOperatorBlock


class ResidualOperator(StochasticOperatorBlock):
    """
    Residual operator capturing microstructure + unexplained effects.

    Upgrades:
    - multiplicative stable scaling
    - directional confidence gating
    - uncertainty-aware modulation
    - residual trust mechanism
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

        # Log-scale parameters (stable)
        self.log_base_scale = nn.Parameter(torch.log(torch.tensor(0.3)))
        self.log_uncertainty_scale = nn.Parameter(torch.log(torch.tensor(0.1)))

        # State-dependent magnitude gate
        self.mag_gate = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )

        # 🔥 NEW: directional confidence (trust residual direction)
        self.direction_gate = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),  # allows sign modulation
        )

        # 🔥 NEW: residual trust (should we even apply residual?)
        self.trust_gate = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor):
        """
        Returns:
            z_next, mu, sigma
        """

        z = self.norm(z)

        z_next_raw, mu, logvar = super().forward(z)

        # ---------------------------------------------------------
        # Uncertainty
        # ---------------------------------------------------------

        sigma = torch.exp(0.5 * logvar)
        sigma = torch.clamp(sigma, 1e-4, 5.0)

        # ---------------------------------------------------------
        # Residual delta
        # ---------------------------------------------------------

        delta = z_next_raw - z

        # ---------------------------------------------------------
        # Stable multiplicative scaling
        # ---------------------------------------------------------

        base_scale = torch.exp(self.log_base_scale)
        uncertainty_scale = torch.exp(self.log_uncertainty_scale)

        mag = self.mag_gate(z)

        scale = base_scale * (1 + mag) * (1 + uncertainty_scale * sigma)

        scale = torch.clamp(scale, 0.0, 3.0)

        # ---------------------------------------------------------
        # Directional modulation (🔥 BIG)
        # ---------------------------------------------------------

        direction = self.direction_gate(z)
        delta = delta * direction

        # ---------------------------------------------------------
        # Residual trust (🔥 VERY IMPORTANT)
        # ---------------------------------------------------------

        trust = self.trust_gate(z)  # [B, 1] or [B, T, 1]

        # ---------------------------------------------------------
        # Final update
        # ---------------------------------------------------------

        z_next = z + trust * scale * delta

        return z_next, mu, sigma