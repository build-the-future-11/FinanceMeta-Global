from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fi_jepa.models.operator_modules import StochasticOperatorBlock


class ResidualOperator(StochasticOperatorBlock):

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

        self.log_base_scale = nn.Parameter(torch.log(torch.tensor(0.25)))
        self.log_uncertainty_scale = nn.Parameter(torch.log(torch.tensor(0.15)))
        self.log_dt = nn.Parameter(torch.log(torch.tensor(1.0)))

        self.mag_gate = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )

        self.direction_gate = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
        )

        self.trust_gate = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 1),
            nn.Sigmoid(),
        )

        self.entropy_gate = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 1),
            nn.Sigmoid(),
        )

        self.heavy_tail_scale = nn.Parameter(torch.tensor(1.5))

    def _orthogonal_component(self, delta, z):
        proj = (delta * z).sum(dim=-1, keepdim=True) / (z.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
        return delta - proj * z

    def forward(self, z: torch.Tensor):

        z = self.norm(z)

        z_next_raw, mu, logvar = super().forward(z)

        sigma = torch.exp(0.5 * logvar)
        sigma = torch.clamp(sigma, 1e-4, 5.0)

        delta = z_next_raw - z

        delta = self._orthogonal_component(delta, z)

        base_scale = torch.exp(self.log_base_scale)
        uncertainty_scale = torch.exp(self.log_uncertainty_scale)
        dt = torch.exp(self.log_dt)

        mag = self.mag_gate(z)

        scale = base_scale * (1 + mag) * (1 + uncertainty_scale * sigma)
        scale = scale * torch.sqrt(dt + 1e-6)
        scale = torch.clamp(scale, 0.0, 3.0)

        direction = self.direction_gate(z)
        delta = delta * direction

        trust = self.trust_gate(z)

        entropy_proxy = torch.log(sigma + 1e-6)
        entropy_mod = self.entropy_gate(z) * entropy_proxy

        heavy_tail = torch.tanh(self.heavy_tail_scale * delta)

        drift = scale * delta
        diffusion = scale * sigma * heavy_tail

        z_next = z + trust * (drift + diffusion * (1 + entropy_mod))

        return z_next, mu, sigma