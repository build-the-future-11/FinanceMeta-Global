from __future__ import annotations

import torch
import torch.nn as nn


class StochasticOperatorBlock(nn.Module):
    """
    Base stochastic latent transition operator.

    Predicts a Gaussian latent transition:
        z_{t+1} = z_t + Δz
        Δz ~ N(mu, sigma^2)

    Supports both:
        - z: [B, D]
        - z: [B, T, D]
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim or (2 * latent_dim)

        layers = []
        dim = latent_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, self.hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            dim = self.hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.mu_head = nn.Linear(self.hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(self.hidden_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: latent state [B, D] or [B, T, D]

        Returns:
            z_next, mu, logvar
        """
        h = self.mlp(z)

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        delta = mu + eps * std
        z_next = self.norm(z + delta)

        return z_next, mu, logvar