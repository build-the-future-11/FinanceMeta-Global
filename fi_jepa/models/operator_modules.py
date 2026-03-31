from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticOperatorBlock(nn.Module):
    """
    Base stochastic latent transition operator.

    Each operator predicts parameters of a Gaussian latent transition:
        z_{t+1} = z_t + Δz

    where Δz ~ N(mu, sigma^2)

    This allows uncertainty propagation and prevents deterministic collapse.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        dim = latent_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: latent state (B, T, D)

        Returns:
            z_next: stochastic latent prediction
            mu: predicted mean
            logvar: predicted log variance
        """

        h = self.mlp(z)

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        delta = mu + eps * std

        z_next = self.norm(z + delta)

        return z_next, mu, logvar