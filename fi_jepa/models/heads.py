from __future__ import annotations
import torch
import torch.nn as nn


class ReturnHead(nn.Module):
    """
    Predict next-step returns from latent state.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor):

        return self.net(z)


class VolatilityHead(nn.Module):
    """
    Predict volatility from latent state.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, z):

        return self.net(z)


class RegimeHead(nn.Module):
    """
    Market regime classification.
    """

    def __init__(
        self,
        latent_dim: int,
        num_regimes: int = 4,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, num_regimes),
        )

    def forward(self, z):

        return self.net(z)