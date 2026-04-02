from __future__ import annotations

import torch
import torch.nn as nn

from fi_jepa.models.operator_modules import StochasticOperatorBlock


class MacroOperator(StochasticOperatorBlock):

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim or (latent_dim * 4),
            num_layers=3,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(latent_dim)

        self.log_time_scale = nn.Parameter(torch.log(torch.tensor(0.05)))
        self.log_diffusion_scale = nn.Parameter(torch.log(torch.tensor(0.1)))
        self.log_dt = nn.Parameter(torch.log(torch.tensor(1.0)))

        self.regime_gate = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )

        self.persistence_gate = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 1),
            nn.Sigmoid(),
        )

        self.drift_direction = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
        )

        self.heavy_tail_scale = nn.Parameter(torch.tensor(1.2))

    def _low_frequency_component(self, delta, window: int = 5):
        if delta.dim() == 3:
            kernel = torch.ones(1, 1, window, device=delta.device) / window
            delta_t = delta.transpose(1, 2)
            smooth = torch.nn.functional.conv1d(
                delta_t,
                kernel,
                padding=window // 2,
                groups=1,
            )
            return smooth.transpose(1, 2)
        return delta

    def forward(self, z: torch.Tensor):

        z = self.norm(z)

        z_next_raw, mu, logvar = super().forward(z)

        sigma = torch.exp(0.5 * logvar)
        sigma = torch.clamp(sigma, 1e-4, 5.0)

        delta = z_next_raw - z

        delta = self._low_frequency_component(delta)

        time_scale = torch.exp(self.log_time_scale)
        diffusion_scale = torch.exp(self.log_diffusion_scale)
        dt = torch.exp(self.log_dt)

        regime = self.regime_gate(z)
        persistence = self.persistence_gate(z)

        drift_dir = self.drift_direction(z)
        delta = delta * drift_dir

        drift = time_scale * regime * delta

        heavy_tail = torch.tanh(self.heavy_tail_scale * delta)
        diffusion = diffusion_scale * sigma * heavy_tail * torch.sqrt(dt + 1e-6)

        z_next = z + persistence * drift + (1 - persistence) * diffusion

        return z_next, mu, sigma