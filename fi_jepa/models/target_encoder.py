from __future__ import annotations

import copy
import torch
import torch.nn as nn


class TargetEncoder(nn.Module):
    """
    Target encoder used in JEPA-style training.

    Characteristics
    ----------------
    • Mirror of the context encoder
    • Parameters never receive gradients
    • Updated only via EMA
    • Always runs in eval mode
    """

    def __init__(self, context_encoder: nn.Module):
        super().__init__()

        # Deep copy architecture + weights
        self.encoder = copy.deepcopy(context_encoder)

        # Freeze parameters
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder.eval()

    # -------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through frozen encoder.

        Args
        ----
        x : tensor
            shape (B, T, D)

        Returns
        -------
        latent : tensor
            shape (B, T, latent_dim)
        """

        return self.encoder(x)

    # -------------------------------------------------------------
    # EMA update
    # -------------------------------------------------------------

    @torch.no_grad()
    def ema_update(self, online_encoder: nn.Module, tau: float = 0.996):
        """
        Update target parameters via exponential moving average.

        θ_target ← τ θ_target + (1 − τ) θ_online
        """

        for t_param, o_param in zip(
            self.encoder.parameters(),
            online_encoder.parameters(),
        ):
            t_param.data.mul_(tau).add_(o_param.data, alpha=1 - tau)

    # -------------------------------------------------------------
    # Device sync
    # -------------------------------------------------------------

    def to(self, *args, **kwargs):
        self.encoder = self.encoder.to(*args, **kwargs)
        return self

    # -------------------------------------------------------------
    # Mode safety
    # -------------------------------------------------------------

    def train(self, mode: bool = True):
        """
        Prevent switching out of eval mode.
        """
        self.encoder.eval()
        return self

    def eval(self):
        self.encoder.eval()
        return self