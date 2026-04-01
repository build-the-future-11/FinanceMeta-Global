from __future__ import annotations

import copy
import torch
import torch.nn as nn


class TargetEncoder(nn.Module):
    """
    Target encoder for FI-JEPA.

    - Mirror of context encoder
    - No gradients
    - Updated ONLY via EMA
    """

    def __init__(self, context_encoder: nn.Module):
        super().__init__()

        # Deep copy to ensure identical architecture but separate params
        self.encoder = copy.deepcopy(context_encoder)

        # Freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Ensure eval mode by default (important!)
        self.encoder.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through frozen encoder.

        Args:
            x: [B, T, D]

        Returns:
            latent: [B, T, latent_dim]
        """
        return self.encoder(x)

    def train(self, mode: bool = True):
        """
        Override train() to ALWAYS stay in eval mode.
        Prevents accidental dropout/batchnorm drift.
        """
        super().train(False)
        return self

    def eval(self):
        return super().eval()