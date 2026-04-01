import torch
import torch.nn as nn
import torch.nn.functional as F


class LiquidityRegularizer(nn.Module):
    """
    Penalizes unrealistic latent transitions when liquidity is low.
    """

    def __init__(self, strength: float = 0.1):
        super().__init__()
        self.strength = strength

    def forward(
        self,
        z_prev: torch.Tensor,
        z_next: torch.Tensor,
        liquidity_gate: torch.Tensor,
    ) -> torch.Tensor:

        delta = z_next - z_prev

        move_mag = torch.norm(delta, dim=-1)

        # low liquidity should suppress moves
        penalty = (1.0 - liquidity_gate) * move_mag

        return self.strength * penalty.mean()