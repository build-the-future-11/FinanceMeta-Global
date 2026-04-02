from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiquidityRegularizer(nn.Module):

    def __init__(
        self,
        sparsity_weight: float = 0.1,
        temporal_weight: float = 0.1,
        shock_weight: float = 0.2,
    ):
        super().__init__()

        self.sparsity_weight = sparsity_weight
        self.temporal_weight = temporal_weight
        self.shock_weight = shock_weight

    def forward(
        self,
        operator_weights: torch.Tensor,
        latent_sequence: torch.Tensor,
    ) -> torch.Tensor:

        w_liq = operator_weights[..., 2]

        sparsity_loss = w_liq.mean()

        if w_liq.dim() == 3:
            temporal_diff = torch.abs(w_liq[:, 1:] - w_liq[:, :-1])
            temporal_loss = temporal_diff.mean()
        else:
            temporal_loss = torch.tensor(0.0, device=w_liq.device)

        if latent_sequence is not None and latent_sequence.dim() == 3:
            returns = latent_sequence[:, 1:] - latent_sequence[:, :-1]
            shocks = torch.norm(returns, dim=-1)

            shocks = (shocks - shocks.mean()) / (shocks.std() + 1e-6)

            w_aligned = w_liq[:, 1:]

            shock_loss = -torch.mean(w_aligned * shocks)
        else:
            shock_loss = torch.tensor(0.0, device=w_liq.device)

        total = (
            self.sparsity_weight * sparsity_loss
            + self.temporal_weight * temporal_loss
            + self.shock_weight * shock_loss
        )

        return total
