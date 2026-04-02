from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from fi_jepa.predictors.macro_operator import MacroOperator
from fi_jepa.predictors.volatility_operator import VolatilityOperator
from fi_jepa.predictors.liquidity_operator import LiquidityOperator
from fi_jepa.predictors.residual_operator import ResidualOperator


class PredictorBank(nn.Module):
    """
    Mixture-of-operators latent dynamics module.

    Upgrades:
    - Multi-step prediction (temporal rollout)
    - Iterative latent evolution (world-model style)
    - Stable gating
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        operators: int = 4,
        horizon: int = 1,  # 🔥 NEW
    ):
        super().__init__()

        if operators != 4:
            raise ValueError("PredictorBank currently expects 4 operators.")

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim or (2 * latent_dim)
        self.horizon = horizon

        # ---------------------------------------------------------
        # Gating network (slightly more stable)
        # ---------------------------------------------------------

        self.gate = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, operators),
        )

        # ---------------------------------------------------------
        # Operators
        # ---------------------------------------------------------

        self.operators = nn.ModuleList([
            MacroOperator(latent_dim, self.hidden_dim, dropout),
            VolatilityOperator(latent_dim, self.hidden_dim, dropout),
            LiquidityOperator(latent_dim, self.hidden_dim, dropout),
            ResidualOperator(latent_dim, self.hidden_dim, dropout),
        ])

    # ---------------------------------------------------------
    # Utils
    # ---------------------------------------------------------

    @staticmethod
    def _latent_only(output):
        return output[0] if isinstance(output, tuple) else output

    @staticmethod
    def _routing_state(z: Tensor) -> Tensor:
        return z[:, -1] if z.dim() == 3 else z

    def _apply_operators(self, z: Tensor) -> list[Tensor]:
        outputs = []
        for op in self.operators:
            out = self._latent_only(op(z))
            if out.shape != z.shape:
                raise ValueError(
                    f"Operator output shape {tuple(out.shape)} "
                    f"does not match input shape {tuple(z.shape)}"
                )
            outputs.append(out)
        return outputs

    # ---------------------------------------------------------
    # Single-step transition
    # ---------------------------------------------------------

    def step(self, z: Tensor) -> Tensor:
        route_z = self._routing_state(z)

        # [B, num_ops]
        g = torch.softmax(self.gate(route_z), dim=-1)

        op_outputs = self._apply_operators(z)

        if z.dim() == 2:
            z_next = sum(
                g[:, i:i+1] * op_outputs[i]
                for i in range(len(op_outputs))
            )
            return z_next

        # z: [B, T, D]
        g = g.unsqueeze(1)  # [B, 1, num_ops]

        z_next = sum(
            g[..., i:i+1] * op_outputs[i]
            for i in range(len(op_outputs))
        )

        return z_next

    # ---------------------------------------------------------
    # Multi-step rollout (🔥 BIG UPGRADE)
    # ---------------------------------------------------------

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: [B, D] or [B, T, D]

        Returns:
            If horizon == 1:
                same shape as input
            Else:
                [B, H, D] or [B, H, T, D]
        """

        if self.horizon == 1:
            return self.step(z)

        preds = []
        z_curr = z

        for _ in range(self.horizon):
            z_curr = self.step(z_curr)
            preds.append(z_curr)

        return torch.stack(preds, dim=1)