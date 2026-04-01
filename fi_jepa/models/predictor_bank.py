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
    Mixture-of-operators latent transition module.

    Supports both:
        - z: [B, D]
        - z: [B, T, D]

    Each operator must return either:
        - Tensor
        - or a tuple whose first element is the predicted latent tensor
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        operators: int = 4,
    ):
        super().__init__()

        if operators != 4:
            raise ValueError(
                f"PredictorBank currently expects 4 operators, got {operators}."
            )

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim or (2 * latent_dim)
        self.operators = operators

        self.gate = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, operators),
        )

        self.macro = MacroOperator(
            latent_dim=latent_dim,
            hidden_dim=self.hidden_dim,
            dropout=dropout,
        )
        self.vol = VolatilityOperator(
            latent_dim=latent_dim,
            hidden_dim=self.hidden_dim,
            dropout=dropout,
        )
        self.liq = LiquidityOperator(
            latent_dim=latent_dim,
            hidden_dim=self.hidden_dim,
            dropout=dropout,
        )
        self.res = ResidualOperator(
            latent_dim=latent_dim,
            hidden_dim=self.hidden_dim,
            dropout=dropout,
        )

    @staticmethod
    def _latent_only(output):
        if isinstance(output, tuple):
            return output[0]
        return output

    @staticmethod
    def _routing_state(z: Tensor) -> Tensor:
        if z.dim() == 3:
            return z[:, -1, :]
        if z.dim() == 2:
            return z
        raise ValueError(f"Expected latent tensor with 2 or 3 dims, got shape {tuple(z.shape)}")

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z:
                [B, D] or [B, T, D]

        Returns:
            Next latent with the same rank as the input.
        """
        route_z = self._routing_state(z)
        g = torch.softmax(self.gate(route_z), dim=-1)

        z_macro = self._latent_only(self.macro(z))
        z_vol = self._latent_only(self.vol(z))
        z_liq = self._latent_only(self.liq(z))
        z_res = self._latent_only(self.res(z))

        if z_macro.shape != z.shape:
            raise ValueError(
                f"Macro operator output shape {tuple(z_macro.shape)} does not match input shape {tuple(z.shape)}"
            )
        if z_vol.shape != z.shape:
            raise ValueError(
                f"Volatility operator output shape {tuple(z_vol.shape)} does not match input shape {tuple(z.shape)}"
            )
        if z_liq.shape != z.shape:
            raise ValueError(
                f"Liquidity operator output shape {tuple(z_liq.shape)} does not match input shape {tuple(z.shape)}"
            )
        if z_res.shape != z.shape:
            raise ValueError(
                f"Residual operator output shape {tuple(z_res.shape)} does not match input shape {tuple(z.shape)}"
            )

        if z.dim() == 2:
            g0 = g[:, 0:1]
            g1 = g[:, 1:2]
            g2 = g[:, 2:3]
            g3 = g[:, 3:4]
            z_next = g0 * z_macro + g1 * z_vol + g2 * z_liq + g3 * z_res
            return z_next

        # z.dim() == 3
        g = g.unsqueeze(1)  # [B, 1, 4]
        z_next = (
            g[..., 0:1] * z_macro
            + g[..., 1:2] * z_vol
            + g[..., 2:3] * z_liq
            + g[..., 3:4] * z_res
        )
        return z_next