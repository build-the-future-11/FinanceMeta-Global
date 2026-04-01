from __future__ import annotations

import torch
import torch.nn as nn

from fi_jepa.predictors.macro_operator import MacroOperator
from fi_jepa.predictors.volatility_operator import VolatilityOperator
from fi_jepa.predictors.liquidity_operator import LiquidityOperator
from fi_jepa.predictors.residual_operator import ResidualOperator



class PredictorBank(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        operators: int = 4,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.operators = operators
    def forward(self, z):

        g = torch.softmax(self.gate(z), dim=-1)

        z_macro, _, _ = self.macro(z)
        z_vol, _, _ = self.vol(z)
        z_liq, _, _ = self.liq(z)
        z_res, _, _ = self.res(z)

        z_next = (
            g[..., 0:1] * z_macro
            + g[..., 1:2] * z_vol
            + g[..., 2:3] * z_liq
            + g[..., 3:4] * z_res
        )

        return z_next