from __future__ import annotations
from typing import Self

import torch
import torch.nn as nn


class EMA:
    """
    Exponential Moving Average (EMA) for target encoder updates.

    Used in FI-JEPA to maintain a stable target network
    similar to BYOL / JEPA style self-supervised training.

    θ_target ← τ θ_target + (1 - τ) θ_online
    """

    def __init__(
        self,
        online_model: nn.Module,
        target_model: nn.Module,
        tau: float = 0.996,
    ):

        self.online_model = online_model
        self.target_model = target_model
        self.tau = tau

        self._initialize()

    @torch.no_grad()
    def _initialize(self):
        """
        Copy parameters from online model to target model.
        """

        for target_param, online_param in zip(
            self.target_model.parameters(),
            self.online_model.parameters(),
        ):
            target_param.copy_(online_param)

    @torch.no_grad()
    def update(ema_self):
        """
        Perform EMA parameter update.
        """

        for target_param, online_param in zip(
            Self.target_model.parameters(),
            Self.online_model.parameters(),
        ):

            target_param.mul_(Self.tau)
            target_param.add_(
                online_param,
                alpha=(1.0 - Self.tau),
            )