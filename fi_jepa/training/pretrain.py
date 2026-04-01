from __future__ import annotations

import torch
import torch.nn as nn


class EMA:
    """
    Exponential Moving Average for FI-JEPA target encoders.

    Keeps the target encoder as a moving average of the online/context encoder.
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
        Copy online weights into target weights before training starts.
        """
        self.target_model.load_state_dict(self.online_model.state_dict(), strict=True)

        for param in self.target_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self):
        """
        EMA update:
            target = tau * target + (1 - tau) * online
        """
        for target_param, online_param in zip(
            self.target_model.parameters(),
            self.online_model.parameters(),
        ):
            target_param.mul_(self.tau)
            target_param.add_(online_param, alpha=1.0 - self.tau)