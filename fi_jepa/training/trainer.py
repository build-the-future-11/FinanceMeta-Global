from __future__ import annotations
import torch
import torch.nn as nn


class EMA:
    """
    Exponential Moving Average for FI-JEPA target encoder.
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

        for t, o in zip(
            self.target_model.parameters(),
            self.online_model.parameters(),
        ):
            t.copy_(o)

    @torch.no_grad()
    def update(self):

        for t, o in zip(
            self.target_model.parameters(),
            self.online_model.parameters(),
        ):
            t.mul_(self.tau)
            t.add_(o, alpha=1 - self.tau)