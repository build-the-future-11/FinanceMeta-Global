from __future__ import annotations

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------
# Functional API (kept)
# ---------------------------------------------------------

def update_ema(
    online_model: nn.Module,
    target_model: nn.Module,
    tau: float,
):
    with torch.no_grad():
        for t, s in zip(target_model.parameters(), online_model.parameters()):
            t.data.mul_(tau).add_(s.data, alpha=1 - tau)


def copy_weights(
    online_model: nn.Module,
    target_model: nn.Module,
):
    target_model.load_state_dict(online_model.state_dict())
    for p in target_model.parameters():
        p.requires_grad = False


def momentum_schedule(
    step: int,
    max_steps: int,
    base_tau: float = 0.996,
    final_tau: float = 1.0,
):
    progress = step / max_steps
    tau = final_tau - (final_tau - base_tau) * (
        (math.cos(progress * math.pi) + 1) / 2
    )
    return tau


# ---------------------------------------------------------
# 🔥 CLASS API (THIS FIXES YOUR ERROR)
# ---------------------------------------------------------

class EMA:
    def __init__(
        self,
        online_model: nn.Module,
        target_model: nn.Module,
        tau: float = 0.996,
    ):
        self.online_model = online_model
        self.target_model = target_model
        self.tau = tau

        copy_weights(online_model, target_model)

    @torch.no_grad()
    def update(self):
        update_ema(
            self.online_model,
            self.target_model,
            self.tau,
        )

    def set_tau(self, tau: float):
        self.tau = tau