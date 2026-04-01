from __future__ import annotations

import math
import torch
import torch.nn as nn


def update_ema(
    online_model: nn.Module,
    target_model: nn.Module,
    tau: float = 0.996,
):
    """
    Exponential Moving Average update used in JEPA / BYOL style training.

    θ_target ← τ θ_target + (1 − τ) θ_online
    """

    with torch.no_grad():

        online_params = dict(online_model.named_parameters())
        target_params = dict(target_model.named_parameters())

        for name, param in online_params.items():

            if name not in target_params:
                continue

            target_param = target_params[name]

            target_param.data.mul_(tau).add_(
                param.data,
                alpha=1 - tau,
            )


def copy_weights(
    online_model: nn.Module,
    target_model: nn.Module,
):
    """
    Hard copy weights from online to target.
    """

    target_model.load_state_dict(online_model.state_dict())

    for p in target_model.parameters():
        p.requires_grad = False


def momentum_schedule(
    step: int,
    max_steps: int,
    base_tau: float = 0.996,
    final_tau: float = 1.0,
):
    """
    Cosine schedule for EMA momentum.
    """

    progress = step / max_steps

    tau = final_tau - (final_tau - base_tau) * (
        (math.cos(progress * math.pi) + 1) / 2
    )

    return tau