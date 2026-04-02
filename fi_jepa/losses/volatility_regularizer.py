import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# 🌍 1. MACRO REGIME REGULARIZER (UPGRADED)
# ---------------------------------------------------------

class MacroRegimeRegularizer(nn.Module):
    """
    Encourages smooth macro regime evolution.
    Includes velocity + acceleration penalties.
    """

    def __init__(
        self,
        strength: float = 0.05,
        velocity_weight: float = 1.0,
        acceleration_weight: float = 0.5,
    ):
        super().__init__()

        self.strength = strength
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, T, D]

        if z.size(1) < 3:
            return torch.tensor(0.0, device=z.device)

        # velocity
        v = z[:, 1:] - z[:, :-1]
        v_loss = torch.norm(v, dim=-1).mean()

        # acceleration
        a = v[:, 1:] - v[:, :-1]
        a_loss = torch.norm(a, dim=-1).mean()

        return self.strength * (
            self.velocity_weight * v_loss +
            self.acceleration_weight * a_loss
        )


# ---------------------------------------------------------
# ⚖️ 2. NO-ARBITRAGE REGULARIZER (REAL VERSION)
# ---------------------------------------------------------

class NoArbitrageRegularizer(nn.Module):
    """
    Enforces consistency between forward and backward dynamics.

    Idea:
        z_t → z_forward → z_reconstructed ≈ z_t

    Prevents:
        - directional inconsistency
        - exploitable latent cycles
    """

    def __init__(self, strength: float = 0.1):
        super().__init__()
        self.strength = strength

    def forward(
        self,
        z_t: torch.Tensor,
        z_forward: torch.Tensor,
        z_backward: torch.Tensor,
    ) -> torch.Tensor:

        # ---------------------------------------------------------
        # 1. Cycle consistency
        # ---------------------------------------------------------

        cycle_loss = F.mse_loss(z_backward, z_t)

        # ---------------------------------------------------------
        # 2. Forward-backward symmetry
        # ---------------------------------------------------------

        forward_step = z_forward - z_t
        backward_step = z_backward - z_forward

        symmetry_loss = F.mse_loss(forward_step, -backward_step)

        # ---------------------------------------------------------
        # Total
        # ---------------------------------------------------------

        loss = cycle_loss + 0.5 * symmetry_loss

        return self.strength * loss