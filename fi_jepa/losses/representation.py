import torch
from torch import nn
import torch.nn.functional as F


class MacroRegimeRegularizer(nn.Module):
    """
    Encourages smooth and structured macro regime evolution.

    Improvements:
    - Smoothness (temporal consistency)
    - Optional velocity + acceleration penalties
    - Scale-invariant normalization
    """

    def __init__(
        self,
        strength: float = 0.05,
        velocity_weight: float = 1.0,
        acceleration_weight: float = 0.5,
        normalize: bool = True,
    ):
        super().__init__()

        self.strength = strength
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight
        self.normalize = normalize

    def forward(self, latent_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_sequence: [B, T, D]
        """

        if latent_sequence.size(1) < 3:
            return torch.tensor(0.0, device=latent_sequence.device)

        z = latent_sequence

        # ---------------------------------------------------------
        # Optional normalization (important for scale invariance)
        # ---------------------------------------------------------

        if self.normalize:
            z = F.normalize(z, dim=-1)

        # ---------------------------------------------------------
        # 1. Velocity (first derivative)
        # ---------------------------------------------------------

        velocity = z[:, 1:] - z[:, :-1]             # [B, T-1, D]
        vel_norm = torch.norm(velocity, dim=-1)    # [B, T-1]

        velocity_loss = vel_norm.mean()

        # ---------------------------------------------------------
        # 2. Acceleration (second derivative)
        # ---------------------------------------------------------

        acceleration = velocity[:, 1:] - velocity[:, :-1]   # [B, T-2, D]
        acc_norm = torch.norm(acceleration, dim=-1)         # [B, T-2]

        acceleration_loss = acc_norm.mean()

        # ---------------------------------------------------------
        # Total loss
        # ---------------------------------------------------------

        loss = (
            self.velocity_weight * velocity_loss
            + self.acceleration_weight * acceleration_loss
        )

        return self.strength * loss