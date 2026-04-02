import torch
import torch.nn.functional as F


def latent_prediction_loss(
    pred,
    target,
    sim_weight: float = 1.0,
    var_weight: float = 0.1,
    cov_weight: float = 0.05,
    eps: float = 1e-4,
):
    """
    JEPA-style latent loss with:
    - cosine similarity (alignment)
    - variance regularization (anti-collapse)
    - covariance regularization (decorrelation)

    Args:
        pred:   [B, T, D]
        target: [B, T, D]
    """

    # ---- Normalize ----
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target.detach(), dim=-1)

    # ---- 1. Similarity Loss (alignment) ----
    sim_loss = 2 - 2 * (pred * target).sum(dim=-1).mean()

    # ---- Flatten time + batch for stats ----
    B, T, D = pred.shape
    pred_flat = pred.reshape(B * T, D)

    # ---- 2. Variance Loss (prevent collapse) ----
    std = torch.sqrt(pred_flat.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(1 - std))

    # ---- 3. Covariance Loss (decorrelation) ----
    pred_centered = pred_flat - pred_flat.mean(dim=0)
    cov = (pred_centered.T @ pred_centered) / (B * T - 1)

    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / D

    # ---- Final Loss ----
    loss = (
        sim_weight * sim_loss
        + var_weight * var_loss
        + cov_weight * cov_loss
    )

    return loss