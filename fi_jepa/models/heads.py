from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# 🔥 1. DISTRIBUTIONAL RETURN HEAD
# ---------------------------------------------------------

class ReturnHead(nn.Module):
    """
    Predict return distribution (mean + std).
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )

        self.mu = nn.Linear(hidden_dim, 1)
        self.sigma = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, z):
        h = self.net(z)
        mu = self.mu(h)
        sigma = self.sigma(h) + 1e-6
        return mu, sigma


# ---------------------------------------------------------
# ⚡ 2. VOLATILITY HEAD (LOG-STABLE)
# ---------------------------------------------------------

class VolatilityHead(nn.Module):
    """
    Predict log-volatility for stability.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        log_vol = self.net(z)
        vol = torch.exp(log_vol)  # more stable than softplus
        return vol


# ---------------------------------------------------------
# 🧠 3. REGIME HEAD (WITH CONFIDENCE)
# ---------------------------------------------------------

class RegimeHead(nn.Module):
    """
    Market regime classification with confidence.
    """

    def __init__(self, latent_dim: int, num_regimes: int = 4):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, num_regimes),
        )

    def forward(self, z):
        logits = self.net(z)
        probs = F.softmax(logits, dim=-1)

        # entropy = uncertainty
        entropy = -(probs * torch.log(probs + 1e-8)).sum(-1, keepdim=True)

        return logits, probs, entropy


# ---------------------------------------------------------
# ⚠️ 4. RISK HEAD (QUANTILES — REAL VaR)
# ---------------------------------------------------------

class RiskHead(nn.Module):
    """
    Predict quantiles (VaR-like).
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )

        # 3 quantiles: 5%, 50%, 95%
        self.quantiles = nn.Linear(hidden_dim, 3)

    def forward(self, z):
        q = self.net(z)
        q = self.quantiles(q)

        # enforce ordering (very important)
        q_sorted, _ = torch.sort(q, dim=-1)

        return q_sorted  # [q05, median, q95]