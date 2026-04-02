from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# 🔥 SHARED FEATURE TRUNK (BETTER REPRESENTATION)
# ---------------------------------------------------------

class HeadBase(nn.Module):

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, z):
        return self.net(z)


# ---------------------------------------------------------
# 🔥 1. MIXTURE RETURN HEAD (MULTI-MODAL DISTRIBUTION)
# ---------------------------------------------------------

class ReturnHead(nn.Module):

    def __init__(self, latent_dim: int, hidden_dim: int = 128, mixtures: int = 3):
        super().__init__()

        self.mixtures = mixtures
        self.base = HeadBase(latent_dim, hidden_dim)

        self.mu = nn.Linear(hidden_dim, mixtures)
        self.logvar = nn.Linear(hidden_dim, mixtures)
        self.weights = nn.Linear(hidden_dim, mixtures)

    def forward(self, z):

        h = self.base(z)

        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), -8, 4)
        var = torch.exp(logvar)

        weights = F.softmax(self.weights(h), dim=-1)

        return {
            "mu": mu,
            "var": var,
            "logvar": logvar,
            "weights": weights,
        }


# ---------------------------------------------------------
# ⚡ 2. VOLATILITY HEAD (CALIBRATED + LOG-STABLE)
# ---------------------------------------------------------

class VolatilityHead(nn.Module):

    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.base = HeadBase(latent_dim, hidden_dim)

        self.logvar = nn.Linear(hidden_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, z):

        h = self.base(z)

        logvar = torch.clamp(self.logvar(h) + self.bias, -8, 4)
        vol = torch.exp(0.5 * logvar)

        return {
            "vol": vol,
            "logvar": logvar,
        }


# ---------------------------------------------------------
# 🧠 3. REGIME HEAD (TEMPERATURE + CONFIDENCE)
# ---------------------------------------------------------

class RegimeHead(nn.Module):

    def __init__(self, latent_dim: int, num_regimes: int = 4):
        super().__init__()

        self.base = HeadBase(latent_dim, latent_dim)

        self.logits = nn.Linear(latent_dim, num_regimes)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, z):

        h = self.base(z)

        logits = self.logits(h) / self.temperature.clamp_min(0.1)
        probs = F.softmax(logits, dim=-1)

        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(-1, keepdim=True)
        confidence = 1.0 - entropy / torch.log(torch.tensor(probs.size(-1), device=z.device))

        return {
            "logits": logits,
            "probs": probs,
            "entropy": entropy,
            "confidence": confidence,
        }


# ---------------------------------------------------------
# ⚠️ 4. RISK HEAD (MONOTONIC QUANTILES — REAL VaR)
# ---------------------------------------------------------

class RiskHead(nn.Module):

    def __init__(self, latent_dim: int, hidden_dim: int = 128, num_quantiles: int = 3):
        super().__init__()

        self.base = HeadBase(latent_dim, hidden_dim)

        self.base_q = nn.Linear(hidden_dim, 1)
        self.delta_q = nn.Linear(hidden_dim, num_quantiles - 1)

    def forward(self, z):

        h = self.base(z)

        q0 = self.base_q(h)
        deltas = F.softplus(self.delta_q(h))

        qs = [q0]
        for i in range(deltas.shape[-1]):
            qs.append(qs[-1] + deltas[..., i:i+1])

        quantiles = torch.cat(qs, dim=-1)

        return {
            "quantiles": quantiles
        }