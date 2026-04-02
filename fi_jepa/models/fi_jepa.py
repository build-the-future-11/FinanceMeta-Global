from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from fi_jepa.encoders.transformer_encoder import TransformerEncoder
from fi_jepa.models.predictor_bank import PredictorBank
from fi_jepa.models.heads import (
    ReturnHead,
    VolatilityHead,
    RegimeHead,
    RiskHead,
)


@dataclass
class FIJEPAOutput:
    latent_context: torch.Tensor
    latent_predictions: torch.Tensor
    latent_target: Optional[torch.Tensor] = None
    operator_weights: Optional[torch.Tensor] = None
    context_gate: Optional[torch.Tensor] = None


class FIJEPA(nn.Module):

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 256,
        pred_horizon: int = 16,
        encoder_depth: int = 6,
        encoder_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.pred_horizon = pred_horizon

        self.context_encoder = TransformerEncoder(
            input_dim=input_dim,
            embed_dim=latent_dim,
            depth=encoder_depth,
            heads=encoder_heads,
            dropout=dropout,
        )

        self.target_encoder = TransformerEncoder(
            input_dim=input_dim,
            embed_dim=latent_dim,
            depth=encoder_depth,
            heads=encoder_heads,
            dropout=dropout,
        )

        self._init_target_encoder()

        self.sequence_norm = nn.LayerNorm(latent_dim)
        self.pool_gate = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),
        )

        self.latent_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 2 * latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * latent_dim, latent_dim),
        )

        self.rollout_refiner = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 2 * latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * latent_dim, latent_dim),
        )

        self.rollout_mix_logit = nn.Parameter(torch.tensor(0.0))
        self.output_norm = nn.LayerNorm(latent_dim)
        self.output_dropout = nn.Dropout(dropout)

        self.predictor = PredictorBank(latent_dim)

        self.return_head = ReturnHead(latent_dim)
        self.volatility_head = VolatilityHead(latent_dim)
        self.regime_head = RegimeHead(latent_dim)
        self.risk_head = RiskHead(latent_dim)

    def _init_target_encoder(self):

        self.target_encoder.load_state_dict(self.context_encoder.state_dict())

        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def _pool_sequence(self, z: torch.Tensor):
        last = z[:, -1]
        mean = z.mean(dim=1)
        gate = torch.sigmoid(self.pool_gate(torch.cat([last, mean], dim=-1)))
        pooled = gate * last + (1.0 - gate) * mean
        return pooled, gate

    def encode_context(self, x: torch.Tensor):

        z_seq = self.context_encoder(x)
        z_seq = self.sequence_norm(z_seq)

        pooled, gate = self._pool_sequence(z_seq)
        pooled = self.latent_proj(pooled)
        pooled = self.output_norm(pooled)
        pooled = self.output_dropout(pooled)

        return pooled, gate

    def encode_target(self, x: torch.Tensor):

        with torch.no_grad():
            z = self.target_encoder(x)
            z = self.sequence_norm(z)
            z = self.latent_proj(z)

        return z

    def rollout(self, z0: torch.Tensor):

        current = z0
        preds = []
        weights = []

        mix = torch.sigmoid(self.rollout_mix_logit)

        for _ in range(self.pred_horizon):

            op_weights = self.predictor.compute_weights(current)
            next_latent = self.predictor.step(current)
            next_latent = self.rollout_refiner(next_latent)
            next_latent = self.output_norm(next_latent)
            next_latent = self.output_dropout(next_latent)

            current = mix * next_latent + (1.0 - mix) * current
            current = self.output_norm(current)

            preds.append(current)
            weights.append(op_weights)

        preds = torch.stack(preds, dim=1)
        weights = torch.stack(weights, dim=1)

        return preds, weights

    def forward(
        self,
        context_window: torch.Tensor,
        target_window: Optional[torch.Tensor] = None,
    ) -> FIJEPAOutput:

        z_context, gate = self.encode_context(context_window)
        latent_preds, op_weights = self.rollout(z_context)

        z_target = None
        if target_window is not None:
            z_t = self.encode_target(target_window)
            z_target = z_t[:, : self.pred_horizon, :]

        return FIJEPAOutput(
            latent_context=z_context,
            latent_predictions=latent_preds,
            latent_target=z_target,
            operator_weights=op_weights,
            context_gate=gate,
        )

    def extract_features(self, x: torch.Tensor):
        z_context, _ = self.encode_context(x)
        return z_context

    def predict_returns(self, x: torch.Tensor):
        return self.return_head(self.extract_features(x))

    def predict_volatility(self, x: torch.Tensor):
        return self.volatility_head(self.extract_features(x))

    def predict_regime(self, x: torch.Tensor):
        return self.regime_head(self.extract_features(x))

    def predict_risk(self, x: torch.Tensor):
        return self.risk_head(self.extract_features(x))

    @torch.no_grad()
    def update_target_encoder(self, tau: float = 0.996):

        for t, s in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
        ):
            t.data.mul_(tau).add_(s.data, alpha=(1 - tau))

    def freeze_encoder(self):

        for p in self.context_encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):

        for p in self.context_encoder.parameters():
            p.requires_grad = True