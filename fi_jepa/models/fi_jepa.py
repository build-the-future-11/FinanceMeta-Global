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


# -------------------------------------------------------------
# Structured output
# -------------------------------------------------------------


@dataclass
class FIJEPAOutput:
    latent_context: torch.Tensor
    latent_predictions: torch.Tensor
    latent_target: Optional[torch.Tensor] = None
    operator_weights: Optional[torch.Tensor] = None


# -------------------------------------------------------------
# FI-JEPA Model
# -------------------------------------------------------------


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

        # -------------------------------------------------
        # Encoders
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Latent projection
        # -------------------------------------------------

        self.latent_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )

        # -------------------------------------------------
        # Predictor
        # -------------------------------------------------

        self.predictor = PredictorBank(latent_dim)

        # -------------------------------------------------
        # Heads
        # -------------------------------------------------

        self.return_head = ReturnHead(latent_dim)
        self.volatility_head = VolatilityHead(latent_dim)
        self.regime_head = RegimeHead(latent_dim)
        self.risk_head = RiskHead(latent_dim)

        # -------------------------------------------------
        # Normalization
        # -------------------------------------------------

        self.norm = nn.LayerNorm(latent_dim)

    # -------------------------------------------------------------
    # Target encoder init
    # -------------------------------------------------------------

    def _init_target_encoder(self):

        self.target_encoder.load_state_dict(
            self.context_encoder.state_dict()
        )

        for p in self.target_encoder.parameters():
            p.requires_grad = False

    # -------------------------------------------------------------
    # Encoding
    # -------------------------------------------------------------

    def encode_context(self, x: torch.Tensor) -> torch.Tensor:

        z = self.context_encoder(x)  # (B,T,D)
        z = self.norm(z)

        z = z[:, -1]                 # last token
        z = self.latent_proj(z)

        return z                     # (B,D)

    def encode_target(self, x: torch.Tensor):

        with torch.no_grad():

            z = self.target_encoder(x)
            z = self.norm(z)

        return z

    # -------------------------------------------------------------
    # Latent rollout
    # -------------------------------------------------------------

    def rollout(self, z0: torch.Tensor):

        z = z0

        preds = []
        weights = []

        for _ in range(self.pred_horizon):

            z, w, _ = self.predictor(z)

            z = self.norm(z)

            preds.append(z)
            weights.append(w)

        preds = torch.stack(preds, dim=1)
        weights = torch.stack(weights, dim=1)

        return preds, weights

    # -------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------

    def forward(
        self,
        context_window: torch.Tensor,
        target_window: Optional[torch.Tensor] = None,
    ) -> FIJEPAOutput:

        z_context = self.encode_context(context_window)

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
        )

    # -------------------------------------------------------------
    # Feature extraction
    # -------------------------------------------------------------

    def extract_features(self, x: torch.Tensor):

        return self.encode_context(x)

    # -------------------------------------------------------------
    # Heads
    # -------------------------------------------------------------

    def predict_returns(self, x: torch.Tensor):

        return self.return_head(self.extract_features(x))

    def predict_volatility(self, x: torch.Tensor):

        return self.volatility_head(self.extract_features(x))

    def predict_regime(self, x: torch.Tensor):

        return self.regime_head(self.extract_features(x))

    def predict_risk(self, x: torch.Tensor):

        return self.risk_head(self.extract_features(x))

    # -------------------------------------------------------------
    # EMA update
    # -------------------------------------------------------------

    @torch.no_grad()
    def update_target_encoder(self, tau: float = 0.996):

        for t, s in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
        ):

            t.data.mul_(tau).add_(s.data, alpha=(1 - tau))

    # -------------------------------------------------------------
    # Freeze utils
    # -------------------------------------------------------------

    def freeze_encoder(self):

        for p in self.context_encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):

        for p in self.context_encoder.parameters():
            p.requires_grad = True