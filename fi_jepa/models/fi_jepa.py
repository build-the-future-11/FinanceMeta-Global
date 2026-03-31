from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict

from fi_jepa.encoders.transformer_encoder import TransformerEncoder
from fi_jepa.models.predictor_bank import PredictorBank
from fi_jepa.models.heads import (
    ReturnHead,
    VolatilityHead,
    RegimeHead,
    RiskHead,
)


# -------------------------------------------------------------
# Structured output (cleaner than dicts)
# -------------------------------------------------------------

@dataclass
class FIJEPAOutput:

    latent_context: torch.Tensor
    latent_predictions: torch.Tensor
    latent_target: Optional[torch.Tensor] = None


# -------------------------------------------------------------
# FI-JEPA Model
# -------------------------------------------------------------


class FIJEPA(nn.Module):
    """
    Financial Joint Embedding Predictive Architecture

    Learns latent market dynamics by predicting future embeddings
    instead of reconstructing raw observations.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 256,
        num_assets: int = 1,
        context_length: int = 128,
        pred_horizon: int = 16,
        encoder_depth: int = 6,
        encoder_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_assets = num_assets
        self.context_length = context_length
        self.pred_horizon = pred_horizon

        # -------------------------------------------------
        # Context Encoder
        # -------------------------------------------------

        self.context_encoder = TransformerEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            depth=encoder_depth,
            heads=encoder_heads,
            dropout=dropout,
        )

        # -------------------------------------------------
        # Target Encoder (EMA)
        # -------------------------------------------------

        self.target_encoder = TransformerEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            depth=encoder_depth,
            heads=encoder_heads,
            dropout=dropout,
        )

        self._init_target_encoder()

        # -------------------------------------------------
        # Predictor Bank
        # -------------------------------------------------

        self.predictor = PredictorBank(
            latent_dim=latent_dim,
            num_assets=num_assets,
        )

        # -------------------------------------------------
        # Heads
        # -------------------------------------------------

        self.return_head = ReturnHead(latent_dim)
        self.volatility_head = VolatilityHead(latent_dim)
        self.regime_head = RegimeHead(latent_dim)
        self.risk_head = RiskHead(latent_dim)

        # -------------------------------------------------
        # Stabilization Layers
        # -------------------------------------------------

        self.latent_norm = nn.LayerNorm(latent_dim)

    # -------------------------------------------------------------
    # Target encoder initialization
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
        """
        Encode historical market context.

        Args
        ----
        x : (B, T, F)

        Returns
        -------
        (B, T, D)
        """

        z = self.context_encoder(x)

        return self.latent_norm(z)

    def encode_target(self, x: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():

            z = self.target_encoder(x)

        return self.latent_norm(z)

    # -------------------------------------------------------------
    # Latent rollout
    # -------------------------------------------------------------

    def rollout(self, z_context: torch.Tensor) -> torch.Tensor:
        """
        Multi-step latent prediction.

        Returns
        -------
        (B, H, T, D)
        """

        z = z_context

        preds = []

        for _ in range(self.pred_horizon):

            z = self.predictor(z)

            z = self.latent_norm(z)

            preds.append(z)

        return torch.stack(preds, dim=1)

    # -------------------------------------------------------------
    # Forward (JEPA training)
    # -------------------------------------------------------------

    def forward(
        self,
        context_window: torch.Tensor,
        target_window: Optional[torch.Tensor] = None,
    ) -> FIJEPAOutput:

        z_context = self.encode_context(context_window)

        latent_preds = self.rollout(z_context)

        z_target = None

        if target_window is not None:
            z_target = self.encode_target(target_window)

        return FIJEPAOutput(
            latent_context=z_context,
            latent_predictions=latent_preds,
            latent_target=z_target,
        )

    # -------------------------------------------------------------
    # Latent feature extraction
    # -------------------------------------------------------------

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns final timestep latent.

        Useful for downstream models.
        """

        z = self.encode_context(x)

        return z[:, -1]

    # -------------------------------------------------------------
    # Downstream predictions
    # -------------------------------------------------------------

    def predict_returns(self, x: torch.Tensor):

        z = self.extract_features(x)

        return self.return_head(z)

    def predict_volatility(self, x: torch.Tensor):

        z = self.extract_features(x)

        return self.volatility_head(z)

    def predict_regime(self, x: torch.Tensor):

        z = self.extract_features(x)

        return self.regime_head(z)

    def predict_risk(self, x: torch.Tensor):

        z = self.extract_features(x)

        return self.risk_head(z)

    # -------------------------------------------------------------
    # EMA Update
    # -------------------------------------------------------------

    @torch.no_grad()
    def update_target_encoder(self, tau: float = 0.996):
        """
        EMA update

        θ_target ← τ θ_target + (1 − τ) θ_online
        """

        for target_param, online_param in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
        ):

            target_param.data.mul_(tau).add_(
                online_param.data,
                alpha=(1 - tau),
            )

    # -------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------

    def freeze_encoder(self):

        for p in self.context_encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):

        for p in self.context_encoder.parameters():
            p.requires_grad = True