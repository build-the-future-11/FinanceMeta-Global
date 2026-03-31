from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from fi_jepa.encoders.transformer_encoder import TransformerEncoder
from fi_jepa.models.predictor_bank import PredictorBank
from fi_jepa.models.heads import (
    ReturnHead,
    VolatilityHead,
    RegimeHead,
    RiskHead,
)


class FIJEPA(nn.Module):
    """
    Financial Joint Embedding Predictive Architecture (FI-JEPA)

    The model learns latent market dynamics by predicting future
    latent embeddings rather than reconstructing market observations.

    Architecture
    -------------
    Context Encoder:
        Encodes historical market context into latent representation.

    Target Encoder (EMA copy):
        Produces stable future embeddings for JEPA targets.

    Predictor Bank:
        Operator-aligned latent dynamics modules including
        macro regime, volatility clustering, liquidity shocks,
        and residual microstructure effects.

    Rollout:
        Multi-step stochastic latent prediction.

    Downstream Heads:
        Optional heads for supervised adaptation tasks.
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
    ):

        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_assets = num_assets
        self.context_length = context_length
        self.pred_horizon = pred_horizon

        # --------------------------------------------------
        # Context encoder
        # --------------------------------------------------

        self.context_encoder = TransformerEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            depth=encoder_depth,
            heads=encoder_heads,
        )

        # --------------------------------------------------
        # Target encoder (EMA updated externally)
        # --------------------------------------------------

        self.target_encoder = TransformerEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            depth=encoder_depth,
            heads=encoder_heads,
        )

        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # --------------------------------------------------
        # Predictor bank
        # --------------------------------------------------

        self.predictor = PredictorBank(
            latent_dim=latent_dim,
            num_assets=num_assets,
        )

        # --------------------------------------------------
        # Downstream heads
        # --------------------------------------------------

        self.return_head = ReturnHead(latent_dim)
        self.volatility_head = VolatilityHead(latent_dim)
        self.regime_head = RegimeHead(latent_dim)
        self.risk_head = RiskHead(latent_dim)

    # --------------------------------------------------
    # Encoding
    # --------------------------------------------------

    def encode_context(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode historical context.

        Args:
            x : (B, T, input_dim)

        Returns:
            latent sequence (B, T, latent_dim)
        """

        return self.context_encoder(x)

    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode target future window.

        Target encoder is updated via EMA.
        """

        with torch.no_grad():
            z = self.target_encoder(x)

        return z

    # --------------------------------------------------
    # Latent rollout
    # --------------------------------------------------

    def rollout(self, z_context: torch.Tensor) -> torch.Tensor:
        """
        Multi-step latent prediction.

        Args:
            z_context : (B, T, D)

        Returns:
            predicted latent trajectory
            (B, H, T, D)
        """

        z = z_context

        preds = []

        for _ in range(self.pred_horizon):

            z = self.predictor(z)

            preds.append(z)

        return torch.stack(preds, dim=1)

    # --------------------------------------------------
    # JEPA latent prediction
    # --------------------------------------------------

    def forward(
        self,
        context_window: torch.Tensor,
        target_window: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass used for self-supervised training.

        Args
        ----
        context_window : (B, Tc, F)
        target_window  : (B, Tt, F)

        Returns
        -------
        Dictionary containing:
            latent_context
            latent_target
            latent_predictions
        """

        z_context = self.encode_context(context_window)

        latent_preds = self.rollout(z_context)

        outputs = {
            "latent_context": z_context,
            "latent_predictions": latent_preds,
        }

        if target_window is not None:

            z_target = self.encode_target(target_window)

            outputs["latent_target"] = z_target

        return outputs

    # --------------------------------------------------
    # Downstream tasks
    # --------------------------------------------------

    def predict_returns(self, x: torch.Tensor) -> torch.Tensor:

        z = self.encode_context(x)

        z_last = z[:, -1]

        return self.return_head(z_last)

    def predict_volatility(self, x: torch.Tensor) -> torch.Tensor:

        z = self.encode_context(x)

        z_last = z[:, -1]

        return self.volatility_head(z_last)

    def predict_regime(self, x: torch.Tensor) -> torch.Tensor:

        z = self.encode_context(x)

        z_last = z[:, -1]

        return self.regime_head(z_last)

    def predict_risk(self, x: torch.Tensor) -> torch.Tensor:

        z = self.encode_context(x)

        z_last = z[:, -1]

        return self.risk_head(z_last)

    # --------------------------------------------------
    # EMA target update
    # --------------------------------------------------

    @torch.no_grad()
    def update_target_encoder(self, tau: float = 0.996):
        """
        EMA update for target encoder.

        θ_target ← τ θ_target + (1−τ) θ_online
        """

        for target_param, online_param in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
        ):
            target_param.data.mul_(tau).add_(
                online_param.data,
                alpha=1 - tau,
            )