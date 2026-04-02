from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import nn, Tensor

from fi_jepa.encoders.embeddings import (
    FeatureEmbedding,
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
)


class TransformerEncoder(nn.Module):
    """
    FI-JEPA Optimized Transformer Encoder

    Improvements:
    - Better masking handling
    - CLS token stability
    - Dropout control for JEPA (important)
    - Optional latent projection head
    - Safer causal masking
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        positional_encoding: Literal["sinusoidal", "learned"] = "learned",
        max_len: int = 4096,
        use_cls_token: bool = False,
        use_projection_head: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        self.use_projection_head = use_projection_head

        # ---------------------------------------------------------
        # Feature embedding
        # ---------------------------------------------------------

        self.input_embed = FeatureEmbedding(
            input_dim,
            embed_dim,
            dropout=dropout,
        )

        # ---------------------------------------------------------
        # Positional encoding
        # ---------------------------------------------------------

        if positional_encoding == "sinusoidal":
            self.pos_encoding = SinusoidalPositionalEncoding(
                embed_dim,
                max_len=max_len,
            )
        else:
            self.pos_encoding = LearnedPositionalEncoding(
                embed_dim,
                max_len=max_len,
            )

        # ---------------------------------------------------------
        # CLS token (better initialization)
        # ---------------------------------------------------------

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        else:
            self.cls_token = None

        # ---------------------------------------------------------
        # Transformer layers
        # ---------------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,  # better stability
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(embed_dim)

        # ---------------------------------------------------------
        # Optional projection head (VERY useful for JEPA)
        # ---------------------------------------------------------

        if use_projection_head:
            self.projection = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.projection = None

    # ---------------------------------------------------------
    # Mask utilities
    # ---------------------------------------------------------

    def _build_causal_mask(self, L: int, device):
        return torch.triu(
            torch.ones(L, L, device=device, dtype=torch.bool),
            diagonal=1,
        )

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        return_sequence: bool = True,
        attention_mask: Optional[Tensor] = None,
        causal: bool = False,
    ) -> Tensor:
        """
        Args
        ----
        x : Tensor
            (B, T, F)

        return_sequence : bool
            True  -> (B, T, D)
            False -> (B, D)

        attention_mask : Optional[Tensor]
            (B, T) where True = pad

        causal : bool
            Apply causal masking
        """

        B, T, _ = x.shape

        # ---------------------------------------------------------
        # Embedding
        # ---------------------------------------------------------

        h = self.input_embed(x)

        # ---------------------------------------------------------
        # CLS token
        # ---------------------------------------------------------

        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)
            h = torch.cat([cls, h], dim=1)

            if attention_mask is not None:
                cls_mask = torch.zeros(B, 1, device=h.device, dtype=torch.bool)
                attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        # ---------------------------------------------------------
        # Positional encoding
        # ---------------------------------------------------------

        h = self.pos_encoding(h)

        # ---------------------------------------------------------
        # Causal mask
        # ---------------------------------------------------------

        mask = None
        if causal:
            mask = self._build_causal_mask(h.size(1), h.device)

        # ---------------------------------------------------------
        # Transformer
        # ---------------------------------------------------------

        h = self.encoder(
            h,
            mask=mask,
            src_key_padding_mask=attention_mask,
        )

        h = self.norm(h)

        # ---------------------------------------------------------
        # Projection head (JEPA trick)
        # ---------------------------------------------------------

        if self.projection is not None:
            h = self.projection(h)

        # ---------------------------------------------------------
        # Output handling
        # ---------------------------------------------------------

        if return_sequence:
            return h

        if self.cls_token is not None:
            return h[:, 0]

        return h.mean(dim=1)