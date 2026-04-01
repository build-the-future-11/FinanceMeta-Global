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
    Transformer encoder used for FI-JEPA context and target encoding.

    Designed for long financial sequences and stable self-supervised training.
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
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token

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
        # Optional CLS token
        # ---------------------------------------------------------

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        # ---------------------------------------------------------
        # Transformer
        # ---------------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(embed_dim)

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
            Shape (B, T, F)

        return_sequence : bool
            True  -> return (B, T, D)
            False -> return pooled latent (B, D)

        attention_mask : Optional[Tensor]
            Shape (B, T)

        causal : bool
            Apply causal masking for autoregressive tasks
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

        # ---------------------------------------------------------
        # Positional encoding
        # ---------------------------------------------------------

        h = self.pos_encoding(h)

        # ---------------------------------------------------------
        # Causal mask
        # ---------------------------------------------------------

        mask = None
        if causal:
            L = h.size(1)
            mask = torch.triu(
                torch.ones(L, L, device=h.device),
                diagonal=1,
            ).bool()

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
        # Output handling
        # ---------------------------------------------------------

        if return_sequence:
            return h

        # CLS pooling
        if self.cls_token is not None:
            return h[:, 0]

        # Mean pooling fallback
        return h.mean(dim=1)