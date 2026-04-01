from __future__ import annotations

from typing import Literal

import torch
from torch import nn, Tensor

from fi_jepa.encoders.embeddings import (
    FeatureEmbedding,
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
)


class TransformerEncoder(nn.Module):
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
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.input_embed = FeatureEmbedding(input_dim, embed_dim, dropout=dropout)

        if positional_encoding == "sinusoidal":
            self.pos_encoding = SinusoidalPositionalEncoding(embed_dim, max_len=max_len)
        else:
            self.pos_encoding = LearnedPositionalEncoding(embed_dim, max_len=max_len)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, return_sequence: bool = True) -> Tensor:
        """
        Args:
            x: [B, T, F]
            return_sequence:
                True  -> return latent sequence [B, T, D]
                False -> return pooled latent [B, D]

        Returns:
            Tensor of shape [B, T, D] or [B, D]
        """
        h = self.input_embed(x)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        h = self.norm(h)

        if return_sequence:
            return h

        # Mean pooling over time for a single latent summary.
        return h.mean(dim=1)