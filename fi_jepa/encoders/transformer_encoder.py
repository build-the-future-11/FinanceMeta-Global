from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import nn, Tensor

from fi_jepa.encoders.embeddings import (
    FeatureEmbedding,
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    HybridPositionalEncoding,
)


class DropPath(nn.Module):

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        rand = torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = (rand < keep_prob).float()
        return x * mask / keep_prob


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        positional_encoding: Literal["sinusoidal", "learned", "hybrid"] = "hybrid",
        max_len: int = 4096,
        use_cls_token: bool = False,
        use_projection_head: bool = False,
        drop_path: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        self.use_projection_head = use_projection_head

        self.input_embed = FeatureEmbedding(input_dim, embed_dim, dropout=dropout)

        if positional_encoding == "sinusoidal":
            self.pos_encoding = SinusoidalPositionalEncoding(embed_dim, max_len=max_len, dropout=dropout)
        elif positional_encoding == "learned":
            self.pos_encoding = LearnedPositionalEncoding(embed_dim, max_len=max_len, dropout=dropout)
        else:
            self.pos_encoding = HybridPositionalEncoding(embed_dim, max_len=max_len, dropout=dropout)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        else:
            self.cls_token = None

        dpr = torch.linspace(0, drop_path, depth).tolist()

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(
                    embed_dim,
                    heads,
                    dropout=dropout,
                    batch_first=True,
                ),
                "mlp": nn.Sequential(
                    nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                ),
                "norm1": nn.LayerNorm(embed_dim),
                "norm2": nn.LayerNorm(embed_dim),
                "drop_path": DropPath(dpr[i]),
            })
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        if use_projection_head:
            self.projection = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.projection = None

    def _build_causal_mask(self, L: int, device):
        return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        x: Tensor,
        return_sequence: bool = True,
        attention_mask: Optional[Tensor] = None,
        causal: bool = False,
    ) -> Tensor:

        B, T, _ = x.shape

        h = self.input_embed(x)

        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)
            h = torch.cat([cls, h], dim=1)

            if attention_mask is not None:
                cls_mask = torch.zeros(B, 1, device=h.device, dtype=torch.bool)
                attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        h = self.pos_encoding(h)

        attn_mask = None
        if causal:
            attn_mask = self._build_causal_mask(h.size(1), h.device)

        for layer in self.layers:

            h_norm = layer["norm1"](h)

            attn_out, _ = layer["attn"](
                h_norm,
                h_norm,
                h_norm,
                attn_mask=attn_mask,
                key_padding_mask=attention_mask,
                need_weights=False,
            )

            h = h + layer["drop_path"](attn_out)

            h_norm = layer["norm2"](h)
            mlp_out = layer["mlp"](h_norm)

            h = h + layer["drop_path"](mlp_out)

        h = self.norm(h)

        if self.projection is not None:
            h = self.projection(h)

        if return_sequence:
            return h

        if self.cls_token is not None:
            return h[:, 0]

        return h.mean(dim=1)