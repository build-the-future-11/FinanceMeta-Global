from __future__ import annotations

import math
import torch
from torch import nn, Tensor


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        t = x.size(1)
        pe = self.pe[:t].to(device=x.device, dtype=x.dtype).unsqueeze(0)
        return self.dropout(x + pe)


class LearnedPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.max_len = max_len
        self.pos = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(1))

        nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        if t > self.max_len:
            raise ValueError(f"Sequence length {t} exceeds max_len {self.max_len}")
        idx = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        pos = self.pos(idx)
        return self.dropout(x + self.scale * pos)


class HybridPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.sin = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=0.0)
        self.learned = LearnedPositionalEncoding(d_model, max_len=max_len, dropout=0.0)
        self.mix_logit = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        alpha = torch.sigmoid(self.mix_logit)
        y = alpha * self.sin(x) + (1.0 - alpha) * self.learned(x)
        return self.dropout(self.norm(y))


class FeatureEmbedding(nn.Module):

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        dropout: float = 0.1,
        use_feature_gate: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.use_feature_gate = use_feature_gate
        self.use_residual = use_residual

        self.proj = nn.Linear(input_dim, embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)

        self.in_norm = nn.LayerNorm(input_dim)
        self.emb_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.GELU()

        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))

        if use_feature_gate:
            self.gate = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid(),
            )
        else:
            self.gate = None

        self.residual_proj = nn.Linear(input_dim, embed_dim) if use_residual else None

        self.feature_summary = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_norm(x)

        h = self.proj(x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.proj2(h)

        if self.use_feature_gate:
            g = self.gate(x)
            h = h * g

        if self.use_residual:
            h = h + self.residual_proj(x)

        s = self.feature_summary(x)
        h = h + 0.25 * s

        h = self.emb_norm(h)
        h = h * self.scale + self.bias
        return self.dropout(h)