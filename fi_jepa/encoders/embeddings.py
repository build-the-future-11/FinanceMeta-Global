from __future__ import annotations

import math
import torch
from torch import nn, Tensor


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

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
        """
        x: [B, T, D]
        """
        t = x.size(1)
        return x + self.pe[:t].unsqueeze(0).to(dtype=x.dtype, device=x.device)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)
        self.max_len = max_len

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        idx = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        return x + self.pos(idx)


class FeatureEmbedding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = self.norm(x)
        return self.dropout(x)