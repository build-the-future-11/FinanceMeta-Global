from __future__ import annotations

import torch
from torch import nn, Tensor


class ResidualSSMEncoder(nn.Module):
    """
    A practical structured state-space style encoder implemented as:
    input projection -> gated GRU dynamics -> residual mixing -> pooled latent.
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        depth: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.layers = nn.ModuleList(
            [
                nn.GRU(
                    input_size=embed_dim,
                    hidden_size=embed_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
                for _ in range(depth)
            ]
        )
        self.proj = nn.Linear(embed_dim * (2 if bidirectional else 1), embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, return_sequence: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """
        x: [B, T, F]
        """
        h = self.input_proj(x)
        for gru in self.layers:
            y, _ = gru(h)
            h = h + self.dropout(self.proj(y))
        h = self.norm(h)
        pooled = h.mean(dim=1)
        if return_sequence:
            return h, pooled
        return pooled