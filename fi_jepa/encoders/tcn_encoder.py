from __future__ import annotations

import torch
from torch import nn, Tensor


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T]
        pad = (self.kernel_size - 1) * self.dilation
        x = nn.functional.pad(x, (pad, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(channels, hidden_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(hidden_channels, channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(hidden_channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.residual = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.dropout(x)
        return self.act(x + residual)


class TCNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        depth: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.blocks = nn.ModuleList(
            [
                TemporalBlock(
                    channels=embed_dim,
                    hidden_channels=embed_dim * 2,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, return_sequence: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """
        x: [B, T, F]
        """
        h = self.input_proj(x).transpose(1, 2)  # [B, D, T]
        for block in self.blocks:
            h = block(h)
        h = h.transpose(1, 2)  # [B, T, D]
        h = self.norm(h)
        pooled = h.mean(dim=1)
        if return_sequence:
            return h, pooled
        return pooled