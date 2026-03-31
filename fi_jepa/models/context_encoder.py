import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):

    def __init__(self, input_dim, embed_dim, depth, heads, mlp_ratio, dropout):

        super().__init__()

        self.input_proj = nn.Linear(input_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):

        x = self.input_proj(x)

        h = self.encoder(x)

        h = self.norm(h)

        return h[:, -1]