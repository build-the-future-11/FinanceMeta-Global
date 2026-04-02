from __future__ import annotations

import copy
import torch
import torch.nn as nn


class TargetEncoder(nn.Module):

    def __init__(self, context_encoder: nn.Module):
        super().__init__()
        self.encoder = copy.deepcopy(context_encoder)
        self._freeze()
        self.encoder.eval()

    def _freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def _sync_buffers(self, online_encoder: nn.Module):
        online_buffers = dict(online_encoder.named_buffers())
        target_buffers = dict(self.encoder.named_buffers())

        for name, buffer in online_buffers.items():
            if name in target_buffers:
                target_buffers[name].data.copy_(buffer.data)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        with torch.inference_mode():
            return self.encoder(x)

    @torch.no_grad()
    def hard_update(self, online_encoder: nn.Module):
        self.encoder.load_state_dict(online_encoder.state_dict(), strict=True)
        self._freeze()
        self.encoder.eval()

    @torch.no_grad()
    def ema_update(
        self,
        online_encoder: nn.Module,
        tau: float = 0.996,
        update_buffers: bool = True,
    ):
        tau = float(tau)
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must be in [0, 1].")

        online_params = dict(online_encoder.named_parameters())
        target_params = dict(self.encoder.named_parameters())

        for name, target_param in target_params.items():
            if name not in online_params:
                continue
            online_param = online_params[name]
            target_param.data.mul_(tau).add_(online_param.data, alpha=1.0 - tau)

        if update_buffers:
            self._sync_buffers(online_encoder)

        self.encoder.eval()

    def to(self, *args, **kwargs):
        self.encoder = self.encoder.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        super().train(False)
        self.encoder.eval()
        self._freeze()
        return self

    def eval(self):
        super().eval()
        self.encoder.eval()
        self._freeze()
        return self