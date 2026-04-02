

import torch

from fi_jepa.losses.no_arbitrage import LiquidityRegularizer


class MacroRegularizer(nn.Module):

    def __init__(
        self,
        smoothness_weight: float = 0.2,
        low_freq_weight: float = 0.2,
        stability_weight: float = 0.1,
    ):
        super().__init__()

        self.smoothness_weight = smoothness_weight
        self.low_freq_weight = low_freq_weight
        self.stability_weight = stability_weight

    def forward(
        self,
        operator_weights: torch.Tensor,
        latent_sequence: torch.Tensor,
    ) -> torch.Tensor:

        w_macro = operator_weights[..., 0]

        if w_macro.dim() == 3:
            diff = w_macro[:, 1:] - w_macro[:, :-1]
            smoothness_loss = (diff ** 2).mean()
        else:
            smoothness_loss = torch.tensor(0.0, device=w_macro.device)

        if w_macro.dim() == 3:
            fft = torch.fft.rfft(w_macro, dim=1)
            power = torch.abs(fft)

            freqs = torch.arange(power.size(1), device=power.device).float()
            freq_weight = freqs / (freqs.max() + 1e-6)

            high_freq_power = (power * freq_weight[None, :, None]).mean()
            low_freq_loss = high_freq_power
        else:
            low_freq_loss = torch.tensor(0.0, device=w_macro.device)

        if latent_sequence is not None and latent_sequence.dim() == 3:
            latent_mean = latent_sequence.mean(dim=1)
            stability_loss = latent_mean.var(dim=0).mean()
        else:
            stability_loss = torch.tensor(0.0, device=w_macro.device)

        total = (
            self.smoothness_weight * smoothness_loss
            + self.low_freq_weight * low_freq_loss
            + self.stability_weight * stability_loss
        )

        return total


class OperatorRegularizer(nn.Module):

    def __init__(self):
        super().__init__()

        self.liquidity = LiquidityRegularizer()
        self.macro = MacroRegularizer()

    def forward(
        self,
        operator_weights: torch.Tensor,
        latent_sequence: torch.Tensor,
    ) -> torch.Tensor:

        liq_loss = self.liquidity(operator_weights, latent_sequence)
        macro_loss = self.macro(operator_weights, latent_sequence)

        return liq_loss + macro_loss