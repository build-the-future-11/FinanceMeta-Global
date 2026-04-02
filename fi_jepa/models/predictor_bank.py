from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from fi_jepa.predictors.macro_operator import MacroOperator
from fi_jepa.predictors.volatility_operator import VolatilityOperator
from fi_jepa.predictors.liquidity_operator import LiquidityOperator
from fi_jepa.predictors.residual_operator import ResidualOperator


class PredictorBank(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        operators: int = 4,
        horizon: int = 1,
        temperature: float = 1.0,
        drift_scale: float = 0.15,
        reaction_scale: float = 0.15,
        diffusion_scale: float = 0.10,
        variance_floor: float = 1e-4,
        variance_ceiling: float = 4.0,
    ):
        super().__init__()

        if operators != 4:
            raise ValueError("PredictorBank currently expects 4 operators.")

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim or (2 * latent_dim)
        self.horizon = horizon
        self.temperature = max(float(temperature), 1e-3)
        self.drift_scale = float(drift_scale)
        self.reaction_scale = float(reaction_scale)
        self.diffusion_scale = float(diffusion_scale)
        self.variance_floor = float(variance_floor)
        self.variance_ceiling = float(variance_ceiling)
        self.num_operators = operators

        self.gate = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, operators),
        )

        self.operators = nn.ModuleList([
            MacroOperator(latent_dim, self.hidden_dim, dropout),
            VolatilityOperator(latent_dim, self.hidden_dim, dropout),
            LiquidityOperator(latent_dim, self.hidden_dim, dropout),
            ResidualOperator(latent_dim, self.hidden_dim, dropout),
        ])

        self.operator_logvar_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, latent_dim),
            )
            for _ in range(operators)
        ])

        self.drift_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, latent_dim),
        )

        self.reaction_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, latent_dim),
        )

        self.uncertainty_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, latent_dim),
        )

        self.interaction_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, latent_dim),
        )

        self.diffusion_kernel = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)

        self.residual_mix_logit = nn.Parameter(torch.tensor(0.0))
        self.output_norm = nn.LayerNorm(latent_dim)
        self.output_dropout = nn.Dropout(dropout)

    @staticmethod
    def _latent_only(output):
        return output[0] if isinstance(output, tuple) else output

    @staticmethod
    def _routing_state(z: Tensor) -> Tensor:
        return z[:, -1] if z.dim() == 3 else z

    @staticmethod
    def _broadcast_like(x: Tensor, like: Tensor) -> Tensor:
        if like.dim() == 2:
            return x
        return x.unsqueeze(1).expand(-1, like.size(1), -1)

    def _apply_operator(self, op: nn.Module, z: Tensor) -> Tensor:
        if z.dim() == 2:
            return self._latent_only(op(z))

        b, t, d = z.shape
        flat = z.reshape(b * t, d)
        out = self._latent_only(op(flat))
        return out.reshape(b, t, d)

    def _apply_operators(self, z: Tensor) -> list[Tensor]:
        outputs = []
        for op in self.operators:
            out = self._apply_operator(op, z)
            if out.shape != z.shape:
                raise ValueError(
                    f"Operator output shape {tuple(out.shape)} "
                    f"does not match input shape {tuple(z.shape)}"
                )
            outputs.append(out)
        return outputs

    def _latent_diffusion(self, z: Tensor) -> Tensor:
        if z.dim() == 2:
            return self.diffusion_kernel(z.unsqueeze(1)).squeeze(1)

        b, t, d = z.shape
        flat = z.reshape(b * t, d)
        diff = self.diffusion_kernel(flat.unsqueeze(1)).squeeze(1)
        return diff.reshape(b, t, d)

    def compute_weights(self, z: Tensor) -> Tensor:
        route_z = self._routing_state(z)
        logits = self.gate(route_z) / self.temperature
        return torch.softmax(logits, dim=-1)

    def gate_entropy(self, z: Tensor) -> Tensor:
        weights = self.compute_weights(z)
        return -(weights * (weights.clamp_min(1e-8).log())).sum(dim=-1)

    def _mix(self, weights: Tensor, tensors: list[Tensor], z: Tensor) -> Tensor:
        if z.dim() == 2:
            return sum(
                weights[:, i:i + 1] * tensors[i]
                for i in range(self.num_operators)
            )

        w = weights.unsqueeze(1)
        return sum(
            w[..., i:i + 1] * tensors[i]
            for i in range(self.num_operators)
        )

    def predict_distribution(self, z: Tensor):
        route_z = self._routing_state(z)
        weights = self.compute_weights(z)
        op_means = self._apply_operators(z)

        op_logvars = [
            self._broadcast_like(head(route_z), z)
            for head in self.operator_logvar_heads
        ]

        mix_mean = self._mix(weights, op_means, z)
        mix_second = self._mix(
            weights,
            [torch.exp(logv) + mean.pow(2) for mean, logv in zip(op_means, op_logvars)],
            z,
        )

        mix_var = (mix_second - mix_mean.pow(2)).clamp_min(self.variance_floor)

        disagreement = torch.var(torch.stack(op_means, dim=0), dim=0, unbiased=False)
        drift = self._broadcast_like(self.drift_head(route_z), z)
        reaction = self._broadcast_like(self.reaction_head(route_z), z)
        interaction = self._broadcast_like(self.interaction_head(route_z), z)
        diffusion = self._latent_diffusion(z)

        residual_mix = torch.sigmoid(self.residual_mix_logit)

        base = residual_mix * mix_mean + (1.0 - residual_mix) * z
        correction = (
            self.drift_scale * drift
            + self.reaction_scale * reaction
            + self.diffusion_scale * diffusion
            + 0.10 * interaction
        )

        mu = self.output_norm(base + correction)
        mu = self.output_dropout(mu)

        extra_logvar = self._broadcast_like(self.uncertainty_head(route_z), z)
        logvar = (
            extra_logvar
            + 0.5 * torch.log(mix_var + 1e-8)
            + 0.25 * torch.log(disagreement + 1e-8)
        )
        logvar = torch.clamp(logvar, min=-8.0, max=2.0)

        return mu, logvar, weights, op_means, op_logvars, disagreement

    def step(
        self,
        z: Tensor,
        deterministic: bool | None = None,
        return_stats: bool = False,
    ):
        mu, logvar, weights, op_means, op_logvars, disagreement = self.predict_distribution(z)

        if deterministic is None:
            deterministic = not self.training

        if deterministic:
            out = mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            out = mu + eps * std

        if return_stats:
            return out, {
                "mu": mu,
                "logvar": logvar,
                "weights": weights,
                "operator_means": op_means,
                "operator_logvars": op_logvars,
                "disagreement": disagreement,
            }

        return out

    def forward(self, z: Tensor) -> Tensor:
        if self.horizon == 1:
            return self.step(z)

        preds = []
        z_curr = z

        for _ in range(self.horizon):
            z_curr = self.step(z_curr)
            preds.append(z_curr)

        return torch.stack(preds, dim=1)