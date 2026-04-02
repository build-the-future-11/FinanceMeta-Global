from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from fi_jepa.training.ema import EMA


class FIJEPATrainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        ema_tau: float = 0.996,
        grad_clip: float = 1.0,
        use_amp: bool = True,
    ):
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.grad_clip = grad_clip

        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        if not hasattr(self.model, "context_encoder") or not hasattr(self.model, "target_encoder"):
            raise AttributeError("Model must expose context_encoder and target_encoder")

        self.ema = EMA(
            online_model=self.model.context_encoder,
            target_model=self.model.target_encoder,
            tau=ema_tau,
        )

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, dict):
            return batch["context"], batch["target"]
        return batch[0], batch[1]

    @staticmethod
    def _normalize(z):
        return F.normalize(z, dim=-1)

    def _loss(self, pred, target):

        pred = self._normalize(pred)
        target = self._normalize(target)

        cos_loss = 2 - 2 * (pred * target).sum(dim=-1)
        cos_loss = cos_loss.mean()

        std = torch.sqrt(pred.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1 - std))

        temporal_loss = 0.0
        if pred.dim() == 3:
            temporal_loss = F.mse_loss(pred[:, 1:], pred[:, :-1])

        return cos_loss + 0.1 * var_loss + 0.1 * temporal_loss

    def train_epoch(self, loader: DataLoader):

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(loader, desc="train", leave=False)

        for batch in progress:

            context, target = self._unpack_batch(batch)

            context = context.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):

                output = self.model(context, target)

                pred_latent = output.latent_predictions
                target_latent = output.latent_target

                if target_latent is None:
                    raise RuntimeError("latent_target missing")

                loss = self._loss(pred_latent, target_latent)

            self.scaler.scale(loss).backward()

            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.ema.update()

            total_loss += float(loss.item())
            num_batches += 1

            progress.set_postfix(loss=f"{loss.item():.6f}")

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, loader: DataLoader):

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(loader, desc="val", leave=False)

        for batch in progress:

            context, target = self._unpack_batch(batch)

            context = context.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):

                output = self.model(context, target)

                pred_latent = output.latent_predictions
                target_latent = output.latent_target

                if target_latent is None:
                    raise RuntimeError("latent_target missing")

                loss = self._loss(pred_latent, target_latent)

            total_loss += float(loss.item())
            num_batches += 1

            progress.set_postfix(loss=f"{loss.item():.6f}")

        return total_loss / max(num_batches, 1)