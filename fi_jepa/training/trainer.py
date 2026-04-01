from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fi_jepa.training.ema import EMA


class FIJEPATrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        device: str = "cuda",
        ema_tau: float = 0.996,
        grad_clip: float = 1.0,
    ):
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.grad_clip = grad_clip

        if not hasattr(self.model, "context_encoder") or not hasattr(self.model, "target_encoder"):
            raise AttributeError(
                "FIJEPATrainer expects the model to expose `context_encoder` and `target_encoder`."
            )

        self.ema = EMA(
            online_model=self.model.context_encoder,
            target_model=self.model.target_encoder,
            tau=ema_tau,
        )

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, dict):
            if "context" not in batch or "target" not in batch:
                raise KeyError("Batch dict must contain 'context' and 'target' keys.")
            return batch["context"], batch["target"]

        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]

        raise TypeError(
            "Unsupported batch format. Expected dict with keys "
            "`context`/`target` or a tuple/list `(context, target)`."
        )

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

            output = self.model(context, target)
            pred_latent = output.latent_predictions
            target_latent = output.latent_target

            if target_latent is None:
                raise RuntimeError(
                    "Model did not return `latent_target`. Make sure target_window is passed."
                )

            loss = self.loss_fn(pred_latent, target_latent)
            loss.backward()

            if self.grad_clip is not None and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
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

            output = self.model(context, target)
            pred_latent = output.latent_predictions
            target_latent = output.latent_target

            if target_latent is None:
                raise RuntimeError(
                    "Model did not return `latent_target`. Make sure target_window is passed."
                )

            loss = self.loss_fn(pred_latent, target_latent)

            total_loss += float(loss.item())
            num_batches += 1
            progress.set_postfix(loss=f"{loss.item():.6f}")

        return total_loss / max(num_batches, 1)