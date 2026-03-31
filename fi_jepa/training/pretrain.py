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
    ):

        self.model = model.to(device)
        self.device = device

        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.ema = EMA(
            online_model=self.model.context_encoder,
            target_model=self.model.target_encoder,
            tau=ema_tau,
        )

    def train_epoch(self, loader: DataLoader):

        self.model.train()

        total_loss = 0.0

        for batch in tqdm(loader):

            context = batch["context"].to(self.device)
            target = batch["target"].to(self.device)

            self.optimizer.zero_grad()

            pred_latent, target_latent = self.model(context, target)

            loss = self.loss_fn(pred_latent, target_latent)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                1.0,
            )

            self.optimizer.step()

            self.ema.update()

            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def validate(self, loader):

        self.model.eval()

        total_loss = 0.0

        for batch in loader:

            context = batch["context"].to(self.device)
            target = batch["target"].to(self.device)

            pred_latent, target_latent = self.model(context, target)

            loss = self.loss_fn(pred_latent, target_latent)

            total_loss += loss.item()

        return total_loss / len(loader)