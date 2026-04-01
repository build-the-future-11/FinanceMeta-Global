from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from fi_jepa.models.fi_jepa import FIJEPA
from fi_jepa.training.trainer import FIJEPATrainer


class SyntheticMarketDataset(Dataset):
    """
    Smoke-test dataset for end-to-end training.

    Generates smooth synthetic series so the model has a learnable signal.
    Each sample returns:
        context: [context_length, input_dim]
        target:  [pred_horizon, input_dim]
    """

    def __init__(
        self,
        num_samples: int,
        context_length: int,
        pred_horizon: int,
        input_dim: int,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.context_length = context_length
        self.pred_horizon = pred_horizon
        self.input_dim = input_dim

        g = torch.Generator().manual_seed(seed)
        total_len = context_length + pred_horizon

        self.samples = []
        for _ in range(num_samples):
            base = torch.randn(total_len, input_dim, generator=g) * 0.05
            walk = torch.cumsum(base, dim=0)
            trend = torch.linspace(0, 1, total_len).unsqueeze(-1)
            trend = trend * torch.randn(1, input_dim, generator=g) * 0.02
            series = walk + trend

            context = series[:context_length]
            target = series[context_length:]
            self.samples.append((context, target))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        context, target = self.samples[idx]
        return {
            "context": context.clone(),
            "target": target.clone(),
        }


def build_model(args) -> FIJEPA:
    return FIJEPA(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        num_assets=args.num_assets,
        context_length=args.context_length,
        pred_horizon=args.pred_horizon,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        dropout=args.dropout,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--num-assets", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--pred-horizon", type=int, default=16)

    parser.add_argument("--encoder-depth", type=int, default=6)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ema-tau", type=float, default=0.996)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--train-split", type=float, default=0.9)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    dataset = SyntheticMarketDataset(
        num_samples=args.num_samples,
        context_length=args.context_length,
        pred_horizon=args.pred_horizon,
        input_dim=args.input_dim,
        seed=args.seed,
    )

    train_size = int(len(dataset) * args.train_split)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )

    model = build_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.MSELoss()

    trainer = FIJEPATrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        ema_tau=args.ema_tau,
        grad_clip=args.grad_clip,
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        "fi_jepa_checkpoint.pt",
    )
    print("saved checkpoint to fi_jepa_checkpoint.pt")


if __name__ == "__main__":
    main()