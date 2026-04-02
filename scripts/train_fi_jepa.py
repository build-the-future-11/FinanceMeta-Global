from __future__ import annotations

import argparse
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from fi_jepa.models.fi_jepa import FIJEPA
from fi_jepa.training.trainer import FIJEPATrainer


# ---------------------------------------------------------
# Synthetic Dataset
# ---------------------------------------------------------

class SyntheticMarketDataset(Dataset):

    def __init__(
        self,
        num_samples: int,
        context_length: int,
        pred_horizon: int,
        input_dim: int,
        seed: int = 42,
    ):
        super().__init__()

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
        return len(self.samples)

    def __getitem__(self, idx):
        context, target = self.samples[idx]
        return {
            "context": context.clone(),
            "target": target.clone(),
        }


# ---------------------------------------------------------
# Model Builder
# ---------------------------------------------------------

def build_model(args) -> FIJEPA:
    return FIJEPA(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        pred_horizon=args.pred_horizon,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        dropout=args.dropout,
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=256)
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

    # ---------------------------------------------------------
    # Seed
    # ---------------------------------------------------------

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

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
        pin_memory=device.startswith("cuda"),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.startswith("cuda"),
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = build_model(args)

    # ---------------------------------------------------------
    # Sanity Check
    # ---------------------------------------------------------

    x = torch.randn(2, args.context_length, args.input_dim)
    y = torch.randn(2, args.pred_horizon, args.input_dim)

    out = model(x, y)

    assert out.latent_predictions.shape == (2, args.pred_horizon, args.latent_dim)
    assert out.latent_target.shape == (2, args.pred_horizon, args.latent_dim)

    print("Model sanity check passed")

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    # ---------------------------------------------------------
    # Trainer (FIXED)
    # ---------------------------------------------------------

    trainer = FIJEPATrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        ema_tau=args.ema_tau,
        grad_clip=args.grad_clip,
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    for epoch in range(1, args.epochs + 1):

        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

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