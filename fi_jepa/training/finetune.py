from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def finetune(
    encoder,
    head: nn.Module,
    dataset,
    optimizer,
    criterion,
    epochs: int = 20,
    batch_size: int = 64,
    device: str = "cuda",
):

    encoder.to(device)
    head.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    for epoch in range(epochs):

        encoder.train()
        head.train()

        total_loss = 0

        for batch in tqdm(loader):

            x = batch["features"].to(device)
            y = batch["target"].to(device)

            optimizer.zero_grad()

            latent = encoder(x)

            preds = head(latent)

            loss = criterion(preds, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Finetune Epoch {epoch+1}/{epochs} "
            f"loss={total_loss/len(loader):.6f}"
        )