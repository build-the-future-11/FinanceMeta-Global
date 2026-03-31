import torch


def no_arbitrage_loss(pred_returns):

    drift = pred_returns.mean(dim=0)

    penalty = torch.abs(drift).mean()

    return penalty