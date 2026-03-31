import torch
import torch.nn.functional as F


def latent_prediction_loss(pred, target):

    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target.detach(), dim=-1)

    return 1 - (pred * target).sum(-1).mean()