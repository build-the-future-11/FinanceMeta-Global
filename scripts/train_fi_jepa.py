import torch
from torch.utils.data import DataLoader

from fi_jepa.utils.config import load_config
from fi_jepa.models.context_encoder import TransformerEncoder
from fi_jepa.models.predictor_bank import PredictorBank
from fi_jepa.training.ema import update_ema
from fi_jepa.losses.latent_prediction import latent_prediction_loss


def train():

    model_cfg = load_config("configs/model/fi_jepa_base.yaml")
    train_cfg = load_config("configs/training/pretrain.yaml")

    encoder_cfg = model_cfg["model"]["context_encoder"]

    encoder = TransformerEncoder(
        encoder_cfg["input_dim"],
        encoder_cfg["embed_dim"],
        encoder_cfg["depth"],
        encoder_cfg["heads"],
        encoder_cfg["mlp_ratio"],
        encoder_cfg["dropout"],
    )

    predictor = PredictorBank(
        model_cfg["model"]["prediction"]["latent_dim"],
        model_cfg["model"]["predictor_bank"]["operators"],
    )

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=train_cfg["training"]["learning_rate"],
    )

    for step in range(10000):

        x = torch.randn(64, 128, 64)
        future = torch.randn(64, 128, 64)

        h = encoder(x)

        pred = predictor(h)

        with torch.no_grad():
            target = encoder(future)

        loss = latent_prediction_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print("step", step, "loss", loss.item())


if __name__ == "__main__":
    train()