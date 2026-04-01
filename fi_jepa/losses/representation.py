class MacroRegimeRegularizer(nn.Module):
    """
    Encourages smooth macro regime evolution.
    """

    def __init__(self, strength: float = 0.05):
        super().__init__()
        self.strength = strength

    def forward(self, latent_sequence: torch.Tensor) -> torch.Tensor:

        # latent_sequence: [B, T, D]

        delta = latent_sequence[:, 1:] - latent_sequence[:, :-1]

        penalty = torch.norm(delta, dim=-1)

        return self.strength * penalty.mean()