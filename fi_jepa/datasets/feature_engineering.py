from __future__ import annotations

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from fi_jepa.datasets.window_sampler import WindowSampler
from fi_jepa.datasets.feature_engineering import FeatureEngineer


class MarketDataset(Dataset):
    """
    Financial time-series dataset for FI-JEPA.

    Returns:
        {
            "context": Tensor [C, F]
            "target": Tensor [T, F]
        }
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        context_length: int = 128,
        target_length: int = 16,
        stride: int = 1,
    ):

        self.feature_engineer = FeatureEngineer()

        self.features = self.feature_engineer.transform(dataframe)

        self.sampler = WindowSampler(
            context_length=context_length,
            target_length=target_length,
            stride=stride,
        )

        self.indices = self.sampler.compute_valid_indices(
            len(self.features)
        )

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):

        series_idx = self.indices[idx]

        context, target = self.sampler.sample(
            self.features,
            series_idx,
        )

        context = torch.from_numpy(context)
        target = torch.from_numpy(target)

        return {
            "context": context,
            "target": target,
        }