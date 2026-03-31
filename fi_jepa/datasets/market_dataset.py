from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Converts OHLCV data into model features.

    Input dataframe must contain:
    open, high, low, close, volume
    """

    def __init__(
        self,
        add_returns: bool = True,
        add_volatility: bool = True,
        add_volume_features: bool = True,
    ):

        self.add_returns = add_returns
        self.add_volatility = add_volatility
        self.add_volume_features = add_volume_features

    def transform(self, df: pd.DataFrame):

        features = []

        close = df["close"]

        if self.add_returns:

            log_returns = np.log(close).diff().fillna(0)

            features.append(log_returns.values)

        if self.add_volatility:

            vol_5 = close.pct_change().rolling(5).std().fillna(0)
            vol_20 = close.pct_change().rolling(20).std().fillna(0)

            features.append(vol_5.values)
            features.append(vol_20.values)

        if self.add_volume_features:

            vol = df["volume"]

            vol_z = (vol - vol.rolling(20).mean()) / (
                vol.rolling(20).std() + 1e-6
            )

            features.append(vol_z.fillna(0).values)

        spread = (df["high"] - df["low"]) / df["close"]

        features.append(spread.fillna(0).values)

        feature_matrix = np.stack(features, axis=1)

        return feature_matrix.astype(np.float32)