from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:

    def __init__(
        self,
        add_returns: bool = True,
        add_volatility: bool = True,
        add_volume_features: bool = True,
        add_microstructure: bool = True,
        add_trend: bool = True,
        eps: float = 1e-6,
    ):
        self.add_returns = add_returns
        self.add_volatility = add_volatility
        self.add_volume_features = add_volume_features
        self.add_microstructure = add_microstructure
        self.add_trend = add_trend
        self.eps = eps

    def _safe_log(self, x):
        return np.log(np.clip(x, self.eps, None))

    def _zscore(self, x, window):
        mean = x.rolling(window).mean()
        std = x.rolling(window).std()
        return (x - mean) / (std + self.eps)

    def _realized_vol(self, returns, window):
        return returns.rolling(window).std()

    def transform(self, df: pd.DataFrame):

        df = df.copy()

        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        features = []

        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]
        volume = df["volume"]

        log_close = self._safe_log(close)

        if self.add_returns:
            log_returns = log_close.diff()
            returns = close.pct_change()

            features.append(log_returns.fillna(0).values)
            features.append(returns.fillna(0).values)

        if self.add_volatility:
            returns = close.pct_change()

            vol_5 = self._realized_vol(returns, 5)
            vol_20 = self._realized_vol(returns, 20)
            vol_60 = self._realized_vol(returns, 60)

            features.append(vol_5.fillna(0).values)
            features.append(vol_20.fillna(0).values)
            features.append(vol_60.fillna(0).values)

        if self.add_volume_features:
            vol_z_20 = self._zscore(volume, 20)
            vol_z_60 = self._zscore(volume, 60)

            vol_change = volume.pct_change()

            features.append(vol_z_20.fillna(0).values)
            features.append(vol_z_60.fillna(0).values)
            features.append(vol_change.fillna(0).values)

        if self.add_microstructure:
            spread = (high - low) / (close + self.eps)

            body = (close - open_) / (close + self.eps)
            wick_up = (high - np.maximum(open_, close)) / (close + self.eps)
            wick_down = (np.minimum(open_, close) - low) / (close + self.eps)

            features.append(spread.fillna(0).values)
            features.append(body.fillna(0).values)
            features.append(wick_up.fillna(0).values)
            features.append(wick_down.fillna(0).values)

        if self.add_trend:
            ma_10 = close.rolling(10).mean()
            ma_50 = close.rolling(50).mean()

            trend_short = (close - ma_10) / (ma_10 + self.eps)
            trend_long = (close - ma_50) / (ma_50 + self.eps)

            momentum_10 = close.pct_change(10)
            momentum_20 = close.pct_change(20)

            features.append(trend_short.fillna(0).values)
            features.append(trend_long.fillna(0).values)
            features.append(momentum_10.fillna(0).values)
            features.append(momentum_20.fillna(0).values)

        feature_matrix = np.column_stack(features)

        return feature_matrix.astype(np.float32)