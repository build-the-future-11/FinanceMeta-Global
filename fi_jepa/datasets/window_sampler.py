from __future__ import annotations

import numpy as np


class WindowSampler:
    """
    Samples context and target windows for FI-JEPA training.

    Example:
        context: [t - C, ..., t]
        target:  [t+1, ..., t+T]
    """

    def __init__(
        self,
        context_length: int = 128,
        target_length: int = 16,
        stride: int = 1,
    ):

        self.context_length = context_length
        self.target_length = target_length
        self.stride = stride

    def compute_valid_indices(self, series_length: int):

        min_idx = self.context_length
        max_idx = series_length - self.target_length

        return np.arange(min_idx, max_idx, self.stride)

    def sample(self, series: np.ndarray, idx: int):

        c_start = idx - self.context_length
        c_end = idx

        t_start = idx
        t_end = idx + self.target_length

        context = series[c_start:c_end]
        target = series[t_start:t_end]

        return context, target