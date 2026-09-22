from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class WindowedSplit:
    train_context: FloatArray
    train_target: FloatArray
    validation_context: FloatArray
    validation_target: FloatArray
    train_indices: IntArray
    validation_indices: IntArray
    split_index: int


def _require_integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    integer = int(value)
    if integer < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return integer


def make_synthetic_market(
    *,
    observations: int = 640,
    features: int = 6,
    seed: int = 7,
    normalize: bool = True,
) -> FloatArray:
    """Create a deterministic regime-switching panel.

    The historical public behavior remains normalized by default. Internal
    leakage-safe pipelines can request the raw generated scale and then fit any
    normalization only after the chronological train boundary is fixed.
    """
    observations = _require_integer("observations", observations, minimum=80)
    features = _require_integer("features", features, minimum=2)
    seed = _require_integer("seed", seed, minimum=0)
    if not isinstance(normalize, (bool, np.bool_)):
        raise ValueError("normalize must be a boolean")
    normalize = bool(normalize)

    rng = np.random.default_rng(seed)
    series = np.zeros((observations, features), dtype=np.float64)
    loadings = rng.normal(0.15, 0.04, size=features)
    idiosyncratic_scale = np.linspace(0.35, 0.75, features)
    factor = 0.0

    for t in range(1, observations):
        regime = 0.55 if (t // 80) % 2 == 0 else -0.25
        factor = regime * factor + rng.normal(0.0, 0.8)
        autoregressive = 0.35 * series[t - 1]
        cross_section = loadings * factor
        noise = rng.normal(0.0, idiosyncratic_scale)
        series[t] = autoregressive + cross_section + noise

    if not normalize:
        return series

    means = series.mean(axis=0, keepdims=True)
    scales = series.std(axis=0, keepdims=True)
    return (series - means) / np.where(scales < 1e-12, 1.0, scales)


def chronological_windows(
    series: FloatArray,
    *,
    context_length: int = 24,
    target_length: int = 6,
    train_fraction: float = 0.7,
) -> WindowedSplit:
    """Create disjoint normalized windows using training-only normalization statistics."""
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("series must have shape [time, features]")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("series must have non-empty time and feature axes")
    if not np.isfinite(values).all():
        raise ValueError("series must contain only finite values")

    context_length = _require_integer("context_length", context_length, minimum=2)
    target_length = _require_integer("target_length", target_length, minimum=1)

    if isinstance(train_fraction, (bool, np.bool_)) or not isinstance(
        train_fraction, (int, float, np.integer, np.floating)
    ):
        raise ValueError("train_fraction must be a finite number in [0.5, 0.9)")
    train_fraction = float(train_fraction)
    if not np.isfinite(train_fraction) or not 0.5 <= train_fraction < 0.9:
        raise ValueError("train_fraction must be a finite number in [0.5, 0.9)")

    total = values.shape[0]
    split_index = int(total * train_fraction)
    window_span = context_length + target_length
    if split_index < window_span or total - split_index < window_span:
        raise ValueError("not enough observations for disjoint chronological windows")

    train_reference = values[:split_index]
    with np.errstate(over="ignore", invalid="ignore"):
        means = train_reference.mean(axis=0, keepdims=True)
        scales = train_reference.std(axis=0, keepdims=True)
    if not np.isfinite(means).all() or not np.isfinite(scales).all():
        raise ValueError("training normalization statistics must be finite")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = (values - means) / np.where(scales < 1e-12, 1.0, scales)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized series must contain only finite values")

    starts = np.arange(0, total - window_span + 1, dtype=np.int64)
    context_end = starts + context_length
    target_end = context_end + target_length

    train_mask = target_end <= split_index
    validation_mask = starts >= split_index
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("not enough observations for disjoint chronological windows")

    def build(indices: IntArray) -> tuple[FloatArray, FloatArray]:
        contexts = np.stack([normalized[i : i + context_length] for i in indices])
        targets = np.stack(
            [
                normalized[i + context_length : i + context_length + target_length]
                for i in indices
            ]
        )
        return contexts, targets

    train_indices = starts[train_mask]
    validation_indices = starts[validation_mask]
    train_context, train_target = build(train_indices)
    validation_context, validation_target = build(validation_indices)

    return WindowedSplit(
        train_context=train_context,
        train_target=train_target,
        validation_context=validation_context,
        validation_target=validation_target,
        train_indices=train_indices,
        validation_indices=validation_indices,
        split_index=split_index,
    )
