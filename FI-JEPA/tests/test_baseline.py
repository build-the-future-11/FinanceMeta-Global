from __future__ import annotations

import numpy as np
import pytest

from fi_jepa.cli import run
from fi_jepa.data import chronological_windows, make_synthetic_market
from fi_jepa.model import FIJEPA, fit_ridge_probe


def test_synthetic_market_is_deterministic_and_preserves_default_scale_contract() -> None:
    first = make_synthetic_market(seed=11)
    second = make_synthetic_market(seed=11)
    different = make_synthetic_market(seed=12)
    raw = make_synthetic_market(seed=11, normalize=False)

    np.testing.assert_allclose(first, second)
    assert not np.allclose(first, different)
    assert not np.allclose(first, raw)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(first.std(axis=0), 1.0, atol=1e-12)


def test_chronological_split_has_no_shared_observations() -> None:
    series = make_synthetic_market(observations=320, seed=3, normalize=False)
    split = chronological_windows(series, context_length=20, target_length=5)
    train_last_observation = int(split.train_indices.max() + 20 + 5 - 1)
    validation_first_observation = int(split.validation_indices.min())
    assert train_last_observation < split.split_index
    assert validation_first_observation >= split.split_index
    assert train_last_observation < validation_first_observation


def test_chronological_windows_rejects_non_finite_inputs() -> None:
    series = make_synthetic_market(observations=320, seed=19, normalize=False)
    for invalid in (np.nan, np.inf, -np.inf):
        malformed = series.copy()
        malformed[17, 2] = invalid
        with pytest.raises(ValueError, match="series must contain only finite values"):
            chronological_windows(malformed, context_length=20, target_length=5)


def test_chronological_windows_rejects_empty_axes() -> None:
    with pytest.raises(
        ValueError, match="series must have non-empty time and feature axes"
    ):
        chronological_windows(np.empty((100, 0), dtype=np.float64))

    with pytest.raises(
        ValueError, match="series must have non-empty time and feature axes"
    ):
        chronological_windows(np.empty((0, 4), dtype=np.float64))


def test_validation_changes_cannot_change_normalized_training_values() -> None:
    series = make_synthetic_market(observations=320, seed=13, normalize=False)
    split = chronological_windows(series, context_length=20, target_length=5)

    perturbed = series.copy()
    perturbed[split.split_index :] += np.linspace(
        1_000.0, 10_000.0, perturbed.shape[1], dtype=np.float64
    )
    perturbed_split = chronological_windows(
        perturbed, context_length=20, target_length=5
    )

    assert perturbed_split.split_index == split.split_index
    np.testing.assert_allclose(perturbed_split.train_context, split.train_context)
    np.testing.assert_allclose(perturbed_split.train_target, split.train_target)


def test_validation_uses_training_fitted_normalization() -> None:
    series = make_synthetic_market(observations=320, seed=17, normalize=False)
    split = chronological_windows(series, context_length=20, target_length=5)

    train_reference = series[: split.split_index]
    means = train_reference.mean(axis=0, keepdims=True)
    scales = train_reference.std(axis=0, keepdims=True)
    expected = (series - means) / np.where(scales < 1e-12, 1.0, scales)

    first_validation_start = int(split.validation_indices[0])
    np.testing.assert_allclose(
        split.validation_context[0],
        expected[first_validation_start : first_validation_start + 20],
    )
    np.testing.assert_allclose(
        split.validation_target[0],
        expected[
            first_validation_start + 20 : first_validation_start + 20 + 5
        ],
    )


def test_jepa_optimization_reduces_training_objective() -> None:
    series = make_synthetic_market(observations=360, seed=5, normalize=False)
    split = chronological_windows(series, context_length=18, target_length=4)
    model = FIJEPA(features=series.shape[1], target_length=4, seed=5)
    losses = model.fit(split.train_context, split.train_target, epochs=80)
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0] * 0.92
    embeddings = model.encode(split.validation_context)
    assert embeddings.shape == (split.validation_context.shape[0], 12)
    assert float(np.var(embeddings)) > 1e-5


def test_probe_and_cli_report_finite_metrics() -> None:
    series = make_synthetic_market(observations=360, seed=9, normalize=False)
    split = chronological_windows(series, context_length=18, target_length=4)
    model = FIJEPA(features=series.shape[1], target_length=4, seed=9)
    model.fit(split.train_context, split.train_target, epochs=30)
    metrics = fit_ridge_probe(
        model,
        split.train_context,
        split.train_target,
        split.validation_context,
        split.validation_target,
    )
    assert metrics.mse >= 0
    assert metrics.persistence_mse >= 0
    assert 0 <= metrics.directional_accuracy <= 1

    report = run(seed=9, epochs=8)
    assert report["status"] == "synthetic_baseline_only"
    assert report["train_windows"] > 0
    assert report["validation_windows"] > 0
    assert report["final_loss"] < report["initial_loss"]
