from __future__ import annotations

import numpy as np
import pytest

from fi_jepa.model import FIJEPA, fit_ridge_probe


def _valid_batches() -> tuple[np.ndarray, np.ndarray]:
    contexts = np.zeros((2, 3, 2), dtype=np.float64)
    targets = np.zeros((2, 1, 2), dtype=np.float64)
    return contexts, targets


def test_fit_rejects_malformed_epoch_controls_before_training() -> None:
    model = FIJEPA(features=2, target_length=1, seed=7)
    contexts, targets = _valid_batches()

    for invalid in (True, 1.5, 0, -1):
        with pytest.raises(ValueError, match="epochs must be a positive integer"):
            model.fit(contexts, targets, epochs=invalid)  # type: ignore[arg-type]


def test_probe_rejects_non_finite_or_non_numeric_ridge_controls() -> None:
    model = FIJEPA(features=2, target_length=1, seed=7)
    contexts, targets = _valid_batches()

    invalid_values: tuple[object, ...] = (
        True,
        "0.01",
        np.nan,
        np.inf,
        -np.inf,
        0.0,
        -1.0,
    )
    for invalid in invalid_values:
        with pytest.raises(ValueError, match="ridge must be a finite positive number"):
            fit_ridge_probe(
                model,
                contexts,
                targets,
                contexts,
                targets,
                ridge=invalid,  # type: ignore[arg-type]
            )
