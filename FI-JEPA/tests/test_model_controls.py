from __future__ import annotations

import numpy as np
import pytest

from fi_jepa.model import FIJEPA, fit_ridge_probe


def _valid_batches() -> tuple[np.ndarray, np.ndarray]:
    contexts = np.zeros((2, 3, 2), dtype=np.float64)
    targets = np.zeros((2, 1, 2), dtype=np.float64)
    return contexts, targets


def test_constructor_rejects_malformed_dimension_and_seed_controls() -> None:
    for invalid in (True, 1.5, 0, -1):
        with pytest.raises(ValueError, match="features must be a positive integer"):
            FIJEPA(features=invalid, target_length=1)  # type: ignore[arg-type]

    for invalid in (True, 1.5, 0, -1):
        with pytest.raises(ValueError, match="target_length must be a positive integer"):
            FIJEPA(features=2, target_length=invalid)  # type: ignore[arg-type]

    for invalid in (True, 1.5, 0, -1):
        with pytest.raises(ValueError, match="embedding_dim must be"):
            FIJEPA(features=2, target_length=1, embedding_dim=invalid)  # type: ignore[arg-type]

    for invalid in (True, 1.5, -1):
        with pytest.raises(ValueError, match="seed must be a non-negative integer"):
            FIJEPA(features=2, target_length=1, seed=invalid)  # type: ignore[arg-type]


def test_constructor_rejects_non_finite_or_non_numeric_rate_controls() -> None:
    invalid_values: tuple[object, ...] = (True, "0.04", np.nan, np.inf, -np.inf)
    for invalid in invalid_values:
        with pytest.raises(ValueError, match="learning_rate must be a finite number"):
            FIJEPA(features=2, target_length=1, learning_rate=invalid)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="ema_momentum must be a finite number"):
            FIJEPA(features=2, target_length=1, ema_momentum=invalid)  # type: ignore[arg-type]

    for invalid in (0.0, -0.1, 0.21):
        with pytest.raises(ValueError, match="learning_rate must be in"):
            FIJEPA(features=2, target_length=1, learning_rate=invalid)

    for invalid in (-0.1, 1.0):
        with pytest.raises(ValueError, match="ema_momentum must be in"):
            FIJEPA(features=2, target_length=1, ema_momentum=invalid)


def test_context_and_target_inputs_must_be_nonempty_and_finite() -> None:
    model = FIJEPA(features=2, target_length=1, seed=7)
    contexts, targets = _valid_batches()

    with pytest.raises(ValueError, match="contexts must contain at least one non-empty window"):
        model.encode(np.zeros((2, 0, 2), dtype=np.float64))

    with pytest.raises(ValueError, match="contexts must contain at least one non-empty window"):
        model.fit(np.zeros((0, 3, 2), dtype=np.float64), np.zeros((0, 1, 2)), epochs=1)

    non_finite_contexts = contexts.copy()
    non_finite_contexts[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="contexts must contain only finite values"):
        model.fit(non_finite_contexts, targets, epochs=1)

    non_finite_targets = targets.copy()
    non_finite_targets[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="targets must contain only finite values"):
        model.fit(contexts, non_finite_targets, epochs=1)


def test_loss_and_probe_reject_misaligned_batches() -> None:
    model = FIJEPA(features=2, target_length=1, seed=7)
    contexts, targets = _valid_batches()

    with pytest.raises(ValueError, match="context and target batches must align"):
        model.loss(contexts[:1], targets)

    with pytest.raises(ValueError, match="training context and target batches must align"):
        fit_ridge_probe(model, contexts[:1], targets, contexts, targets)

    with pytest.raises(ValueError, match="validation context and target batches must align"):
        fit_ridge_probe(model, contexts, targets, contexts[:1], targets)


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
