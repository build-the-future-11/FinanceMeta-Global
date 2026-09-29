import numpy as np

from fi_jepa.data import chronological_windows, make_synthetic_market


def test_post_split_validation_perturbation_cannot_change_normalized_training_values():
    series = make_synthetic_market(observations=320, features=4, seed=11, normalize=False)
    original = chronological_windows(series, context_length=16, target_length=4, train_fraction=0.7)

    perturbed = series.copy()
    perturbed[original.split_index:] = (
        perturbed[original.split_index:] * 1000.0 + 12345.0
    )
    changed = chronological_windows(
        perturbed,
        context_length=16,
        target_length=4,
        train_fraction=0.7,
    )

    np.testing.assert_allclose(original.train_context, changed.train_context)
    np.testing.assert_allclose(original.train_target, changed.train_target)
    np.testing.assert_array_equal(original.train_indices, changed.train_indices)


def test_validation_uses_training_fitted_normalization_statistics():
    series = make_synthetic_market(observations=320, features=4, seed=13, normalize=False)
    split = chronological_windows(
        series,
        context_length=16,
        target_length=4,
        train_fraction=0.7,
    )

    train_reference = series[: split.split_index]
    train_mean = train_reference.mean(axis=0, keepdims=True)
    train_scale = train_reference.std(axis=0, keepdims=True)
    train_scale = np.where(train_scale < 1e-12, 1.0, train_scale)
    expected = (series - train_mean) / train_scale

    first_validation_start = int(split.validation_indices[0])
    np.testing.assert_allclose(
        split.validation_context[0],
        expected[first_validation_start : first_validation_start + 16],
    )
    np.testing.assert_allclose(
        split.validation_target[0],
        expected[
            first_validation_start + 16 : first_validation_start + 20
        ],
    )

    validation_reference = series[split.split_index:]
    validation_mean = validation_reference.mean(axis=0, keepdims=True)
    assert not np.allclose(train_mean, validation_mean)
