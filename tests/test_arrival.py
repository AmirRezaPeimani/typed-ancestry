import numpy as np

from scripts.analyze_ancestor_arrival import (
    hierarchical_bootstrap,
    observed_contrast,
    processed_rows_by_fraction,
    randomized_average_contrast,
)


def test_processed_rows_match_frozen_training_loop() -> None:
    assert processed_rows_by_fraction() == {
        0.2: 902,
        0.4: 1805,
        0.6: 2707,
        0.8: 3610,
        1.0: 4512,
    }


def test_observed_arrival_contrast() -> None:
    delta = np.zeros((1, 2, 2, 6), dtype=float)
    mask = np.zeros((2, 2, 6), dtype=bool)
    mask[:, :, :3] = True
    delta[:, :, :, :3] = 0.75
    delta[:, :, :, 3:] = 0.25
    observed, by_seed = observed_contrast(delta, mask)
    assert by_seed.shape == (1, 2, 2)
    assert np.allclose(by_seed, 0.5)
    assert np.allclose(observed, 0.5)


def test_arrival_bootstrap_preserves_constant_contrast() -> None:
    target = np.zeros((1, 3, 2, 12), dtype=float)
    clean = np.zeros_like(target)
    mask = np.zeros((3, 2, 12), dtype=bool)
    mask[:, :, :6] = True
    target[:, :, :, :6] = 0.75
    target[:, :, :, 6:] = 0.25
    first = hierarchical_bootstrap(
        target,
        mask,
        clean,
        mask,
        replicates=111,
        seed=17,
        batch_size=19,
    )
    second = hierarchical_bootstrap(
        target,
        mask,
        clean,
        mask,
        replicates=111,
        seed=17,
        batch_size=19,
    )
    assert np.array_equal(first["target"], second["target"])
    assert np.allclose(first["target"], 0.5)
    assert np.allclose(first["clean_placebo"], 0.0)
    assert np.allclose(first["target_minus_placebo"], 0.5)


def test_arrival_randomization_is_deterministic() -> None:
    rng = np.random.default_rng(9)
    delta = rng.normal(size=(2, 3, 2, 10))
    positions = np.tile(np.arange(10), (3, 1))
    thresholds = np.asarray([3, 7])
    first = randomized_average_contrast(
        delta,
        positions,
        thresholds,
        replicates=103,
        seed=21,
        batch_size=17,
    )
    second = randomized_average_contrast(
        delta,
        positions,
        thresholds,
        replicates=103,
        seed=21,
        batch_size=17,
    )
    assert first.shape == (103, 2)
    assert np.array_equal(first, second)
