import numpy as np

from ancestry_audit.dynamics import (
    hierarchical_did_bootstrap,
    hierarchical_effect_bootstrap,
    observed_difference_in_differences,
    observed_selection_regret,
    selection_policy_bootstrap,
    selection_regret_bootstrap,
)


def test_observed_difference_in_differences() -> None:
    target = np.zeros((2, 3, 5, 8), dtype=float)
    clean = np.zeros((2, 3, 5, 8), dtype=float)
    target[0] = 0.25
    target[1] = -0.125
    observed, by_seed = observed_difference_in_differences(target, clean)
    assert observed.shape == (2, 5)
    assert by_seed.shape == (2, 3, 5)
    assert np.allclose(observed[0], 0.25)
    assert np.allclose(observed[1], -0.125)


def test_hierarchical_bootstrap_is_deterministic() -> None:
    rng = np.random.default_rng(4)
    target = rng.integers(-1, 2, size=(2, 3, 5, 12)).astype(float)
    clean = rng.integers(-1, 2, size=(2, 3, 5, 11)).astype(float)
    first = hierarchical_did_bootstrap(
        target, clean, replicates=120, seed=19, batch_size=17
    )
    second = hierarchical_did_bootstrap(
        target, clean, replicates=120, seed=19, batch_size=17
    )
    assert first.shape == (120, 2, 5)
    assert np.array_equal(first, second)


def test_selection_regret_uses_earliest_tie() -> None:
    target = np.zeros((1, 2, 3, 4), dtype=float)
    clean = np.zeros_like(target)
    downstream = np.zeros_like(target)
    target[:, :, 0] = 1.0
    target[:, :, 1] = 1.0
    clean[:, :, 2] = 1.0
    downstream[:, :, 0] = 0.75
    downstream[:, :, 2] = 0.25
    regret, target_selected, clean_selected = observed_selection_regret(
        target, clean, downstream
    )
    assert np.array_equal(target_selected, [[0, 0]])
    assert np.array_equal(clean_selected, [[2, 2]])
    assert np.allclose(regret, [0.5])


def test_selection_regret_bootstrap_preserves_constant_contrast() -> None:
    target = np.zeros((1, 3, 3, 8), dtype=float)
    clean = np.zeros_like(target)
    downstream = np.zeros_like(target)
    target[:, :, 0] = 1.0
    clean[:, :, 2] = 1.0
    downstream[:, :, 0] = 0.75
    downstream[:, :, 2] = 0.25
    first = selection_regret_bootstrap(
        target,
        clean,
        [downstream],
        replicates=121,
        seed=23,
        batch_size=19,
    )[0]
    second = selection_regret_bootstrap(
        target,
        clean,
        [downstream],
        replicates=121,
        seed=23,
        batch_size=19,
    )[0]
    assert first.shape == (121, 1)
    assert np.array_equal(first, second)
    assert np.allclose(first, 0.5)
    policies = selection_policy_bootstrap(
        target,
        clean,
        [downstream],
        replicates=121,
        seed=23,
        batch_size=19,
    )[0]
    assert np.allclose(policies["target_minus_clean"], 0.5)
    assert np.allclose(policies["target_minus_final"], 0.5)
    assert np.allclose(policies["clean_minus_final"], 0.0)


def test_effect_bootstrap_is_deterministic_and_well_shaped() -> None:
    rng = np.random.default_rng(31)
    delta = rng.integers(-1, 2, size=(4, 3, 5, 17)).astype(float)
    first = hierarchical_effect_bootstrap(
        delta, replicates=103, seed=29, batch_size=16
    )
    second = hierarchical_effect_bootstrap(
        delta, replicates=103, seed=29, batch_size=16
    )
    assert first.shape == (103, 4, 5)
    assert np.array_equal(first, second)
