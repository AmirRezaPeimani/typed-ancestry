"""Statistical utilities for the frozen factorial learning-dynamics analysis."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def observed_difference_in_differences(
    target_delta: np.ndarray, clean_delta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return aggregate and seed-level DiD values.

    Arrays have shape ``(model, seed, checkpoint, context)`` and contain
    paired naive-minus-safe correctness differences.
    """

    if target_delta.ndim != 4 or clean_delta.ndim != 4:
        raise ValueError("delta arrays must be model × seed × checkpoint × row")
    if target_delta.shape[:3] != clean_delta.shape[:3]:
        raise ValueError("target and clean leading dimensions differ")
    seed_values = target_delta.mean(axis=-1) - clean_delta.mean(axis=-1)
    return seed_values.mean(axis=1), seed_values


def _sample_context_means(
    values: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """Sample context means as batch × model × seed × checkpoint."""

    sampled = values[..., indices]
    return sampled.mean(axis=-1).transpose(3, 0, 1, 2)


def _seed_counts(draws: np.ndarray, seed_count: int) -> np.ndarray:
    return np.stack(
        [(draws == index).sum(axis=1) for index in range(seed_count)],
        axis=1,
    )


def hierarchical_did_bootstrap(
    target_delta: np.ndarray,
    clean_delta: np.ndarray,
    *,
    replicates: int,
    seed: int,
    batch_size: int = 200,
) -> np.ndarray:
    """Paired context and hierarchical training-seed bootstrap for the DiD."""

    observed, _ = observed_difference_in_differences(
        target_delta, clean_delta
    )
    model_count, seed_count, checkpoint_count = target_delta.shape[:3]
    result = np.empty(
        (replicates, model_count, checkpoint_count), dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    target_n = target_delta.shape[-1]
    clean_n = clean_delta.shape[-1]

    for start in range(0, replicates, batch_size):
        stop = min(replicates, start + batch_size)
        count = stop - start
        target_indices = rng.integers(
            0, target_n, size=(count, target_n)
        )
        clean_indices = rng.integers(0, clean_n, size=(count, clean_n))
        seed_draws = rng.integers(
            0, seed_count, size=(count, seed_count)
        )
        weights = _seed_counts(seed_draws, seed_count)
        target_means = _sample_context_means(
            target_delta, target_indices
        )
        clean_means = _sample_context_means(clean_delta, clean_indices)
        did_by_seed = target_means - clean_means
        result[start:stop] = (
            did_by_seed * weights[:, None, :, None]
        ).sum(axis=2) / seed_count

    if result.shape[1:] != observed.shape:
        raise AssertionError("bootstrap result has an unexpected shape")
    return result


def hierarchical_effect_bootstrap(
    delta: np.ndarray,
    *,
    replicates: int,
    seed: int,
    batch_size: int = 250,
) -> np.ndarray:
    """Bootstrap a naive-minus-safe effect for one evaluation set."""

    if delta.ndim != 4:
        raise ValueError("delta must be model × seed × checkpoint × row")
    model_count, seed_count, checkpoint_count, row_count = delta.shape
    result = np.empty(
        (replicates, model_count, checkpoint_count), dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    for start in range(0, replicates, batch_size):
        stop = min(replicates, start + batch_size)
        count = stop - start
        row_indices = rng.integers(0, row_count, size=(count, row_count))
        seed_draws = rng.integers(
            0, seed_count, size=(count, seed_count)
        )
        weights = _seed_counts(seed_draws, seed_count)
        means = _sample_context_means(delta, row_indices)
        result[start:stop] = (
            means * weights[:, None, :, None]
        ).sum(axis=2) / seed_count
    return result


def earliest_argmax(values: np.ndarray) -> np.ndarray:
    """Return the earliest maximum along the final axis."""

    return np.argmax(values, axis=-1)


def observed_selection_regret(
    target: np.ndarray,
    clean: np.ndarray,
    downstream: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute target- versus clean-selected downstream performance.

    Inputs have shape ``(model, seed, checkpoint, context)`` and come from
    the naive-boundary run. Returns model-level mean regret plus the two
    seed-level selected-checkpoint indices.
    """

    if target.shape[:3] != clean.shape[:3]:
        raise ValueError("validation arrays do not align")
    if target.shape[:3] != downstream.shape[:3]:
        raise ValueError("downstream array does not align")
    target_selection = earliest_argmax(target.mean(axis=-1))
    clean_selection = earliest_argmax(clean.mean(axis=-1))
    downstream_accuracy = downstream.mean(axis=-1)
    selected_target = np.take_along_axis(
        downstream_accuracy, target_selection[..., None], axis=2
    ).squeeze(axis=2)
    selected_clean = np.take_along_axis(
        downstream_accuracy, clean_selection[..., None], axis=2
    ).squeeze(axis=2)
    return (
        (selected_target - selected_clean).mean(axis=1),
        target_selection,
        clean_selection,
    )


def selection_regret_bootstrap(
    target: np.ndarray,
    clean: np.ndarray,
    downstream_sets: Iterable[np.ndarray],
    *,
    replicates: int,
    seed: int,
    batch_size: int = 150,
) -> list[np.ndarray]:
    """Bootstrap validation selection and downstream evaluation jointly."""

    policies = selection_policy_bootstrap(
        target,
        clean,
        downstream_sets,
        replicates=replicates,
        seed=seed,
        batch_size=batch_size,
    )
    return [item["target_minus_clean"] for item in policies]


def selection_policy_bootstrap(
    target: np.ndarray,
    clean: np.ndarray,
    downstream_sets: Iterable[np.ndarray],
    *,
    replicates: int,
    seed: int,
    batch_size: int = 150,
) -> list[dict[str, np.ndarray]]:
    """Bootstrap target-, clean-, and final-checkpoint policy contrasts."""

    downstream = list(downstream_sets)
    if not downstream:
        raise ValueError("at least one downstream array is required")
    model_count, seed_count, checkpoint_count = target.shape[:3]
    target_n = target.shape[-1]
    clean_n = clean.shape[-1]
    results = [
        {
            name: np.empty(
                (replicates, model_count), dtype=np.float64
            )
            for name in (
                "target_minus_clean",
                "target_minus_final",
                "clean_minus_final",
            )
        }
        for _ in downstream
    ]
    rng = np.random.default_rng(seed)

    for start in range(0, replicates, batch_size):
        stop = min(replicates, start + batch_size)
        count = stop - start
        target_indices = rng.integers(
            0, target_n, size=(count, target_n)
        )
        clean_indices = rng.integers(0, clean_n, size=(count, clean_n))
        target_means = _sample_context_means(target, target_indices)
        clean_means = _sample_context_means(clean, clean_indices)
        target_selection = earliest_argmax(target_means)
        clean_selection = earliest_argmax(clean_means)
        seed_draws = rng.integers(
            0, seed_count, size=(count, seed_count)
        )
        weights = _seed_counts(seed_draws, seed_count)

        for outputs, values in zip(results, downstream):
            row_n = values.shape[-1]
            row_indices = rng.integers(
                0, row_n, size=(count, row_n)
            )
            means = _sample_context_means(values, row_indices)
            target_selected = np.take_along_axis(
                means, target_selection[..., None], axis=3
            ).squeeze(axis=3)
            clean_selected = np.take_along_axis(
                means, clean_selection[..., None], axis=3
            ).squeeze(axis=3)
            final_checkpoint = means[..., -1]
            contrasts = {
                "target_minus_clean": target_selected - clean_selected,
                "target_minus_final": target_selected - final_checkpoint,
                "clean_minus_final": clean_selected - final_checkpoint,
            }
            for name, by_seed in contrasts.items():
                outputs[name][start:stop] = (
                    by_seed * weights[:, None, :]
                ).sum(axis=2) / seed_count

    if any(
        values.shape != (replicates, model_count)
        for result in results
        for values in result.values()
    ):
        raise AssertionError("selection bootstrap result shape changed")
    return results
