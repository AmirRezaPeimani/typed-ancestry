#!/usr/bin/env python3
"""Analyze the competence-passing final neural factorial at the frozen fraction."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


SEEDS = (20260727, 20260728, 20260729)
BOUNDARIES = ("naive", "safe")
SETS = ("target", "clean_validation", "untouched_clean_test")


def read_predictions(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 512:
        raise ValueError(f"expected 512 rows in {path}, observed {len(rows)}")
    values = {row["pair_id"]: float(row["correct"]) for row in rows}
    if len(values) != len(rows):
        raise ValueError(f"duplicate pair identifier in {path}")
    return values


def read_simple_predictions(
    path: Path,
) -> dict[tuple[str, int, str, str], dict[str, float]]:
    selected: dict[tuple[str, int, str, str], dict[str, float]] = {}
    models = ("controlled_word", "controlled_character")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] not in models:
                continue
            if float(row["fraction"]) != 1.0:
                continue
            if row["evaluation_set"] not in ("target", "clean_validation"):
                continue
            key = (
                row["model"],
                int(row["seed"]),
                row["boundary"],
                row["evaluation_set"],
            )
            values = selected.setdefault(key, {})
            pair_id = row["pair_id"]
            if pair_id in values:
                raise ValueError(f"duplicate lexical prediction: {key}/{pair_id}")
            values[pair_id] = float(row["correct"])
    expected = {
        (model, seed, boundary, evaluation_set)
        for model in models
        for seed in SEEDS
        for boundary in BOUNDARIES
        for evaluation_set in ("target", "clean_validation")
    }
    if set(selected) != expected:
        raise ValueError("lexical prediction grid is incomplete")
    if any(len(values) != 512 for values in selected.values()):
        raise ValueError("lexical prediction cells must contain 512 pairs")
    return selected


def paired_gap(
    naive: dict[str, float],
    safe: dict[str, float],
    indices: np.ndarray | None = None,
) -> float:
    if set(naive) != set(safe):
        raise ValueError("naive and safe prediction identifiers differ")
    keys = sorted(naive)
    left = np.asarray([naive[key] for key in keys])
    right = np.asarray([safe[key] for key in keys])
    if indices is not None:
        left = left[indices]
        right = right[indices]
    return float(np.mean(left - right))


def resampled_accuracy(
    values: dict[str, float],
    indices: np.ndarray | None = None,
) -> float:
    array = np.asarray([values[key] for key in sorted(values)])
    if indices is not None:
        array = array[indices]
    return float(np.mean(array))


def interval(draws: np.ndarray) -> list[float]:
    return [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def paired_differences_by_context(
    predictions: dict[tuple[int, str, str], dict[str, float]],
    evaluation_set: str,
    seeds: list[int],
) -> np.ndarray:
    """Average paired ancestor-included-minus-ancestor-excluded correctness within context over seeds."""
    keys = sorted(predictions[(SEEDS[0], "naive", evaluation_set)])
    rows = []
    for seed in seeds:
        naive = predictions[(seed, "naive", evaluation_set)]
        safe = predictions[(seed, "safe", evaluation_set)]
        if sorted(naive) != keys or sorted(safe) != keys:
            raise ValueError(
                f"prediction identifiers differ for {seed}/{evaluation_set}"
            )
        rows.append(
            np.asarray([naive[key] - safe[key] for key in keys], dtype=float)
        )
    return np.mean(np.stack(rows), axis=0)


def unique_seed_representatives(
    predictions: dict[tuple[int, str, str], dict[str, float]],
) -> tuple[list[int], dict[int, list[int]]]:
    groups: dict[str, list[int]] = {}
    for seed in SEEDS:
        digest = hashlib.sha256()
        for boundary in BOUNDARIES:
            for evaluation_set in SETS:
                values = predictions[(seed, boundary, evaluation_set)]
                digest.update(
                    np.asarray(
                        [
                            2 * values[key]
                            for key in sorted(values)
                        ],
                        dtype=np.int8,
                    ).tobytes()
                )
        groups.setdefault(digest.hexdigest(), []).append(seed)
    representatives = [values[0] for values in groups.values()]
    return representatives, {
        values[0]: values for values in groups.values()
    }


def sign_randomization_p(
    observed: float,
    draws: np.ndarray,
) -> float:
    """Monte Carlo two-sided p-value with the standard plus-one adjustment."""
    return float(
        (1 + np.sum(np.abs(draws) >= abs(observed))) / (len(draws) + 1)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint-selection", type=Path, required=True)
    parser.add_argument("--lexical-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261011)
    parser.add_argument("--randomization-seed", type=int, default=20261012)
    parser.add_argument(
        "--protocol",
        default="docs/controlled_factorial_protocol.md",
    )
    parser.add_argument(
        "--interaction-protocol",
        default="docs/controlled_cross_learner_protocol.md",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    selection = json.loads(
        args.checkpoint_selection.read_text(encoding="utf-8")
    )
    if selection["status"] != "selected":
        raise RuntimeError(
            "factorial analysis requires a valid checkpoint-selection record"
        )
    fraction = int(selection["selected_checkpoint_fraction"])

    predictions: dict[tuple[int, str, str], dict[str, float]] = {}
    accuracies = []
    for seed in SEEDS:
        for boundary in BOUNDARIES:
            evaluation = (
                args.root
                / "evaluations"
                / f"{boundary}_seed{seed}"
            )
            for evaluation_set in SETS:
                path = (
                    evaluation
                    / f"checkpoint-{fraction:03d}_{evaluation_set}.csv"
                )
                values = read_predictions(path)
                predictions[(seed, boundary, evaluation_set)] = values
                accuracies.append(
                    {
                        "seed": seed,
                        "boundary": boundary,
                        "evaluation_set": evaluation_set,
                        "accuracy": float(np.mean(list(values.values()))),
                    }
                )
    representative_seeds, seed_groups = unique_seed_representatives(
        predictions
    )
    simple_predictions = read_simple_predictions(args.lexical_predictions)
    for model in ("controlled_word", "controlled_character"):
        for seed in SEEDS:
            for boundary in BOUNDARIES:
                for evaluation_set in ("target", "clean_validation"):
                    if set(
                        simple_predictions[
                            (model, seed, boundary, evaluation_set)
                        ]
                    ) != set(
                        predictions[(seed, boundary, evaluation_set)]
                    ):
                        raise ValueError(
                            "simple and neural evaluation identifiers "
                            f"differ: {model}/{seed}/{boundary}/"
                            f"{evaluation_set}"
                        )

    seed_effects = []
    for seed in SEEDS:
        target_gap = paired_gap(
            predictions[(seed, "naive", "target")],
            predictions[(seed, "safe", "target")],
        )
        clean_gap = paired_gap(
            predictions[(seed, "naive", "clean_validation")],
            predictions[(seed, "safe", "clean_validation")],
        )
        untouched_gap = paired_gap(
            predictions[(seed, "naive", "untouched_clean_test")],
            predictions[(seed, "safe", "untouched_clean_test")],
        )
        seed_effects.append(
            {
                "seed": seed,
                "target_naive_minus_safe": target_gap,
                "clean_naive_minus_safe": clean_gap,
                "difference_in_differences": target_gap - clean_gap,
                "untouched_naive_minus_safe": untouched_gap,
            }
        )

    rng = np.random.default_rng(args.bootstrap_seed)
    did_draws = np.empty(args.bootstrap_replicates)
    untouched_draws = np.empty(args.bootstrap_replicates)
    interaction_models = ("controlled_word", "controlled_character")
    learner_interaction_draws = {
        model: np.empty(args.bootstrap_replicates)
        for model in interaction_models
    }
    accuracy_draws = {
        (boundary, evaluation_set): np.empty(args.bootstrap_replicates)
        for boundary in BOUNDARIES
        for evaluation_set in SETS
    }
    for replicate in range(args.bootstrap_replicates):
        sampled_seeds = rng.integers(
            0,
            len(representative_seeds),
            len(representative_seeds),
        )
        did_values = []
        untouched_values = []
        learner_interaction_values = {
            model: [] for model in interaction_models
        }
        accuracy_values = {
            key: [] for key in accuracy_draws
        }
        for seed_index in sampled_seeds:
            seed = representative_seeds[int(seed_index)]
            target_indices = rng.integers(0, 512, 512)
            clean_indices = rng.integers(0, 512, 512)
            untouched_indices = rng.integers(0, 512, 512)
            target_gap = paired_gap(
                predictions[(seed, "naive", "target")],
                predictions[(seed, "safe", "target")],
                target_indices,
            )
            clean_gap = paired_gap(
                predictions[(seed, "naive", "clean_validation")],
                predictions[(seed, "safe", "clean_validation")],
                clean_indices,
            )
            did_values.append(target_gap - clean_gap)
            untouched_values.append(
                paired_gap(
                    predictions[(seed, "naive", "untouched_clean_test")],
                    predictions[(seed, "safe", "untouched_clean_test")],
                    untouched_indices,
                )
            )
            for model in interaction_models:
                simple_target_gap = paired_gap(
                    simple_predictions[
                        (model, seed, "naive", "target")
                    ],
                    simple_predictions[
                        (model, seed, "safe", "target")
                    ],
                    target_indices,
                )
                simple_clean_gap = paired_gap(
                    simple_predictions[
                        (model, seed, "naive", "clean_validation")
                    ],
                    simple_predictions[
                        (model, seed, "safe", "clean_validation")
                    ],
                    clean_indices,
                )
                learner_interaction_values[model].append(
                    (target_gap - clean_gap)
                    - (simple_target_gap - simple_clean_gap)
                )
            for boundary in BOUNDARIES:
                accuracy_values[(boundary, "target")].append(
                    resampled_accuracy(
                        predictions[(seed, boundary, "target")],
                        target_indices,
                    )
                )
                accuracy_values[(boundary, "clean_validation")].append(
                    resampled_accuracy(
                        predictions[(seed, boundary, "clean_validation")],
                        clean_indices,
                    )
                )
                accuracy_values[(boundary, "untouched_clean_test")].append(
                    resampled_accuracy(
                        predictions[
                            (seed, boundary, "untouched_clean_test")
                        ],
                        untouched_indices,
                    )
                )
        did_draws[replicate] = np.mean(did_values)
        untouched_draws[replicate] = np.mean(untouched_values)
        for model in interaction_models:
            learner_interaction_draws[model][replicate] = np.mean(
                learner_interaction_values[model]
            )
        for key, values in accuracy_values.items():
            accuracy_draws[key][replicate] = np.mean(values)

    did_point = float(
        np.mean(
            [
                row["difference_in_differences"]
                for row in seed_effects
                if row["seed"] in representative_seeds
            ]
        )
    )
    untouched_point = float(
        np.mean(
            [
                row["untouched_naive_minus_safe"]
                for row in seed_effects
                if row["seed"] in representative_seeds
            ]
        )
    )
    simple_seed_effects = {}
    learner_interaction_points = {}
    for model in interaction_models:
        effects = []
        for seed in SEEDS:
            effects.append(
                paired_gap(
                    simple_predictions[
                        (model, seed, "naive", "target")
                    ],
                    simple_predictions[
                        (model, seed, "safe", "target")
                    ],
                )
                - paired_gap(
                    simple_predictions[
                        (model, seed, "naive", "clean_validation")
                    ],
                    simple_predictions[
                        (model, seed, "safe", "clean_validation")
                    ],
                )
            )
        simple_seed_effects[model] = effects
        effect_by_seed = dict(zip(SEEDS, effects))
        learner_interaction_points[model] = float(
            np.mean(
                [
                    neural["difference_in_differences"]
                    - effect_by_seed[neural["seed"]]
                    for neural in seed_effects
                    if neural["seed"] in representative_seeds
                ]
            )
        )

    target_context_differences = paired_differences_by_context(
        predictions, "target", representative_seeds
    )
    clean_context_differences = paired_differences_by_context(
        predictions, "clean_validation", representative_seeds
    )
    untouched_context_differences = paired_differences_by_context(
        predictions, "untouched_clean_test", representative_seeds
    )
    randomization_rng = np.random.default_rng(args.randomization_seed)
    target_signs = (
        2
        * randomization_rng.integers(
            0,
            2,
            size=(
                args.bootstrap_replicates,
                len(target_context_differences),
            ),
            dtype=np.int8,
        )
        - 1
    )
    target_randomized = (
        target_signs @ target_context_differences
    ) / len(target_context_differences)
    del target_signs
    clean_signs = (
        2
        * randomization_rng.integers(
            0,
            2,
            size=(
                args.bootstrap_replicates,
                len(clean_context_differences),
            ),
            dtype=np.int8,
        )
        - 1
    )
    clean_randomized = (
        clean_signs @ clean_context_differences
    ) / len(clean_context_differences)
    del clean_signs
    did_randomized = target_randomized - clean_randomized
    untouched_signs = (
        2
        * randomization_rng.integers(
            0,
            2,
            size=(
                args.bootstrap_replicates,
                len(untouched_context_differences),
            ),
            dtype=np.int8,
        )
        - 1
    )
    untouched_randomized = (
        untouched_signs @ untouched_context_differences
    ) / len(untouched_context_differences)

    payload = {
        "status": "complete",
        "protocol": args.protocol,
        "interaction_protocol": args.interaction_protocol,
        "nominal_seeds": list(SEEDS),
        "unique_prediction_vectors": len(representative_seeds),
        "prediction_vector_seed_groups": {
            str(representative): group
            for representative, group in seed_groups.items()
        },
        "uncertainty_unit": (
            "unique final prediction vector and paired evaluation context"
        ),
        "selected_checkpoint_fraction": fraction,
        "accuracy_rows": accuracies,
        "accuracy_summary": [
            {
                "boundary": boundary,
                "evaluation_set": evaluation_set,
                "accuracy": float(
                    np.mean(
                        [
                            row["accuracy"]
                            for row in accuracies
                            if row["boundary"] == boundary
                            and row["evaluation_set"] == evaluation_set
                            and row["seed"] in representative_seeds
                        ]
                    )
                ),
                "hierarchical_bootstrap_95": interval(
                    accuracy_draws[(boundary, evaluation_set)]
                ),
            }
            for boundary in BOUNDARIES
            for evaluation_set in SETS
        ],
        "seed_effects": seed_effects,
        "learner_interaction": {
            "estimand": (
                "Qwen2.5-1.5B difference-in-differences minus "
                "word-TF-IDF difference-in-differences"
            ),
            "qwen_minus_word_difference_in_differences": (
                learner_interaction_points["controlled_word"]
            ),
            "hierarchical_paired_bootstrap_95": interval(
                learner_interaction_draws["controlled_word"]
            ),
            "lexical_seed_effects": simple_seed_effects["controlled_word"],
        },
        "learner_interactions": {
            model: {
                "estimand": (
                    "Qwen2.5-1.5B difference-in-differences minus "
                    f"{model} difference-in-differences"
                ),
                "qwen_minus_simple_difference_in_differences": (
                    learner_interaction_points[model]
                ),
                "hierarchical_paired_bootstrap_95": interval(
                    learner_interaction_draws[model]
                ),
                "simple_seed_effects": simple_seed_effects[model],
            }
            for model in interaction_models
        },
        "primary": {
            "estimand": (
                "(target naive-safe accuracy gap) minus "
                "(clean-validation naive-safe accuracy gap)"
            ),
            "difference_in_differences": did_point,
            "hierarchical_bootstrap_95": interval(did_draws),
            "paired_sign_randomization_two_sided_p": sign_randomization_p(
                did_point, did_randomized
            ),
            "replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "randomization_seed": args.randomization_seed,
        },
        "untouched_clean_consequence": {
            "naive_minus_safe_accuracy": untouched_point,
            "hierarchical_bootstrap_95": interval(untouched_draws),
            "paired_sign_randomization_two_sided_p": sign_randomization_p(
                untouched_point, untouched_randomized
            ),
            "interpretation": (
                "Independent clean performance difference at the fixed "
                "checkpoint; not used for checkpoint selection."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        did=did_draws,
        untouched=untouched_draws,
        learner_interaction=learner_interaction_draws["controlled_word"],
        learner_interaction_word=learner_interaction_draws["controlled_word"],
        learner_interaction_character=learner_interaction_draws[
            "controlled_character"
        ],
        did_sign_randomization=did_randomized,
        untouched_sign_randomization=untouched_randomized,
        **{
            f"accuracy_{boundary}_{evaluation_set}": draws
            for (boundary, evaluation_set), draws in accuracy_draws.items()
        },
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
