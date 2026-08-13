#!/usr/bin/env python3
"""Analyze controlled sparse baselines with context-level uncertainty."""

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
MODELS = ("controlled_word", "controlled_character")


def read_predictions(
    path: Path,
) -> dict[tuple[str, int, str, str], dict[str, float]]:
    selected: dict[tuple[str, int, str, str], dict[str, float]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if float(row["fraction"]) != 1.0:
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
                raise ValueError(f"duplicate prediction: {key}/{pair_id}")
            values[pair_id] = float(row["correct"])
    expected = {
        (model, seed, boundary, evaluation_set)
        for model in MODELS
        for seed in SEEDS
        for boundary in BOUNDARIES
        for evaluation_set in SETS
    }
    if set(selected) != expected:
        raise ValueError("final-checkpoint prediction grid is incomplete")
    if any(len(values) != 512 for values in selected.values()):
        raise ValueError("every prediction cell must contain 512 pairs")
    return selected


def array(values: dict[str, float]) -> np.ndarray:
    return np.asarray([values[key] for key in sorted(values)], dtype=float)


def paired_gap(
    predictions: dict[tuple[str, int, str, str], dict[str, float]],
    model: str,
    seed: int,
    evaluation_set: str,
    indices: np.ndarray | None = None,
) -> float:
    left = predictions[(model, seed, "naive", evaluation_set)]
    right = predictions[(model, seed, "safe", evaluation_set)]
    if set(left) != set(right):
        raise ValueError("naive and safe pair identifiers differ")
    differences = array(left) - array(right)
    if indices is not None:
        differences = differences[indices]
    return float(np.mean(differences))


def accuracy(
    predictions: dict[tuple[str, int, str, str], dict[str, float]],
    key: tuple[str, int, str, str],
    indices: np.ndarray | None = None,
) -> float:
    values = array(predictions[key])
    if indices is not None:
        values = values[indices]
    return float(np.mean(values))


def interval(draws: np.ndarray) -> list[float]:
    return [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def sign_randomization_p(observed: float, draws: np.ndarray) -> float:
    return float(
        (1 + np.sum(np.abs(draws) >= abs(observed))) / (len(draws) + 1)
    )


def unique_seed_representatives(
    predictions: dict[tuple[str, int, str, str], dict[str, float]],
    model: str,
) -> tuple[list[int], dict[int, list[int]]]:
    groups: dict[str, list[int]] = {}
    for seed in SEEDS:
        digest = hashlib.sha256()
        for boundary in BOUNDARIES:
            for evaluation_set in SETS:
                values = predictions[
                    (model, seed, boundary, evaluation_set)
                ]
                digest.update(
                    np.asarray(
                        [values[key] for key in sorted(values)],
                        dtype=np.int8,
                    ).tobytes()
                )
        groups.setdefault(digest.hexdigest(), []).append(seed)
    representatives = [values[0] for values in groups.values()]
    return representatives, {
        values[0]: values for values in groups.values()
    }


def analyze_model(
    predictions: dict[tuple[str, int, str, str], dict[str, float]],
    model: str,
    *,
    replicates: int,
    bootstrap_seed: int,
    randomization_seed: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    representative_seeds, seed_groups = unique_seed_representatives(
        predictions, model
    )
    seed_effects = []
    accuracy_rows = []
    for seed in SEEDS:
        for boundary in BOUNDARIES:
            for evaluation_set in SETS:
                accuracy_rows.append(
                    {
                        "seed": seed,
                        "boundary": boundary,
                        "evaluation_set": evaluation_set,
                        "accuracy": accuracy(
                            predictions,
                            (model, seed, boundary, evaluation_set),
                        ),
                    }
                )
        target_gap = paired_gap(predictions, model, seed, "target")
        clean_gap = paired_gap(
            predictions, model, seed, "clean_validation"
        )
        untouched_gap = paired_gap(
            predictions, model, seed, "untouched_clean_test"
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

    rng = np.random.default_rng(bootstrap_seed)
    did_draws = np.empty(replicates)
    untouched_draws = np.empty(replicates)
    accuracy_draws = {
        (boundary, evaluation_set): np.empty(replicates)
        for boundary in BOUNDARIES
        for evaluation_set in SETS
    }
    for replicate in range(replicates):
        sampled_seeds = rng.integers(
            0,
            len(representative_seeds),
            len(representative_seeds),
        )
        did_values = []
        untouched_values = []
        cell_values = {key: [] for key in accuracy_draws}
        for seed_index in sampled_seeds:
            seed = representative_seeds[int(seed_index)]
            indices = {
                evaluation_set: rng.integers(0, 512, 512)
                for evaluation_set in SETS
            }
            target_gap = paired_gap(
                predictions, model, seed, "target", indices["target"]
            )
            clean_gap = paired_gap(
                predictions,
                model,
                seed,
                "clean_validation",
                indices["clean_validation"],
            )
            did_values.append(target_gap - clean_gap)
            untouched_values.append(
                paired_gap(
                    predictions,
                    model,
                    seed,
                    "untouched_clean_test",
                    indices["untouched_clean_test"],
                )
            )
            for boundary in BOUNDARIES:
                for evaluation_set in SETS:
                    cell_values[(boundary, evaluation_set)].append(
                        accuracy(
                            predictions,
                            (model, seed, boundary, evaluation_set),
                            indices[evaluation_set],
                        )
                    )
        did_draws[replicate] = float(np.mean(did_values))
        untouched_draws[replicate] = float(np.mean(untouched_values))
        for key, values in cell_values.items():
            accuracy_draws[key][replicate] = float(np.mean(values))

    context_differences = {}
    for evaluation_set in SETS:
        rows = []
        for seed in representative_seeds:
            left = predictions[(model, seed, "naive", evaluation_set)]
            right = predictions[(model, seed, "safe", evaluation_set)]
            rows.append(array(left) - array(right))
        context_differences[evaluation_set] = np.mean(
            np.stack(rows), axis=0
        )
    randomization_rng = np.random.default_rng(randomization_seed)
    randomized = {}
    for evaluation_set, differences in context_differences.items():
        signs = (
            2
            * randomization_rng.integers(
                0,
                2,
                size=(replicates, len(differences)),
                dtype=np.int8,
            )
            - 1
        )
        randomized[evaluation_set] = signs @ differences / len(differences)
    did_randomized = (
        randomized["target"] - randomized["clean_validation"]
    )
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
    payload = {
        "nominal_seeds": list(SEEDS),
        "unique_prediction_vectors": len(representative_seeds),
        "prediction_vector_seed_groups": {
            str(representative): group
            for representative, group in seed_groups.items()
        },
        "uncertainty_unit": (
            "unique final prediction vector and paired evaluation context"
        ),
        "accuracy_rows": accuracy_rows,
        "accuracy_summary": [
            {
                "boundary": boundary,
                "evaluation_set": evaluation_set,
                "accuracy": float(
                    np.mean(
                        [
                            row["accuracy"]
                            for row in accuracy_rows
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
        "primary": {
            "estimand": (
                "(target naive-safe accuracy gap) minus "
                "(clean-validation naive-safe accuracy gap)"
            ),
            "difference_in_differences": did_point,
            "hierarchical_bootstrap_95": interval(did_draws),
            "paired_sign_randomization_two_sided_p": (
                sign_randomization_p(did_point, did_randomized)
            ),
            "replicates": replicates,
            "bootstrap_seed": bootstrap_seed,
            "randomization_seed": randomization_seed,
        },
        "untouched_clean_consequence": {
            "naive_minus_safe_accuracy": untouched_point,
            "hierarchical_bootstrap_95": interval(untouched_draws),
            "paired_sign_randomization_two_sided_p": sign_randomization_p(
                untouched_point, randomized["untouched_clean_test"]
            ),
        },
    }
    arrays = {
        "did": did_draws,
        "untouched": untouched_draws,
        "did_sign_randomization": did_randomized,
        "untouched_sign_randomization": randomized[
            "untouched_clean_test"
        ],
        **{
            f"accuracy_{boundary}_{evaluation_set}": draws
            for (boundary, evaluation_set), draws in accuracy_draws.items()
        },
    }
    return payload, arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261021)
    parser.add_argument("--randomization-seed", type=int, default=20261022)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("refusing to overwrite baseline analysis")

    predictions = read_predictions(args.predictions)
    payload = {
        "status": "complete",
        "protocol": "docs/controlled_factorial_protocol.md",
        "models": {},
    }
    arrays = {}
    for model_index, model in enumerate(MODELS):
        result, model_arrays = analyze_model(
            predictions,
            model,
            replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed + model_index,
            randomization_seed=args.randomization_seed + model_index,
        )
        payload["models"][model] = result
        arrays.update(
            {f"{model}_{name}": values for name, values in model_arrays.items()}
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
