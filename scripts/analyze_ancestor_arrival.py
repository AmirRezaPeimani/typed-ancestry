#!/usr/bin/env python3
"""Analyze randomized target-ancestor arrival across frozen checkpoints."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ancestry_audit.canonical import normalize_text
from ancestry_audit.factorial import (
    context_hash,
    load_jsonl_gz,
    stable_rank,
)
from ancestry_audit.reward import preferred_response


SEEDS = (20260727, 20260728, 20260729)
BOUNDARIES = ("naive", "safe")
FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
ARRIVAL_FRACTIONS = FRACTIONS[:-1]
TRAINING_PAIRS = 4512
BOOTSTRAP_SEED = 20260901
RANDOMIZATION_SEED = 20260903


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_model_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("model must be LABEL=PATH")
    return label.strip(), Path(raw_path)


def prediction_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(path.glob("evaluations/*_full/predictions.csv.gz"))
    if not files:
        files = sorted(path.rglob("predictions.csv.gz"))
    if not files:
        raise FileNotFoundError(f"no predictions under {path}")
    return files


def build_donor_map(
    targets: list[dict[str, Any]],
    naive_train: list[dict[str, Any]],
    safe_train: list[dict[str, Any]],
) -> dict[str, str]:
    safe_ids = {row["pair_id"] for row in safe_train}
    naive_only = [
        row for row in naive_train if row["pair_id"] not in safe_ids
    ]
    if len(naive_only) != len(targets):
        raise ValueError(
            f"expected {len(targets)} naïve-only rows, found {len(naive_only)}"
        )
    by_context: dict[str, list[dict[str, Any]]] = {}
    for row in naive_only:
        by_context.setdefault(context_hash(row), []).append(row)

    result = {}
    donor_ids = set()
    for target in targets:
        preferred, _ = preferred_response(target)
        candidates = [
            row
            for row in by_context.get(context_hash(target), [])
            if normalize_text(preferred)
            in {
                normalize_text(row["response1"]),
                normalize_text(row["response2"]),
            }
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"{target['pair_id']} has {len(candidates)} exact donors"
            )
        donor_id = candidates[0]["pair_id"]
        if donor_id in donor_ids:
            raise ValueError("a donor maps to more than one target")
        donor_ids.add(donor_id)
        result[target["pair_id"]] = donor_id
    return result


def processed_rows_by_fraction() -> dict[float, int]:
    return {
        fraction: round(TRAINING_PAIRS * fraction)
        for fraction in FRACTIONS
    }


def donor_positions(
    data_dir: Path,
    targets: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, int]], dict[str, str], list[dict[str, Any]]]:
    first_naive = load_jsonl_gz(
        data_dir / f"train_naive_seed{SEEDS[0]}.jsonl.gz"
    )
    first_safe = load_jsonl_gz(
        data_dir / f"train_safe_seed{SEEDS[0]}.jsonl.gz"
    )
    target_to_donor = build_donor_map(targets, first_naive, first_safe)
    expected_donors = set(target_to_donor.values())
    positions: dict[int, dict[str, int]] = {}
    inputs = []
    for seed in SEEDS:
        path = data_dir / f"train_naive_seed{seed}.jsonl.gz"
        rows = load_jsonl_gz(path)
        index = {
            row["pair_id"]: position for position, row in enumerate(rows)
        }
        if not expected_donors.issubset(index):
            raise ValueError(f"seed {seed} lacks a frozen target donor")
        seed_positions = {
            target_id: index[donor_id]
            for target_id, donor_id in target_to_donor.items()
        }
        if len(set(seed_positions.values())) != len(targets):
            raise ValueError("donor positions are not one-to-one")
        positions[seed] = seed_positions
        inputs.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return positions, target_to_donor, inputs


def load_model_rows(
    model: str,
    path: Path,
) -> tuple[
    dict[tuple[int, str, float, str, str], tuple[int, int, float]],
    list[dict[str, Any]],
]:
    rows = {}
    inputs = []
    for file in prediction_files(path):
        inputs.append(
            {
                "model": model,
                "path": str(file.resolve()),
                "bytes": file.stat().st_size,
                "sha256": file_sha256(file),
            }
        )
        with gzip.open(file, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "seed",
                "boundary",
                "fraction",
                "evaluation_set",
                "pair_id",
                "label_response2_preferred",
                "margin_response2_minus_response1",
                "correct",
            }
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{file} lacks required prediction fields")
            for row in reader:
                if row["evaluation_set"] not in {
                    "target",
                    "clean_validation",
                }:
                    continue
                key = (
                    int(row["seed"]),
                    row["boundary"],
                    float(row["fraction"]),
                    row["evaluation_set"],
                    row["pair_id"],
                )
                value = (
                    int(row["correct"]),
                    int(row["label_response2_preferred"]),
                    float(row["margin_response2_minus_response1"]),
                )
                if key in rows:
                    raise ValueError(f"duplicate prediction key: {model}/{key}")
                rows[key] = value
    return rows, inputs


def build_prediction_arrays(
    model_rows: dict[
        str,
        dict[tuple[int, str, float, str, str], tuple[int, int, float]],
    ],
    target_ids: list[str],
    clean_ids: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    arrays = {}
    for model, rows in model_rows.items():
        arrays[model] = {}
        for evaluation_set, ids in (
            ("target", target_ids),
            ("clean_validation", clean_ids),
        ):
            correct = np.empty(
                (
                    len(SEEDS),
                    len(BOUNDARIES),
                    len(FRACTIONS),
                    len(ids),
                ),
                dtype=np.float64,
            )
            signed_margin = np.empty_like(correct)
            for seed_index, seed in enumerate(SEEDS):
                for boundary_index, boundary in enumerate(BOUNDARIES):
                    for fraction_index, fraction in enumerate(FRACTIONS):
                        for item_index, pair_id in enumerate(ids):
                            key = (
                                seed,
                                boundary,
                                fraction,
                                evaluation_set,
                                pair_id,
                            )
                            if key not in rows:
                                raise ValueError(
                                    f"incomplete grid for {model}/{key}"
                                )
                            item_correct, label, margin = rows[key]
                            correct[
                                seed_index,
                                boundary_index,
                                fraction_index,
                                item_index,
                            ] = item_correct
                            signed_margin[
                                seed_index,
                                boundary_index,
                                fraction_index,
                                item_index,
                            ] = (1 if label else -1) * margin
            expected = (
                len(SEEDS)
                * len(BOUNDARIES)
                * len(FRACTIONS)
                * (len(target_ids) + len(clean_ids))
            )
            if len(rows) != expected:
                raise ValueError(
                    f"{model} contains rows outside the arrival-analysis grid"
                )
            arrays[model][evaluation_set] = {
                "correct": correct,
                "signed_margin": signed_margin,
            }
    return arrays


def arrival_masks(
    item_ids: list[str],
    positions: dict[int, dict[str, int]],
    processed: dict[float, int],
) -> tuple[np.ndarray, dict[int, dict[str, int]]]:
    mask = np.empty(
        (len(SEEDS), len(ARRIVAL_FRACTIONS), len(item_ids)),
        dtype=bool,
    )
    counts = {}
    for seed_index, seed in enumerate(SEEDS):
        counts[seed] = {}
        for fraction_index, fraction in enumerate(ARRIVAL_FRACTIONS):
            mask[seed_index, fraction_index] = [
                positions[seed][item_id] < processed[fraction]
                for item_id in item_ids
            ]
            count = int(mask[seed_index, fraction_index].sum())
            if not 0 < count < len(item_ids):
                raise ValueError("arrival checkpoint lacks a comparison arm")
            counts[seed][str(fraction)] = count
    return mask, counts


def clean_placebo_positions(
    clean_ids: list[str],
    target_positions: dict[int, dict[str, int]],
) -> dict[int, dict[str, int]]:
    placebo = {}
    for seed in SEEDS:
        ordered_clean = sorted(
            clean_ids,
            key=lambda pair_id: stable_rank(
                seed, "clean_placebo_arrival", pair_id
            ),
        )
        ordered_positions = sorted(target_positions[seed].values())
        placebo[seed] = dict(
            zip(ordered_clean, ordered_positions, strict=True)
        )
    return placebo


def observed_contrast(
    delta: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return model-level and seed-level seen-minus-not-yet contrasts."""

    if delta.ndim != 4 or mask.ndim != 3:
        raise ValueError("arrival inputs have invalid ranks")
    if delta.shape[1:] != mask.shape:
        raise ValueError("arrival arrays do not align")
    treated = np.where(mask[None], delta, np.nan)
    untreated = np.where(~mask[None], delta, np.nan)
    by_seed = np.nanmean(treated, axis=-1) - np.nanmean(
        untreated, axis=-1
    )
    return by_seed.mean(axis=1), by_seed


def hierarchical_bootstrap(
    target_delta: np.ndarray,
    target_mask: np.ndarray,
    clean_delta: np.ndarray,
    clean_mask: np.ndarray,
    *,
    replicates: int,
    seed: int,
    batch_size: int = 100,
) -> dict[str, np.ndarray]:
    """Jointly bootstrap target, clean placebo, and their difference."""

    model_count, seed_count, checkpoint_count, target_n = target_delta.shape
    clean_n = clean_delta.shape[-1]
    outputs = {
        name: np.empty(
            (replicates, model_count, checkpoint_count), dtype=np.float64
        )
        for name in ("target", "clean_placebo", "target_minus_placebo")
    }
    rng = np.random.default_rng(seed)
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
        seed_weights = np.stack(
            [
                (seed_draws == index).sum(axis=1)
                for index in range(seed_count)
            ],
            axis=1,
        )

        def sample_contrast(
            values: np.ndarray,
            masks: np.ndarray,
            indices: np.ndarray,
        ) -> np.ndarray:
            sampled_values = values[..., indices].transpose(
                3, 0, 1, 2, 4
            )
            sampled_masks = masks[..., indices].transpose(2, 0, 1, 3)
            treated_n = sampled_masks.sum(axis=-1)
            untreated_n = (~sampled_masks).sum(axis=-1)
            if not np.all(treated_n) or not np.all(untreated_n):
                raise RuntimeError("bootstrap produced an empty arrival arm")
            treated_mean = (
                sampled_values * sampled_masks[:, None]
            ).sum(axis=-1) / treated_n[:, None]
            untreated_mean = (
                sampled_values * (~sampled_masks[:, None])
            ).sum(axis=-1) / untreated_n[:, None]
            return treated_mean - untreated_mean

        target = sample_contrast(
            target_delta, target_mask, target_indices
        )
        clean = sample_contrast(clean_delta, clean_mask, clean_indices)
        for name, values in (
            ("target", target),
            ("clean_placebo", clean),
            ("target_minus_placebo", target - clean),
        ):
            outputs[name][start:stop] = (
                values * seed_weights[:, None, :, None]
            ).sum(axis=2) / seed_count
    return outputs


def randomized_average_contrast(
    delta: np.ndarray,
    positions: np.ndarray,
    thresholds: np.ndarray,
    *,
    replicates: int,
    seed: int,
    batch_size: int = 200,
) -> np.ndarray:
    """Permute complete target-to-position assignments within each seed."""

    model_count, seed_count, checkpoint_count, item_count = delta.shape
    result = np.empty((replicates, model_count), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, replicates, batch_size):
        stop = min(replicates, start + batch_size)
        count = stop - start
        permuted = np.empty(
            (count, seed_count, item_count), dtype=np.int64
        )
        for replicate in range(count):
            for seed_index in range(seed_count):
                permuted[replicate, seed_index] = rng.permutation(
                    positions[seed_index]
                )
        masks = (
            permuted[:, :, None, :]
            < thresholds[None, None, :, None]
        )
        treated_n = masks.sum(axis=-1)
        untreated_n = (~masks).sum(axis=-1)
        values = delta[None]
        treated_mean = (
            values * masks[:, None]
        ).sum(axis=-1) / treated_n[:, None]
        untreated_mean = (
            values * (~masks[:, None])
        ).sum(axis=-1) / untreated_n[:, None]
        contrasts = treated_mean - untreated_mean
        result[start:stop] = contrasts.mean(axis=(2, 3))
    return result


def interval(values: np.ndarray) -> list[float]:
    return [
        float(value)
        for value in np.quantile(values, [0.025, 0.975])
    ]


def randomization_p(draws: np.ndarray, observed: float) -> float:
    return float(
        (1 + np.sum(np.abs(draws) >= abs(observed)))
        / (len(draws) + 1)
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=parse_model_spec, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--randomization-replicates", type=int, default=20_000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.bootstrap_replicates < 100:
        raise ValueError("too few bootstrap replicates")
    if args.randomization_replicates < 100:
        raise ValueError("too few randomization replicates")
    model_paths = dict(args.model)
    if len(model_paths) != len(args.model):
        raise ValueError("model labels must be unique")

    targets = load_jsonl_gz(args.data_dir / "target_512.jsonl.gz")
    clean = load_jsonl_gz(args.data_dir / "clean_validation_512.jsonl.gz")
    target_ids = [row["pair_id"] for row in targets]
    clean_ids = [row["pair_id"] for row in clean]
    if len(set(target_ids)) != 512 or len(set(clean_ids)) != 512:
        raise ValueError("evaluation identifiers are not unique")
    positions, target_to_donor, training_inputs = donor_positions(
        args.data_dir, targets
    )
    processed = processed_rows_by_fraction()
    target_mask, arrival_counts = arrival_masks(
        target_ids, positions, processed
    )
    placebo_positions = clean_placebo_positions(clean_ids, positions)
    clean_mask, placebo_counts = arrival_masks(
        clean_ids, placebo_positions, processed
    )
    if arrival_counts != placebo_counts:
        raise AssertionError("clean placebo does not preserve arrival counts")

    model_rows = {}
    prediction_inputs = []
    for model, path in model_paths.items():
        model_rows[model], inputs = load_model_rows(model, path)
        prediction_inputs.extend(inputs)
    arrays = build_prediction_arrays(model_rows, target_ids, clean_ids)
    model_names = list(model_paths)

    target_correct = np.stack(
        [
            arrays[model]["target"]["correct"][:, 0, :-1]
            - arrays[model]["target"]["correct"][:, 1, :-1]
            for model in model_names
        ]
    )
    clean_correct = np.stack(
        [
            arrays[model]["clean_validation"]["correct"][:, 0, :-1]
            - arrays[model]["clean_validation"]["correct"][:, 1, :-1]
            for model in model_names
        ]
    )
    target_margin = np.stack(
        [
            arrays[model]["target"]["signed_margin"][:, 0, :-1]
            - arrays[model]["target"]["signed_margin"][:, 1, :-1]
            for model in model_names
        ]
    )
    clean_margin = np.stack(
        [
            arrays[model]["clean_validation"]["signed_margin"][:, 0, :-1]
            - arrays[model]["clean_validation"]["signed_margin"][:, 1, :-1]
            for model in model_names
        ]
    )

    observed_correct, by_seed_correct = observed_contrast(
        target_correct, target_mask
    )
    observed_clean_correct, by_seed_clean_correct = observed_contrast(
        clean_correct, clean_mask
    )
    observed_margin, by_seed_margin = observed_contrast(
        target_margin, target_mask
    )
    observed_clean_margin, by_seed_clean_margin = observed_contrast(
        clean_margin, clean_mask
    )
    correct_bootstrap = hierarchical_bootstrap(
        target_correct,
        target_mask,
        clean_correct,
        clean_mask,
        replicates=args.bootstrap_replicates,
        seed=BOOTSTRAP_SEED,
    )
    margin_bootstrap = hierarchical_bootstrap(
        target_margin,
        target_mask,
        clean_margin,
        clean_mask,
        replicates=args.bootstrap_replicates,
        seed=BOOTSTRAP_SEED,
    )
    position_array = np.asarray(
        [
            [positions[seed][pair_id] for pair_id in target_ids]
            for seed in SEEDS
        ],
        dtype=np.int64,
    )
    thresholds = np.asarray(
        [processed[fraction] for fraction in ARRIVAL_FRACTIONS],
        dtype=np.int64,
    )
    randomized_correct = randomized_average_contrast(
        target_correct,
        position_array,
        thresholds,
        replicates=args.randomization_replicates,
        seed=RANDOMIZATION_SEED,
    )
    randomized_margin = randomized_average_contrast(
        target_margin,
        position_array,
        thresholds,
        replicates=args.randomization_replicates,
        seed=RANDOMIZATION_SEED,
    )

    checkpoint_rows = []
    summary_models = {}
    for model_index, model in enumerate(model_names):
        for fraction_index, fraction in enumerate(ARRIVAL_FRACTIONS):
            checkpoint_rows.append(
                {
                    "model": model,
                    "fraction": fraction,
                    "target_correct_arrival_contrast": float(
                        observed_correct[model_index, fraction_index]
                    ),
                    "target_correct_ci_low": interval(
                        correct_bootstrap["target"][
                            :, model_index, fraction_index
                        ]
                    )[0],
                    "target_correct_ci_high": interval(
                        correct_bootstrap["target"][
                            :, model_index, fraction_index
                        ]
                    )[1],
                    "clean_placebo_correct_arrival_contrast": float(
                        observed_clean_correct[
                            model_index, fraction_index
                        ]
                    ),
                    "clean_placebo_correct_ci_low": interval(
                        correct_bootstrap["clean_placebo"][
                            :, model_index, fraction_index
                        ]
                    )[0],
                    "clean_placebo_correct_ci_high": interval(
                        correct_bootstrap["clean_placebo"][
                            :, model_index, fraction_index
                        ]
                    )[1],
                    "target_minus_placebo_correct_contrast": float(
                        observed_correct[model_index, fraction_index]
                        - observed_clean_correct[
                            model_index, fraction_index
                        ]
                    ),
                    "target_minus_placebo_correct_ci_low": interval(
                        correct_bootstrap["target_minus_placebo"][
                            :, model_index, fraction_index
                        ]
                    )[0],
                    "target_minus_placebo_correct_ci_high": interval(
                        correct_bootstrap["target_minus_placebo"][
                            :, model_index, fraction_index
                        ]
                    )[1],
                    "target_margin_arrival_contrast": float(
                        observed_margin[model_index, fraction_index]
                    ),
                    "target_margin_ci_low": interval(
                        margin_bootstrap["target"][
                            :, model_index, fraction_index
                        ]
                    )[0],
                    "target_margin_ci_high": interval(
                        margin_bootstrap["target"][
                            :, model_index, fraction_index
                        ]
                    )[1],
                    "clean_placebo_margin_arrival_contrast": float(
                        observed_clean_margin[
                            model_index, fraction_index
                        ]
                    ),
                    "clean_placebo_margin_ci_low": interval(
                        margin_bootstrap["clean_placebo"][
                            :, model_index, fraction_index
                        ]
                    )[0],
                    "clean_placebo_margin_ci_high": interval(
                        margin_bootstrap["clean_placebo"][
                            :, model_index, fraction_index
                        ]
                    )[1],
                    "seen_counts_by_seed": json.dumps(
                        [
                            arrival_counts[seed][str(fraction)]
                            for seed in SEEDS
                        ]
                    ),
                }
            )
        target_average = float(observed_correct[model_index].mean())
        clean_average = float(
            observed_clean_correct[model_index].mean()
        )
        margin_average = float(observed_margin[model_index].mean())
        clean_margin_average = float(
            observed_clean_margin[model_index].mean()
        )
        target_boot_average = correct_bootstrap["target"][
            :, model_index
        ].mean(axis=1)
        clean_boot_average = correct_bootstrap["clean_placebo"][
            :, model_index
        ].mean(axis=1)
        difference_boot_average = correct_bootstrap[
            "target_minus_placebo"
        ][:, model_index].mean(axis=1)
        margin_boot_average = margin_bootstrap["target"][
            :, model_index
        ].mean(axis=1)
        summary_models[model] = {
            "target_correct_mean_arrival_contrast": target_average,
            "target_correct_mean_arrival_contrast_95ci": interval(
                target_boot_average
            ),
            "target_correct_randomization_two_sided_p": randomization_p(
                randomized_correct[:, model_index], target_average
            ),
            "clean_placebo_correct_mean_arrival_contrast": clean_average,
            "clean_placebo_correct_mean_arrival_contrast_95ci": interval(
                clean_boot_average
            ),
            "target_minus_placebo_correct_mean_contrast": (
                target_average - clean_average
            ),
            "target_minus_placebo_correct_mean_contrast_95ci": interval(
                difference_boot_average
            ),
            "target_margin_mean_arrival_contrast": margin_average,
            "target_margin_mean_arrival_contrast_95ci": interval(
                margin_boot_average
            ),
            "target_margin_randomization_two_sided_p": randomization_p(
                randomized_margin[:, model_index], margin_average
            ),
            "clean_placebo_margin_mean_arrival_contrast": (
                clean_margin_average
            ),
            "target_correct_by_seed_and_checkpoint": (
                by_seed_correct[model_index].tolist()
            ),
            "clean_placebo_correct_by_seed_and_checkpoint": (
                by_seed_clean_correct[model_index].tolist()
            ),
            "target_margin_by_seed_and_checkpoint": (
                by_seed_margin[model_index].tolist()
            ),
            "clean_placebo_margin_by_seed_and_checkpoint": (
                by_seed_clean_margin[model_index].tolist()
            ),
        }

    item_rows = []
    for model_index, model in enumerate(model_names):
        for seed_index, seed in enumerate(SEEDS):
            for fraction_index, fraction in enumerate(ARRIVAL_FRACTIONS):
                for item_index, pair_id in enumerate(target_ids):
                    item_rows.append(
                        {
                            "model": model,
                            "seed": seed,
                            "fraction": fraction,
                            "pair_id": pair_id,
                            "donor_pair_id": target_to_donor[pair_id],
                            "donor_position_zero_based": positions[seed][
                                pair_id
                            ],
                            "ancestor_seen": int(
                                target_mask[
                                    seed_index,
                                    fraction_index,
                                    item_index,
                                ]
                            ),
                            "naive_minus_safe_correct": target_correct[
                                model_index,
                                seed_index,
                                fraction_index,
                                item_index,
                            ],
                            "naive_minus_safe_signed_margin": target_margin[
                                model_index,
                                seed_index,
                                fraction_index,
                                item_index,
                            ],
                        }
                    )

    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "arrival_contrasts.csv", checkpoint_rows)
    with gzip.open(
        args.output_dir / "arrival_item_effects.csv.gz",
        "wt",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(item_rows[0]))
        writer.writeheader()
        writer.writerows(item_rows)
    np.savez_compressed(
        args.output_dir / "bootstrap_draws.npz",
        target_correct=correct_bootstrap["target"],
        clean_placebo_correct=correct_bootstrap["clean_placebo"],
        target_minus_placebo_correct=correct_bootstrap[
            "target_minus_placebo"
        ],
        target_margin=margin_bootstrap["target"],
    )
    np.savez_compressed(
        args.output_dir / "randomization_draws.npz",
        target_correct=randomized_correct,
        target_margin=randomized_margin,
    )
    summary = {
        "status": "complete",
        "models": model_names,
        "seeds": list(SEEDS),
        "arrival_fractions": list(ARRIVAL_FRACTIONS),
        "processed_rows_by_fraction": {
            str(key): value for key, value in processed.items()
        },
        "arrival_counts": arrival_counts,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": args.bootstrap_replicates,
        "randomization_seed": RANDOMIZATION_SEED,
        "randomization_replicates": args.randomization_replicates,
        "model_summaries": summary_models,
        "evaluation_inputs": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in (
                args.data_dir / "target_512.jsonl.gz",
                args.data_dir / "clean_validation_512.jsonl.gz",
                args.data_dir / "manifest.json",
            )
        ],
        "training_inputs": training_inputs,
        "prediction_inputs": prediction_inputs,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
