#!/usr/bin/env python3
"""Verify controlled results, scoring, evidence paths, and final figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


Z_95 = 1.959963984540054
SEEDS = (20260727, 20260728, 20260729)
LEARNERS = (
    "controlled_word",
    "controlled_character",
    "controlled_neural",
)
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".yml",
    ".yaml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{actual} != {expected}")


def strict_wilson(successes: int, total: int) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1.0 + Z_95**2 / total
    center = proportion + Z_95**2 / (2.0 * total)
    radius = Z_95 * math.sqrt(
        proportion * (1.0 - proportion) / total
        + Z_95**2 / (4.0 * total**2)
    )
    return (
        (center - radius) / denominator,
        (center + radius) / denominator,
    )


def verify_competence(root: Path) -> dict[str, Any]:
    path = (
        root
        / "results"
        / "controlled_qwen15_factorial"
        / "competence_predictions.csv"
    )
    frame = pd.read_csv(path)
    values = frame["correct"].astype(float)
    counts = {
        "pairs": int(len(values)),
        "correct": int((values == 1.0).sum()),
        "incorrect": int((values == 0.0).sum()),
        "exact_ties": int((values == 0.5).sum()),
    }
    expected = {
        "pairs": 400,
        "correct": 317,
        "incorrect": 82,
        "exact_ties": 1,
    }
    if counts != expected:
        raise AssertionError(f"competence counts differ: {counts}")
    tie_adjusted = (counts["correct"] + 0.5 * counts["exact_ties"]) / 400
    strict = counts["correct"] / 400
    low, high = strict_wilson(counts["correct"], 400)
    close(tie_adjusted, 0.79375)
    close(strict, 0.7925)
    close(low, 0.7500694660392275)
    close(high, 0.829365841775209)

    summary = read_json(
        root
        / "results"
        / "controlled_qwen15_factorial"
        / "competence.json"
    )
    for key, value in counts.items():
        if summary[key] != value:
            raise AssertionError(f"competence summary mismatch: {key}")
    close(summary["tie_adjusted_accuracy"], tie_adjusted)
    close(summary["strict_accuracy"], strict)
    close(summary["strict_wilson_95"][0], low)
    close(summary["strict_wilson_95"][1], high)

    macros = (
        root / "macros" / "controlled_neural_macros.tex"
    ).read_text(encoding="utf-8")
    required = (
        r"\newcommand{\ControlledNeuralCompetenceCorrect}{317\xspace}",
        r"\newcommand{\ControlledNeuralCompetenceIncorrect}{82\xspace}",
        r"\newcommand{\ControlledNeuralCompetenceTies}{1\xspace}",
        r"\newcommand{\ControlledNeuralCompetenceTieAdjustedAccuracy}{79.38\xspace}",
        r"\newcommand{\ControlledNeuralCompetenceStrictAccuracy}{79.25\xspace}",
        r"\newcommand{\ControlledNeuralCompetenceStrictWilsonLow}{75.01\xspace}",
        r"\newcommand{\ControlledNeuralCompetenceStrictWilsonHigh}{82.94\xspace}",
    )
    missing = [line for line in required if line not in macros]
    if missing:
        raise AssertionError(f"missing competence macros: {missing}")
    cross_macros = (
        root / "macros" / "controlled_cross_learner_macros.tex"
    ).read_text(encoding="utf-8")
    cross_missing = [line for line in required if line not in cross_macros]
    if cross_missing:
        raise AssertionError(
            f"missing self-contained competence macros: {cross_missing}"
        )
    if r"\input{" in cross_macros:
        raise AssertionError("cross-learner macro file is not self-contained")
    if r"\ControlledWordDid" not in cross_macros:
        raise AssertionError("cross-learner macro file omits sparse values")
    return {
        **counts,
        "tie_adjusted_accuracy": tie_adjusted,
        "strict_accuracy": strict,
        "strict_wilson_95": [low, high],
    }


def per_seed_effects(frame: pd.DataFrame, learner: str) -> list[dict[str, float]]:
    selected = frame[frame["learner"] == learner]
    rows = []
    for seed in SEEDS:
        seed_frame = selected[selected["seed"] == seed]
        means = seed_frame.groupby(
            ["boundary", "evaluation_set"]
        )["correct"].mean()
        target = means[("naive", "target")] - means[("safe", "target")]
        clean = (
            means[("naive", "clean_validation")]
            - means[("safe", "clean_validation")]
        )
        untouched = (
            means[("naive", "untouched_clean_test")]
            - means[("safe", "untouched_clean_test")]
        )
        rows.append(
            {
                "difference_in_differences": float(target - clean),
                "untouched_naive_minus_safe": float(untouched),
            }
        )
    return rows


def vector_hashes(frame: pd.DataFrame, learner: str) -> set[str]:
    selected = frame[frame["learner"] == learner]
    hashes = set()
    for seed in SEEDS:
        rows = selected[selected["seed"] == seed].sort_values(
            ["boundary", "evaluation_set", "pair_id"]
        )
        digest = hashlib.sha256()
        digest.update(
            np.asarray(2.0 * rows["correct"].to_numpy(), dtype=np.int8).tobytes()
        )
        hashes.add(digest.hexdigest())
    return hashes


def verify_factorials(root: Path) -> dict[str, Any]:
    sparse_path = (
        root / "results" / "controlled_sparse_factorial" / "predictions.csv.gz"
    )
    sparse = pd.read_csv(sparse_path)
    sparse = sparse[sparse["fraction"] == 1.0].rename(
        columns={"model": "learner"}
    )
    neural = pd.read_csv(
        root
        / "results"
        / "controlled_qwen15_factorial"
        / "predictions.csv.gz"
    )
    combined = pd.concat(
        [
            sparse[
                [
                    "learner",
                    "seed",
                    "boundary",
                    "evaluation_set",
                    "pair_id",
                    "correct",
                ]
            ],
            neural[
                [
                    "learner",
                    "seed",
                    "boundary",
                    "evaluation_set",
                    "pair_id",
                    "correct",
                ]
            ],
        ],
        ignore_index=True,
    )
    table = pd.read_csv(root / "tables" / "controlled_cross_learner_factorial.csv")
    untouched_table = pd.read_csv(
        root / "tables" / "untouched_naive_vs_safe.csv"
    )
    report: dict[str, Any] = {}
    expected_vectors = {
        "controlled_word": 1,
        "controlled_character": 1,
        "controlled_neural": 3,
    }
    for learner in LEARNERS:
        effects = per_seed_effects(combined, learner)
        did = float(np.mean([row["difference_in_differences"] for row in effects]))
        untouched = float(
            np.mean([row["untouched_naive_minus_safe"] for row in effects])
        )
        table_did = float(
            table.loc[table["learner"] == learner, "estimate"].iloc[0]
        )
        table_untouched = float(
            untouched_table.loc[
                untouched_table["learner"] == learner, "estimate"
            ].iloc[0]
        )
        close(did, table_did)
        close(untouched, table_untouched)
        vectors = len(vector_hashes(combined, learner))
        if vectors != expected_vectors[learner]:
            raise AssertionError(f"{learner}: {vectors} prediction vectors")
        report[learner] = {
            "did": did,
            "untouched": untouched,
            "unique_prediction_vectors": vectors,
        }

    expected = {
        "controlled_word": (0.15625, 0.01171875),
        "controlled_character": (0.0859375, 0.009765625),
        "controlled_neural": (
            -0.0022786458333333335,
            -0.012044270833333334,
        ),
    }
    for learner, values in expected.items():
        close(report[learner]["did"], values[0])
        close(report[learner]["untouched"], values[1])
    return report


def verify_registry(root: Path) -> int:
    registry = read_json(root / "evidence" / "evidence_registry.json")
    checked = 0
    for path, metadata in registry["files"].items():
        target = root / path
        if not target.is_file():
            raise FileNotFoundError(target)
        if target.stat().st_size != metadata["bytes"]:
            raise AssertionError(f"size mismatch: {path}")
        if sha256(target) != metadata["sha256"]:
            raise AssertionError(f"hash mismatch: {path}")
        checked += 1
    return checked


def scan_tree(root: Path) -> dict[str, Any]:
    absolute_paths: list[dict[str, Any]] = []
    home_prefix = str(Path.home()) + "/"
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if path.name == "verify_controlled_release.py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if home_prefix in text:
            absolute_paths.append({"path": relative.as_posix()})
    if absolute_paths:
        raise AssertionError(f"machine-specific paths: {absolute_paths}")
    return {
        "machine_specific_path_matches": 0,
    }


def verify_figures(root: Path) -> list[dict[str, Any]]:
    expected = (
        "learning_dynamics_main",
        "learning_dynamics_acquisition",
        "controlled_cross_learner_factorial_main",
        "candidate_selection_main",
        "candidate_selection_supp",
    )
    rows = []
    for name in expected:
        pdf = root / "figures" / f"{name}.pdf"
        png = root / "figures" / f"{name}.png"
        if not pdf.is_file() or not png.is_file():
            raise FileNotFoundError(name)
        if pdf.stat().st_size < 5_000 or png.stat().st_size < 20_000:
            raise AssertionError(f"figure appears incomplete: {name}")
        rows.append(
            {
                "figure": name,
                "pdf_sha256": sha256(pdf),
                "png_sha256": sha256(png),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (
        args.output.resolve()
        if args.output
        else root / "manifests" / "verification_results.json"
    )

    result = {
        "status": "pass",
        "competence": verify_competence(root),
        "factorials": verify_factorials(root),
        "registry_files_checked": verify_registry(root),
        "figures": verify_figures(root),
        "scans": scan_tree(root),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
