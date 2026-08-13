#!/usr/bin/env python3
"""Load and verify evidence used by the cross-learner figure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parents[1]
RELEASE_ROOT = REVISION_ROOT
RESULTS_ROOT = RELEASE_ROOT / "results"
TABLES_ROOT = RELEASE_ROOT / "tables"


INK = "#20262E"
MID = "#6F7782"
LIGHT = "#D8DDE4"
PALE = "#F2F4F7"
TARGET = "#20262E"
CLEAN = "#8A929C"
QWEN_SEED = "#A7ADB5"
BLUE = "#2368B8"
OCHRE = "#B97914"
TEAL = "#287D78"
WHITE = "#FFFFFF"

LEARNERS = ["controlled_word", "controlled_character", "controlled_neural"]
DISPLAY = {
    "controlled_word": "Word TF–IDF",
    "controlled_character": "Character TF–IDF",
    "controlled_neural": "Qwen2.5-1.5B",
}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pct(value: float, digits: int = 1, signed: bool = False) -> str:
    number = 100.0 * value
    if signed:
        raw = f"{number:+.{digits}f}"
    else:
        raw = f"{number:.{digits}f}"
    return raw.replace("-", "−")


def interval_text(row: pd.Series, digits: int = 2) -> str:
    estimate = pct(float(row["estimate"]), digits, signed=True)
    low = pct(float(row["ci_low"]), digits).replace("-", "−")
    high = pct(float(row["ci_high"]), digits).replace("-", "−")
    return f"{estimate} [{low}, {high}]"


def load_data() -> dict[str, pd.DataFrame]:
    sparse_path = RESULTS_ROOT / "controlled_sparse_factorial" / "predictions.csv.gz"
    qwen_path = RESULTS_ROOT / "controlled_qwen15_factorial" / "predictions.csv.gz"
    factorial_path = RESULTS_ROOT / "controlled_cross_learner_factorial" / "factorial_estimates.csv"
    untouched_path = RESULTS_ROOT / "controlled_cross_learner_factorial" / "untouched_estimates.csv"
    interaction_path = RESULTS_ROOT / "controlled_cross_learner_factorial" / "interaction_estimates.csv"
    competence_path = TABLES_ROOT / "controlled_neural_competence.csv"

    sparse = pd.read_csv(sparse_path)
    sparse = sparse.loc[np.isclose(sparse["fraction"], 1.0)].copy()
    sparse = sparse.rename(columns={"model": "learner"})
    qwen = pd.read_csv(qwen_path)

    qwen_seed_cells = (
        qwen.groupby(["learner", "seed", "boundary", "evaluation_set"], as_index=False)
        .agg(accuracy=("correct", "mean"), n=("correct", "size"))
    )
    sparse_seed_cells = (
        sparse.groupby(["learner", "seed", "boundary", "evaluation_set"], as_index=False)
        .agg(accuracy=("correct", "mean"), n=("correct", "size"))
    )

    for learner in ("controlled_word", "controlled_character"):
        probe = sparse_seed_cells.loc[sparse_seed_cells["learner"].eq(learner)]
        unique_by_cell = probe.groupby(["boundary", "evaluation_set"])["accuracy"].nunique()
        if not unique_by_cell.eq(1).all():
            raise ValueError(f"{learner} does not collapse to one final prediction vector")

    aggregate_cells = pd.concat(
        [
            sparse_seed_cells.groupby(
                ["learner", "boundary", "evaluation_set"], as_index=False
            ).agg(accuracy=("accuracy", "mean"), n=("n", "first")),
            qwen_seed_cells.groupby(
                ["learner", "boundary", "evaluation_set"], as_index=False
            ).agg(accuracy=("accuracy", "mean"), n=("n", "first")),
        ],
        ignore_index=True,
    )
    aggregate_cells["display_name"] = aggregate_cells["learner"].map(DISPLAY)
    aggregate_cells["level"] = "aggregate"
    aggregate_cells["seed"] = pd.NA
    qwen_seed_cells["display_name"] = qwen_seed_cells["learner"].map(DISPLAY)
    qwen_seed_cells["level"] = "seed"

    factorial = pd.read_csv(factorial_path)
    untouched = pd.read_csv(untouched_path)
    interactions = pd.read_csv(interaction_path)
    competence = pd.read_csv(competence_path)

    # Recompute every displayed aggregate contrast from the released predictions.
    wide = aggregate_cells.pivot_table(
        index="learner",
        columns=["evaluation_set", "boundary"],
        values="accuracy",
    )
    for learner in LEARNERS:
        did = (
            wide.loc[learner, ("target", "naive")]
            - wide.loc[learner, ("target", "safe")]
            - wide.loc[learner, ("clean_validation", "naive")]
            + wide.loc[learner, ("clean_validation", "safe")]
        )
        released = float(factorial.set_index("learner").loc[learner, "estimate"])
        if not np.isclose(did, released, atol=1e-12):
            raise ValueError(f"raw-cell DiD mismatch for {learner}: {did} vs {released}")
        untouched_effect = (
            wide.loc[learner, ("untouched_clean_test", "naive")]
            - wide.loc[learner, ("untouched_clean_test", "safe")]
        )
        released_untouched = float(
            untouched.set_index("learner").loc[learner, "estimate"]
        )
        if not np.isclose(untouched_effect, released_untouched, atol=1e-12):
            raise ValueError(f"untouched-effect mismatch for {learner}")

    comp = competence.iloc[0]
    if not (
        int(comp["pairs"]) == 400
        and int(comp["correct"]) == 317
        and int(comp["incorrect"]) == 82
        and int(comp["exact_ties"]) == 1
        and np.isclose(float(comp["tie_adjusted_accuracy"]), 0.79375)
        and np.isclose(float(comp["strict_accuracy"]), 0.7925)
    ):
        raise ValueError("competence table does not match the frozen 317/82/1 result")

    return {
        "aggregate_cells": aggregate_cells,
        "qwen_seed_cells": qwen_seed_cells,
        "factorial": factorial,
        "untouched": untouched,
        "interactions": interactions,
        "competence": competence,
        "sources": pd.DataFrame(
            {
                "path": [
                    sparse_path.relative_to(REVISION_ROOT),
                    qwen_path.relative_to(REVISION_ROOT),
                    factorial_path.relative_to(REVISION_ROOT),
                    untouched_path.relative_to(REVISION_ROOT),
                    interaction_path.relative_to(REVISION_ROOT),
                    competence_path.relative_to(REVISION_ROOT),
                ],
                "sha256": [
                    sha256(sparse_path),
                    sha256(qwen_path),
                    sha256(factorial_path),
                    sha256(untouched_path),
                    sha256(interaction_path),
                    sha256(competence_path),
                ],
            }
        ),
    }


def cell_value(cells: pd.DataFrame, learner: str, evaluation_set: str, boundary: str) -> float:
    row = cells.loc[
        cells["learner"].eq(learner)
        & cells["evaluation_set"].eq(evaluation_set)
        & cells["boundary"].eq(boundary),
        "accuracy",
    ]
    if len(row) != 1:
        raise ValueError((learner, evaluation_set, boundary, len(row)))
    return float(row.iloc[0])
