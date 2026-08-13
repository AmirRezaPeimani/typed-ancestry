#!/usr/bin/env python3
"""Recompute evidence used by the candidate-selection figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parents[1]
RELEASE_ROOT = REVISION_ROOT
RESULTS_ROOT = RELEASE_ROOT / "results" / "model_selection"

BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260801
WEIGHTS = np.linspace(0.0, 1.0, 101)

INK = "#20262E"
MID = "#737C88"
NEUTRAL = "#B9C0C8"
LIGHT = "#DCE1E7"
PALE = "#F2F4F7"
WHITE = "#FFFFFF"
BLUE = "#2368B8"
OCHRE = "#B97914"
TEAL = "#287D78"

CANDIDATES = [1, 2, 3, 4, 5, 6]
LABELS = {
    1: "Word unigram, C=1",
    2: "Word 1-2 gram, C=1",
    3: "Word 1-2 gram, C=0.1",
    4: "Character 3-5 gram, C=1",
    5: "Character 3-5 gram, C=0.1",
    6: "Word+character union, C=0.1",
}
CODES = {
    1: "WU / C=1",
    2: "W1-2 / C=1",
    3: "W1-2 / C=.1",
    4: "C3-5 / C=1",
    5: "C3-5 / C=.1",
    6: "W+C / C=.1",
}
WINNER_COLORS = {1: BLUE, 2: TEAL, 4: OCHRE}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_with_frozen_ties(values: np.ndarray) -> np.ndarray:
    order = np.lexsort((np.asarray(CANDIDATES), -values))
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def bootstrap_accuracy(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = np.empty((BOOTSTRAP_REPLICATES, matrix.shape[1]), dtype=np.float64)
    chunk_size = 1000
    for start in range(0, BOOTSTRAP_REPLICATES, chunk_size):
        stop = min(start + chunk_size, BOOTSTRAP_REPLICATES)
        indices = rng.integers(
            0,
            matrix.shape[0],
            size=(stop - start, matrix.shape[0]),
            dtype=np.int32,
        )
        output[start:stop] = matrix[indices].mean(axis=1)
    return output


def winner_regions(clean: np.ndarray, exposed: np.ndarray) -> pd.DataFrame:
    intersections = {0.0, 1.0}
    slopes = exposed - clean
    for left in range(len(CANDIDATES)):
        for right in range(left + 1, len(CANDIDATES)):
            denominator = slopes[left] - slopes[right]
            if abs(denominator) < 1e-15:
                continue
            weight = (clean[right] - clean[left]) / denominator
            if 0.0 < weight < 1.0:
                intersections.add(float(weight))
    points = sorted(intersections)
    regions: list[dict[str, float | int | str]] = []
    for start, end in zip(points[:-1], points[1:], strict=True):
        midpoint = (start + end) / 2.0
        scores = midpoint * exposed + (1.0 - midpoint) * clean
        winner = int(np.argmax(scores)) + 1
        if regions and regions[-1]["candidate"] == winner and np.isclose(
            float(regions[-1]["end_weight"]), start
        ):
            regions[-1]["end_weight"] = end
        else:
            regions.append(
                {
                    "start_weight": start,
                    "end_weight": end,
                    "candidate": winner,
                    "candidate_code": CODES[winner],
                    "candidate_label": LABELS[winner],
                }
            )
    return pd.DataFrame(regions)


def load_and_analyze() -> dict[str, object]:
    predictions_path = RESULTS_ROOT / "predictions.csv.gz"
    development_path = RESULTS_ROOT / "development_candidates.csv"
    summary_path = RESULTS_ROOT / "summary.json"
    uncertainty_path = RESULTS_ROOT / "uncertainty_validation.json"

    predictions = pd.read_csv(predictions_path)
    development = pd.read_csv(development_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    uncertainty = json.loads(uncertainty_path.read_text(encoding="utf-8"))

    validation = predictions.loc[
        predictions["evaluation_set"].isin(["target", "clean_validation"])
    ].copy()
    for (candidate, evaluation_set), block in validation.groupby(
        ["candidate", "evaluation_set"]
    ):
        correct = block.pivot(index="pair_id", columns="seed", values="correct")
        margins = block.pivot(
            index="pair_id",
            columns="seed",
            values="margin_response2_minus_response1",
        )
        if not correct.nunique(axis=1).eq(1).all() or not np.allclose(
            margins.to_numpy(), margins.iloc[:, [0]].to_numpy()
        ):
            raise ValueError(
                f"nominal seeds differ for candidate {candidate}, {evaluation_set}"
            )

    seed = int(sorted(validation["seed"].unique())[0])
    unique = validation.loc[validation["seed"].eq(seed)]
    matrices: dict[str, np.ndarray] = {}
    identifiers: dict[str, set[str]] = {}
    for evaluation_set in ["target", "clean_validation"]:
        block = unique.loc[unique["evaluation_set"].eq(evaluation_set)]
        pivot = block.pivot(index="pair_id", columns="candidate", values="correct")
        pivot = pivot.reindex(columns=CANDIDATES).sort_index()
        if pivot.shape != (512, 6) or pivot.isna().any().any():
            raise ValueError(f"unexpected validation matrix for {evaluation_set}")
        matrices[evaluation_set] = pivot.to_numpy(dtype=np.float64)
        identifiers[evaluation_set] = set(pivot.index.astype(str))
    if identifiers["target"] & identifiers["clean_validation"]:
        raise ValueError("validation slices unexpectedly share record identifiers")

    exposed = matrices["target"].mean(axis=0)
    clean = matrices["clean_validation"].mean(axis=0)
    exposed_rank = rank_with_frozen_ties(exposed)
    clean_rank = rank_with_frozen_ties(clean)

    candidate_rows = []
    for index, candidate in enumerate(CANDIDATES):
        candidate_rows.append(
            {
                "candidate": candidate,
                "candidate_code": CODES[candidate],
                "candidate_label": LABELS[candidate],
                "exposed_accuracy": exposed[index],
                "clean_accuracy": clean[index],
                "exposed_rank": int(exposed_rank[index]),
                "clean_rank": int(clean_rank[index]),
            }
        )
    candidate_summary = pd.DataFrame(candidate_rows)

    exposed_order = np.lexsort((np.asarray(CANDIDATES), -exposed))
    clean_order = np.lexsort((np.asarray(CANDIDATES), -clean))
    exposed_winner = int(exposed_order[0]) + 1
    clean_winner = int(clean_order[0]) + 1
    exposed_margin = exposed[exposed_order[0]] - exposed[exposed_order[1]]
    clean_margin = clean[clean_order[0]] - clean[clean_order[1]]

    if exposed_winner != 1 or clean_winner != 4:
        raise ValueError("endpoint selections differ from the frozen result")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    exposed_boot = bootstrap_accuracy(matrices["target"], rng)
    clean_boot = bootstrap_accuracy(matrices["clean_validation"], rng)
    exposed_boot_winner = np.argmax(exposed_boot, axis=1)
    clean_boot_winner = np.argmax(clean_boot, axis=1)

    endpoint_rows = []
    for criterion, winners, observed_winner in [
        ("exposed_validation", exposed_boot_winner, exposed_winner),
        ("ancestry_clean_validation", clean_boot_winner, clean_winner),
    ]:
        frequencies = np.bincount(winners, minlength=6) / BOOTSTRAP_REPLICATES
        for index, frequency in enumerate(frequencies):
            endpoint_rows.append(
                {
                    "criterion": criterion,
                    "candidate": index + 1,
                    "candidate_code": CODES[index + 1],
                    "winner_probability": frequency,
                    "observed_winner": index + 1 == observed_winner,
                }
            )
    endpoint_stability = pd.DataFrame(endpoint_rows)

    probability_rows = []
    for weight in WEIGHTS:
        weighted = weight * exposed_boot + (1.0 - weight) * clean_boot
        winners = np.argmax(weighted, axis=1)
        frequencies = np.bincount(winners, minlength=6) / BOOTSTRAP_REPLICATES
        for candidate, frequency in zip(CANDIDATES, frequencies, strict=True):
            probability_rows.append(
                {
                    "exposed_weight": weight,
                    "candidate": candidate,
                    "candidate_code": CODES[candidate],
                    "winner_probability": frequency,
                }
            )
    winner_probabilities = pd.DataFrame(probability_rows)
    regions = winner_regions(clean, exposed)

    endpoint_accuracy = {
        "exposed_selected_accuracy": float(
            summary["seed_results"][0]["exposed_selected_untouched_accuracy"]
        ),
        "clean_selected_accuracy": float(
            summary["seed_results"][0]["clean_selected_untouched_accuracy"]
        ),
        "clean_minus_exposed": float(uncertainty["frozen_point_estimate"]),
        "ci_low": float(uncertainty["primary_context_paired_bootstrap_95"][0]),
        "ci_high": float(uncertainty["primary_context_paired_bootstrap_95"][1]),
        "clean_only_correct": int(uncertainty["discordant_counts"]["clean_only_correct"]),
        "exposed_only_correct": int(
            uncertainty["discordant_counts"]["exposed_only_correct"]
        ),
        "mcnemar_p": float(uncertainty["mcnemar_exact_two_sided_p"]),
    }

    observed_pairwise_crossover = (
        clean[3] - clean[0]
    ) / ((exposed[0] - clean[0]) - (exposed[3] - clean[3]))

    result_summary = {
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "validation_resampling": (
            "independent context bootstrap within each disjoint validation slice; "
            "six candidate vectors preserved within each resampled slice"
        ),
        "tie_rule": "lowest frozen candidate number",
        "exposed_winner": exposed_winner,
        "exposed_winner_margin": exposed_margin,
        "exposed_winner_bootstrap_frequency": float(
            (exposed_boot_winner == exposed_winner - 1).mean()
        ),
        "clean_winner": clean_winner,
        "clean_winner_margin": clean_margin,
        "clean_winner_bootstrap_frequency": float(
            (clean_boot_winner == clean_winner - 1).mean()
        ),
        "endpoint_winner_pairwise_crossover": float(observed_pairwise_crossover),
        "observed_winner_regions": regions.to_dict(orient="records"),
        "untouched": endpoint_accuracy,
        "procedure_level_limitation": (
            "Untouched predictions are available only for the two endpoint-selected "
            "candidates, so complete-procedure reselection and test-performance-by-weight "
            "estimands are not identifiable from the released artifact."
        ),
    }

    source_paths = [predictions_path, development_path, summary_path, uncertainty_path]
    source_files = pd.DataFrame(
        {
            "path": [path.relative_to(RELEASE_ROOT) for path in source_paths],
            "sha256": [sha256(path) for path in source_paths],
        }
    )

    return {
        "candidate_summary": candidate_summary,
        "endpoint_stability": endpoint_stability,
        "winner_probabilities": winner_probabilities,
        "winner_regions": regions,
        "result_summary": result_summary,
        "source_files": source_files,
    }
