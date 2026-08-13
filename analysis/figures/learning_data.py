#!/usr/bin/env python3
"""Load and recompute evidence used by the learning-dynamics figures."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
EXPOSURE_SUMMARY = PROJECT / "results" / "typed_ancestry" / "exposure_summary.json"
EXPOSURE_TENSOR = PROJECT / "results" / "typed_ancestry" / "exposure_same_any_view.csv"
DYNAMICS_DIR = PROJECT / "results" / "learning_dynamics"
ARRIVAL_DIR = PROJECT / "results" / "ancestor_arrival"
LEXICAL_PREDICTIONS = PROJECT / "results" / "randomized_one_pass" / "word_predictions.csv.gz"
ENCODER_PREDICTIONS = PROJECT / "results" / "randomized_one_pass" / "minilm_predictions.csv.gz"

WORD = "#2F6BDE"
MINILM = "#C47A00"
INK = "#252A31"
MID = "#6B7280"
LIGHT = "#D8DEE8"
PALE = "#F4F6F9"
SAME = "#9AA1AA"
ANY = "#20242A"
SEEDS = (20260727, 20260728, 20260729)
FRACTIONS = (0.2, 0.4, 0.6, 0.8)
TRAINING_PAIRS = 4512
PROCESSED = {0.2: 902, 0.4: 1805, 0.6: 2707, 0.8: 3610}
EVENT_EDGES = np.asarray([-0.8, -0.4, -0.2, 0.0, 0.2, 0.4, 0.8])
EVENT_CENTERS = np.asarray([-0.6, -0.3, -0.1, 0.1, 0.3, 0.6])
EVENT_LABELS = ("-0.8--0.4", "-0.4--0.2", "-0.2-0", "0-0.2", "0.2-0.4", "0.4-0.8")


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.7,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def panel_title(ax: plt.Axes, letter: str, title: str, *, y: float = 1.04) -> None:
    ax.text(
        0.0,
        y,
        f"{letter}  {title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.2,
        fontweight="bold",
        color=INK,
    )


def quiet_axis(ax: plt.Axes, *, grid_axis: str = "x") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=LIGHT, linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)


def load_evidence() -> dict[str, object]:
    exposure = json.loads(EXPOSURE_SUMMARY.read_text())
    endpoints = pd.DataFrame(exposure["endpoints"])
    tensor = pd.read_csv(EXPOSURE_TENSOR)
    effects = pd.read_csv(DYNAMICS_DIR / "boundary_effects.csv")
    trajectory = pd.read_csv(DYNAMICS_DIR / "learning_dynamics.csv")
    cells = pd.read_csv(DYNAMICS_DIR / "accuracy_cells.csv")
    arrival = json.loads((ARRIVAL_DIR / "summary.json").read_text())
    return {
        "endpoints": endpoints,
        "tensor": tensor,
        "effects": effects,
        "trajectory": trajectory,
        "cells": cells,
        "arrival": arrival,
    }


def record_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    badge: str,
    fields: list[str],
    color: str,
) -> dict[str, tuple[float, float]]:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.01,rounding_size=0.018",
        linewidth=0.9,
        edgecolor=INK,
        facecolor="white",
        zorder=2,
    )
    ax.add_patch(box)
    header_h = 0.20 * height
    header = FancyBboxPatch(
        (x, y + height - header_h),
        width,
        header_h,
        boxstyle="round,pad=0.01,rounding_size=0.018",
        linewidth=0.8,
        edgecolor=INK,
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(header)
    title_size = 6.1 if len(title) > 20 else 6.8
    ax.text(x + 0.035 * width, y + height - header_h / 2, title, va="center", ha="left", fontsize=title_size, fontweight="bold")
    if badge:
        ax.text(
            x + 0.94 * width,
            y + height - header_h / 2,
            badge,
            va="center",
            ha="right",
            fontsize=6.1,
            fontweight="bold",
            color=INK,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor=color, linewidth=0.7),
        )
    anchors: dict[str, tuple[float, float]] = {}
    usable = height - header_h
    for index, field in enumerate(fields):
        cy = y + usable - (index + 0.5) * usable / len(fields)
        ax.text(x + 0.055 * width, cy, field, va="center", ha="left", fontsize=6.25, color=MID)
        anchors[field] = (x + width, cy)
        if index:
            sep_y = y + usable - index * usable / len(fields)
            ax.plot([x + 0.05 * width, x + 0.95 * width], [sep_y, sep_y], color=LIGHT, linewidth=0.55, zorder=2)
    return anchors


def draw_schematic(ax: plt.Axes, *, compact: bool) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if compact:
        panel_title(ax, "A", "How row matching fails", y=1.01)
        ax.text(0.0, 0.935, "Descriptive audit: n = 163 validation contexts", fontsize=6.7, color=MID)
        y, h = 0.21, 0.59
    else:
        panel_title(ax, "A", "A typed ancestry path across dataset views", y=1.015)
        ax.text(0.0, 0.95, "Schematic - descriptive audit: n = 163 edit_quality validation contexts", fontsize=7.5, color=MID)
        y, h = 0.19, 0.64
    train_x, train_w = 0.01, 0.36
    val_x, val_w = 0.63, 0.36
    train_fields = ["context", "original response", "feedback", "edited response"]
    val_fields = ["context", "original response", "feedback", "good edited response", "bad edited response"]
    left = record_box(ax, train_x, y, train_w, h, "TRAIN: edit view", "" if compact else "TRAIN", train_fields, "#ABC4F3")
    right_left = val_x
    right = record_box(ax, val_x, y, val_w, h, "VALIDATION: edit_quality", "" if compact else "EVAL", val_fields, "#F1D79B")
    for field, style, label in (
        ("context", "-", "exact context"),
        ("feedback", "--", "exact feedback"),
        ("original response", "-", "exact response"),
    ):
        x0, y0 = left[field]
        _, y1 = right[field]
        ax.plot([x0, right_left], [y0, y1], color=MID, linewidth=0.75, linestyle=style, zorder=1)
        if not compact or field == "context":
            ax.text(0.5, (y0 + y1) / 2 + 0.012, label, ha="center", va="bottom", fontsize=5.8, color=MID)
    x0, y0 = left["edited response"]
    _, y1 = right["good edited response"]
    ax.annotate(
        "",
        xy=(right_left, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color="#2B7A6D", lw=1.25, connectionstyle="arc3,rad=-0.09"),
    )
    ax.text(0.5, min(y0, y1) - 0.012, "directed atom: original -> edited", ha="center", va="top", fontsize=5.6 if compact else 6.2, color="#2B7A6D")
    ax.plot([train_x + train_w, val_x], [y + h + 0.07, y + h + 0.07], color="#9B3F3F", linewidth=0.8)
    ax.text(0.5, y + h + 0.07, "x", ha="center", va="center", fontsize=10, color="#9B3F3F", fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", pad=0.1))
    ax.text(0.5, y + h + 0.105, "Different schemas: no full-row match", ha="center", fontsize=6.4, color="#9B3F3F")
    callout = "0/163 full-row matches\nbut 123/163 context and 103/163 directed-edit exposures" if compact else "0/163 full-row matches, but 123/163 context and 103/163 directed-edit exposures"
    ax.text(0.5, 0.10 if compact else 0.085, callout, ha="center", va="center", fontsize=6.25 if compact else 7.4, linespacing=1.1, fontweight="bold", color=INK, bbox=dict(boxstyle="round,pad=0.24", facecolor=PALE, edgecolor=LIGHT, linewidth=0.7))
    ax.text(0.5, 0.004, "Schematic; the field names encode the preferred good edit (no separate quality-label field).", ha="center", va="bottom", fontsize=5.2 if compact else 6.0, color=MID)


def focal_exposure(endpoints: pd.DataFrame) -> pd.DataFrame:
    names = {"record": "Complete record", "context": "Context", "response": "Response", "feedback": "Feedback", "edit_link": "Directed edit"}
    rows = []
    for atom, label in names.items():
        for scope in ("same_view", "any_view"):
            row = endpoints[(endpoints.evaluation_view == "edit_quality") & (endpoints.atom_type == atom) & (endpoints.scope == scope)].iloc[0]
            rows.append(
                {
                    "atom": atom,
                    "label": label,
                    "scope": scope,
                    "estimate": 100 * float(row.estimate),
                    "ci_low": 100 * float(row.cluster_bootstrap_95ci[0]),
                    "ci_high": 100 * float(row.cluster_bootstrap_95ci[1]),
                    "count": int(row.exposed_rows),
                    "n": int(row.rows),
                }
            )
    return pd.DataFrame(rows)


def draw_dumbbell(
    ax: plt.Axes,
    endpoints: pd.DataFrame,
    *,
    letter: str = "B",
    title: str = "Cross-view exposure missed by same-view audits",
    title_y: float = 1.04,
) -> None:
    data = focal_exposure(endpoints)
    labels = ["Complete record", "Context", "Response", "Feedback", "Directed edit"]
    y = np.arange(len(labels))[::-1]
    quiet_axis(ax, grid_axis="x")
    ax.axvline(0, color=INK, linewidth=0.75, zorder=1)
    for yy, label in zip(y, labels, strict=True):
        same = data[(data.label == label) & (data.scope == "same_view")].iloc[0]
        anyv = data[(data.label == label) & (data.scope == "any_view")].iloc[0]
        ax.plot([same.estimate, anyv.estimate], [yy, yy], color=SAME, linewidth=1.1, zorder=2)
        ax.scatter(same.estimate, yy, s=25, facecolor="white", edgecolor=SAME, linewidth=1.0, zorder=3)
        ax.errorbar(anyv.estimate, yy, xerr=[[anyv.estimate - anyv.ci_low], [anyv.ci_high - anyv.estimate]], fmt="o", ms=4.6, color=ANY, ecolor=ANY, capsize=2.0, elinewidth=0.9, zorder=4)
        ax.text(83.5, yy, f"{anyv['count']}/{anyv.n}", va="center", fontsize=6.8, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(-2, 98)
    ax.set_ylim(-0.6, 5.0)
    ax.set_xlabel("Validation records exposed (%)")
    ax.text(0.01, 0.99, "open gray: same-view   filled black: any-view", transform=ax.transAxes, ha="left", va="top", fontsize=6.4, color=MID)
    panel_title(ax, letter, title, y=title_y)


def final_effects(effects: pd.DataFrame) -> pd.DataFrame:
    labels = {"target": "Exposed target validation", "clean_validation": "Ancestry-clean validation", "untouched_clean_test": "Untouched clean test"}
    subset = effects[(effects.fraction == 1.0) & effects.evaluation_set.isin(labels)].copy()
    subset["evaluation_label"] = subset.evaluation_set.map(labels)
    for col in ("naive_minus_safe", "ci_low", "ci_high"):
        subset[col] *= 100
    return subset


def draw_effect_decomposition(ax: plt.Axes, effects: pd.DataFrame) -> None:
    data = final_effects(effects)
    quiet_axis(ax, grid_axis="x")
    ax.axvline(0, color=INK, linewidth=0.85, zorder=1)
    layouts = [
        ("word_tfidf_bt", WORD, "o", [5, 4, 3], "Word TF-IDF", 20.4, "+14.26 [10.16, 18.36]"),
        ("minilm_frozen_bt", MINILM, "s", [1, 0, -1], "Frozen MiniLM", 6.7, "+1.56 [-1.56, 4.69]"),
    ]
    order = ["target", "clean_validation", "untouched_clean_test"]
    for model, color, marker, ys, group, bracket_x, did_text in layouts:
        part = data[data.model == model].set_index("evaluation_set")
        for yy, key in zip(ys, order, strict=True):
            row = part.loc[key]
            ax.errorbar(row.naive_minus_safe, yy, xerr=[[row.naive_minus_safe - row.ci_low], [row.ci_high - row.naive_minus_safe]], fmt=marker, ms=4.7, mfc=color if model == "word_tfidf_bt" else "white", mec=color, color=color, ecolor=color, capsize=2, elinewidth=0.9, zorder=3)
            ax.text(row.ci_high + 0.6, yy, f"{row.naive_minus_safe:+.1f}", va="center", fontsize=6.5, color=color)
        ax.text(-7.7, ys[0] + 0.48, group, ha="left", va="bottom", fontsize=7.4, fontweight="bold", color=color)
        ax.plot([bracket_x, bracket_x], [ys[1], ys[0]], color=color, linewidth=0.85)
        ax.plot([bracket_x - 0.45, bracket_x], [ys[0], ys[0]], color=color, linewidth=0.85)
        ax.plot([bracket_x - 0.45, bracket_x], [ys[1], ys[1]], color=color, linewidth=0.85)
        ax.text(bracket_x + 0.35, np.mean(ys[:2]), f"Target - clean DiD\n{did_text}", ha="left", va="center", fontsize=6.2, color=color)
    labels = ["Target", "Clean validation", "Untouched", "Target", "Clean validation", "Untouched"]
    ax.set_yticks([5, 4, 3, 1, 0, -1], labels)
    ax.set_xlim(-8, 32)
    ax.set_ylim(-1.75, 5.85)
    ax.set_xlabel("Naïve accuracy - ancestry-safe accuracy (percentage points)")
    panel_title(ax, "C", "The boundary changes exposed validation,\nnot clean performance", y=1.02)


def draw_trajectory(
    ax: plt.Axes,
    trajectory: pd.DataFrame,
    *,
    title: bool = False,
    annotate_final: bool = True,
) -> None:
    quiet_axis(ax, grid_axis="y")
    ax.axhline(0, color=INK, linewidth=0.75, zorder=1)
    for model, color, marker, fill, linestyle, label in (
        ("word_tfidf_bt", WORD, "o", WORD, "-", "Word TF-IDF"),
        ("minilm_frozen_bt", MINILM, "s", "white", "--", "Frozen MiniLM"),
    ):
        rows = trajectory[trajectory.model == model].sort_values("fraction")
        x = 100 * rows.fraction.to_numpy()
        y = 100 * rows.did.to_numpy()
        low = 100 * rows.ci_low.to_numpy()
        high = 100 * rows.ci_high.to_numpy()
        ax.plot(x, y, color=color, linewidth=1.25, linestyle=linestyle, marker=marker, markersize=4.0, markerfacecolor=fill, markeredgecolor=color, zorder=3)
        ax.fill_between(x, low, high, color=color, alpha=0.10, linewidth=0, zorder=2)
        short_label = "Word" if model == "word_tfidf_bt" else "MiniLM"
        ax.text(x[-1] + 1.0, y[-1], short_label, color=color, fontsize=6.6, va="center", fontweight="bold")
    ax.set_xlim(17, 114)
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.set_xlabel("Training completed (%)")
    ax.set_ylabel("Target-minus-clean DiD\n(percentage points)")
    if annotate_final:
        ax.text(100, 18.9, "Word: +14.26 [10.16, 18.36]", ha="right", va="top", fontsize=6.5, color=WORD)
    if title:
        ax.set_title("Checkpoint connections; no interpolation implied", loc="left", fontsize=7.2, color=MID, pad=2)


def arrival_summary_rows(arrival: dict[str, object]) -> pd.DataFrame:
    word = arrival["model_summaries"]["word_tfidf_bt"]
    mini = arrival["model_summaries"]["minilm_frozen_bt"]
    values = [
        ("Word: target ancestry", WORD, "o", True, word["target_correct_mean_arrival_contrast"], word["target_correct_mean_arrival_contrast_95ci"]),
        ("Word: clean placebo", WORD, "o", False, word["clean_placebo_correct_mean_arrival_contrast"], word["clean_placebo_correct_mean_arrival_contrast_95ci"]),
        ("Word: target - placebo", WORD, "D", True, word["target_minus_placebo_correct_mean_contrast"], word["target_minus_placebo_correct_mean_contrast_95ci"]),
        ("MiniLM: target ancestry", MINILM, "s", False, mini["target_correct_mean_arrival_contrast"], mini["target_correct_mean_arrival_contrast_95ci"]),
    ]
    return pd.DataFrame(
        [
            {"label": label, "color": color, "marker": marker, "filled": filled, "estimate": 100 * est, "ci_low": 100 * ci[0], "ci_high": 100 * ci[1]}
            for label, color, marker, filled, est, ci in values
        ]
    )


def draw_arrival_forest(ax: plt.Axes, arrival: dict[str, object], *, compact: bool = True) -> None:
    data = arrival_summary_rows(arrival)
    quiet_axis(ax, grid_axis="x")
    ax.axvline(0, color=INK, linewidth=0.75, zorder=1)
    ys = np.arange(len(data))[::-1]
    for yy, row in zip(ys, data.itertuples(), strict=True):
        ax.errorbar(row.estimate, yy, xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]], fmt=row.marker, ms=5.0 if row.marker == "D" else 4.2, mfc=row.color if row.filled else "white", mec=row.color, color=row.color, ecolor=row.color, capsize=2, elinewidth=0.9, zorder=3)
    ax.set_yticks(ys, data.label)
    ax.set_xlim(-5, 19)
    ax.set_xlabel("Seen - not-yet-seen contrast (percentage points)", fontsize=7.0)
    ax.set_title("Mean over preterminal checkpoints, 20-80%", loc="left", fontsize=7.2, color=MID, pad=2)
    ax.text(0.99, 0.02, r"Word target: $p < 5\times10^{-5}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3, color=WORD)


def prepare_tensor(tensor: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("preference", "record", "Preference record"),
        ("preference", "context", "Preference context"),
        ("feedback", "record", "Feedback record"),
        ("feedback", "context", "Feedback context"),
        ("edit", "record", "Edit record"),
        ("edit", "context", "Edit context"),
        ("principle", "record", "Principle record"),
        ("principle", "context", "Principle context"),
        ("edit_quality", "record", "Edit-quality record"),
        ("edit_quality", "context", "Edit-quality context"),
        ("edit_quality", "response", "Edit-quality response"),
        ("edit_quality", "feedback", "Edit-quality feedback"),
        ("edit_quality", "edit_link", "Edit-quality edit relation"),
    ]
    output = []
    for view, atom, label in rows:
        row = tensor[(tensor.evaluation_view == view) & (tensor.atom_type == atom)].iloc[0]
        output.append(
            {
                "label": label,
                "same_view": 100 * row.same_view_exposure_rate,
                "cross_view_only": 100 * row.cross_view_only_rate,
                "any_view": 100 * row.any_view_exposure_rate,
            }
        )
    return pd.DataFrame(output)


def draw_tensor_heatmap(ax: plt.Axes, tensor: pd.DataFrame) -> None:
    data = prepare_tensor(tensor)
    matrix = data[["same_view", "cross_view_only", "any_view"]].to_numpy()
    cmap = LinearSegmentedColormap.from_list("neutral_exposure", ["#FFFFFF", "#D5DCE1", "#5D6972"])
    x_edges = np.arange(matrix.shape[1] + 1) - 0.5
    y_edges = np.arange(matrix.shape[0] + 1) - 0.5
    image = ax.pcolormesh(x_edges, y_edges, matrix, cmap=cmap, vmin=0, vmax=85, shading="flat")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=5.8, color="white" if value > 52 else INK)
    ax.set_xticks([0, 1, 2], ["Same-view", "Cross-view\nonly", "Any-view"])
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(data)), data.label)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(Rectangle((-0.5, 7.5), 3, 5, fill=False, edgecolor="#8A5B00", linewidth=1.4, clip_on=False))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(len(data) - 0.5, -0.5)
    ax.text(0.0, -0.085, "edit_quality block: row 0.0% | any-view context 75.5% | directed edit 63.2%", transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color="#8A5B00", fontweight="bold")
    panel_title(ax, "B", "Exposure tensor across validation views", y=1.12)
    cax = ax.inset_axes([0.02, -0.17, 0.96, 0.04])
    color_edges = np.linspace(0, 85, 18)
    color_values = ((color_edges[:-1] + color_edges[1:]) / 2)[None, :]
    cax.pcolormesh(color_edges, [0, 1], color_values, cmap=cmap, vmin=0, vmax=85, shading="flat")
    cax.set_xlim(0, 85)
    cax.set_yticks([])
    cax.set_xticks(np.arange(0, 81, 10))
    cax.tick_params(axis="x", labelsize=6, length=2)
    cax.set_xlabel("Validation records exposed (%)", fontsize=6.5, labelpad=2)


def stable_rank(seed: int, identifier: str) -> str:
    payload = f"{seed}\0clean_placebo_arrival\0{identifier}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_predictions() -> pd.DataFrame:
    frames = [pd.read_csv(LEXICAL_PREDICTIONS), pd.read_csv(ENCODER_PREDICTIONS)]
    return pd.concat(frames, ignore_index=True)


def event_aligned_estimates(replicates: int = 20_000, seed: int = 20261011) -> pd.DataFrame:
    predictions = read_predictions()
    predictions = predictions[
        (predictions.model == "word_tfidf_bt")
        & predictions.evaluation_set.isin(["target", "clean_validation"])
        & predictions.fraction.isin(FRACTIONS)
    ].copy()
    pivot = predictions.pivot_table(
        index=["seed", "fraction", "evaluation_set", "pair_id"],
        columns="boundary",
        values="correct",
        aggfunc="first",
    ).reset_index()
    pivot["delta"] = pivot["naive"] - pivot["safe"]
    target_delta = pivot[pivot.evaluation_set == "target"]
    clean_delta = pivot[pivot.evaluation_set == "clean_validation"]
    arrival_items = pd.read_csv(ARRIVAL_DIR / "arrival_item_effects.csv.gz")
    positions = (
        arrival_items[arrival_items.model == "word_tfidf_bt"]
        [["seed", "pair_id", "donor_position_zero_based"]]
        .drop_duplicates()
    )

    seed_data: list[pd.DataFrame] = []
    for seed_value in SEEDS:
        pos = positions[positions.seed == seed_value].sort_values("donor_position_zero_based").reset_index(drop=True)
        clean_ids = sorted(
            clean_delta[clean_delta.seed == seed_value].pair_id.unique(),
            key=lambda pair_id: stable_rank(seed_value, pair_id),
        )
        if len(pos) != 512 or len(clean_ids) != 512:
            raise ValueError("event-aligned analysis requires 512 target-placebo pairs per seed")
        mapping = pd.DataFrame(
            {
                "item_index": np.arange(512),
                "target_id": pos.pair_id,
                "clean_id": clean_ids,
                "position": pos.donor_position_zero_based,
            }
        )
        records = []
        for checkpoint_index, fraction in enumerate(FRACTIONS):
            target = target_delta[(target_delta.seed == seed_value) & (target_delta.fraction == fraction)].set_index("pair_id").delta
            clean = clean_delta[(clean_delta.seed == seed_value) & (clean_delta.fraction == fraction)].set_index("pair_id").delta
            for row in mapping.itertuples(index=False):
                event_time = PROCESSED[fraction] / TRAINING_PAIRS - (row.position + 0.5) / TRAINING_PAIRS
                event_time = float(np.clip(event_time, -0.799999, 0.799999))
                event_bin = int(np.digitize(event_time, EVENT_EDGES[1:-1], right=False))
                records.append(
                    {
                        "seed": seed_value,
                        "item_index": row.item_index,
                        "checkpoint_index": checkpoint_index,
                        "fraction": fraction,
                        "event_time": event_time,
                        "event_bin": event_bin,
                        "outcome": float(target.loc[row.target_id] - clean.loc[row.clean_id]),
                    }
                )
        seed_data.append(pd.DataFrame(records))

    parameter_count = 11
    sxx = np.zeros((3, 3, 512, parameter_count, parameter_count), dtype=np.float64)
    sxy = np.zeros((3, 3, 512, parameter_count), dtype=np.float64)
    observed_counts = np.zeros(6, dtype=np.int64)
    for original_seed_index, data in enumerate(seed_data):
        observed_counts += np.bincount(data.event_bin, minlength=6)
        for slot in range(3):
            for item_index, item in data.groupby("item_index", sort=True):
                x = np.zeros((4, parameter_count), dtype=np.float64)
                x[:, 0] = 1
                bins = item.sort_values("checkpoint_index").event_bin.to_numpy()
                checkpoints = item.sort_values("checkpoint_index").checkpoint_index.to_numpy()
                outcomes = item.sort_values("checkpoint_index").outcome.to_numpy()
                for row_index, event_bin in enumerate(bins):
                    if event_bin:
                        x[row_index, event_bin] = 1
                    if slot:
                        x[row_index, 5 + slot] = 1
                    if checkpoints[row_index]:
                        x[row_index, 7 + checkpoints[row_index]] = 1
                sxx[slot, original_seed_index, item_index] = x.T @ x
                sxy[slot, original_seed_index, item_index] = x.T @ outcomes

    g = np.zeros((6, parameter_count), dtype=np.float64)
    for event_bin in range(6):
        rows = []
        for slot in range(3):
            for checkpoint_index in range(4):
                x = np.zeros(parameter_count, dtype=np.float64)
                x[0] = 1
                if event_bin:
                    x[event_bin] = 1
                if slot:
                    x[5 + slot] = 1
                if checkpoint_index:
                    x[7 + checkpoint_index] = 1
                rows.append(x)
        g[event_bin] = np.mean(rows, axis=0)

    observed_xx = sum(sxx[slot, slot].sum(axis=0) for slot in range(3))
    observed_xy = sum(sxy[slot, slot].sum(axis=0) for slot in range(3))
    observed_beta = np.linalg.solve(observed_xx, observed_xy)
    observed = g @ observed_beta

    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, 6), dtype=np.float64)
    batch_size = 20
    for start in range(0, replicates, batch_size):
        stop = min(replicates, start + batch_size)
        count = stop - start
        original_seed_draws = rng.integers(0, 3, size=(count, 3))
        xx = np.zeros((count, parameter_count, parameter_count), dtype=np.float64)
        xy = np.zeros((count, parameter_count), dtype=np.float64)
        for slot in range(3):
            item_draws = rng.integers(0, 512, size=(count, 512))
            selected_seed = original_seed_draws[:, slot]
            xx += sxx[slot][selected_seed[:, None], item_draws].sum(axis=1)
            xy += sxy[slot][selected_seed[:, None], item_draws].sum(axis=1)
        beta = np.linalg.solve(xx, xy[..., None])[..., 0]
        draws[start:stop] = beta @ g.T

    intervals = np.quantile(draws, [0.025, 0.975], axis=0)
    return pd.DataFrame(
        {
            "event_bin": np.arange(6),
            "event_interval": EVENT_LABELS,
            "event_center": EVENT_CENTERS,
            "estimate": 100 * observed,
            "ci_low": 100 * intervals[0],
            "ci_high": 100 * intervals[1],
            "observations": observed_counts,
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
        }
    )


def accuracy_cell_table(cells: pd.DataFrame) -> pd.DataFrame:
    subset = cells[(cells.fraction == 1.0) & cells.evaluation_set.isin(["target", "clean_validation", "untouched_clean_test"])].copy()
    return subset.groupby(["model", "boundary", "evaluation_set"], as_index=False).accuracy.mean()
