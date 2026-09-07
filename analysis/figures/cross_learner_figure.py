#!/usr/bin/env python3
"""Generate the controlled cross-learner figure from released results."""

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

import cross_learner_data as data_source


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
FIGURES = PROJECT / "figures"
LEARNERS = ["controlled_word", "controlled_character", "controlled_neural"]
DISPLAY = {
    "controlled_word": "Word TF-IDF",
    "controlled_character": "Character TF-IDF",
    "controlled_neural": "Qwen2.5-1.5B",
}

INK = "#20262E"
MID = "#737C88"
LIGHT = "#D8DDE4"
PALE = "#F2F4F7"
WHITE = "#FFFFFF"
BLUE = "#2368B8"
OCHRE = "#B97914"
TEAL = "#287D78"
SEED = "#A8AFB8"
COLORS = {"controlled_word": BLUE, "controlled_character": OCHRE, "controlled_neural": TEAL}


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


def cell(cells: pd.DataFrame, learner: str, evaluation: str, boundary: str) -> float:
    return 100.0 * data_source.cell_value(cells, learner, evaluation, boundary)


def interval(row: pd.Series, digits: int = 2) -> str:
    return f"{100 * row.estimate:+.{digits}f} [{100 * row.ci_low:.{digits}f}, {100 * row.ci_high:.{digits}f}]"


def clean_axis(axis: plt.Axes, grid: str = "x") -> None:
    axis.grid(axis=grid, color=PALE, linewidth=0.7, zorder=0)
    axis.spines[["top", "right", "left"]].set_visible(False)


def draw_decomposition_cell(
    axis: plt.Axes,
    cells: pd.DataFrame,
    seeds: pd.DataFrame,
    evaluation: str,
) -> None:
    ypos = {"controlled_word": 2.0, "controlled_character": 1.0, "controlled_neural": 0.0}
    axis.set_xlim(40, 75)
    axis.set_ylim(-0.48, 2.48)
    axis.axvline(50, color=LIGHT, linestyle=":", linewidth=0.8, zorder=0)
    axis.grid(axis="x", color=PALE, linewidth=0.65, zorder=0)
    for learner in LEARNERS:
        y = ypos[learner]
        safe = cell(cells, learner, evaluation, "safe")
        naive = cell(cells, learner, evaluation, "naive")
        color = COLORS[learner]
        axis.annotate(
            "",
            xy=(naive, y),
            xytext=(safe, y),
            arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.1,
                        "mutation_scale": 7, "shrinkA": 4, "shrinkB": 4},
            zorder=2,
        )
        if learner == "controlled_neural":
            block = seeds.loc[seeds.evaluation_set.eq(evaluation)]
            yjitter = np.asarray([-0.15, 0.0, 0.15])
            for boundary in ("safe", "naive"):
                values = 100 * block.loc[block.boundary.eq(boundary), "accuracy"].to_numpy()
                axis.scatter(
                    values,
                    y + yjitter,
                    s=13,
                    marker="D",
                    facecolor=WHITE,
                    edgecolor=SEED,
                    linewidth=0.7,
                    zorder=3,
                )
        axis.scatter(safe, y, s=36, facecolor=WHITE, edgecolor=color, linewidth=1.25, zorder=4)
        axis.scatter(naive, y, s=36, facecolor=color, edgecolor=color, linewidth=1.0, zorder=4)
        axis.text(safe, y - 0.25, f"{safe:.2f}", ha="center", va="top", fontsize=6.7, color=INK)
        axis.text(naive, y + 0.25, f"{naive:.2f}", ha="center", va="bottom", fontsize=6.7, color=INK)
        label_x = (safe + naive) / 2
        label_ha = "center"
        if learner == "controlled_neural" and evaluation == "clean_validation":
            label_x = 73.0
            label_ha = "right"
        axis.text(
            label_x,
            y + 0.39,
            f"Δ {naive - safe:+.2f} pp",
            ha=label_ha,
            va="bottom",
            fontsize=6.4,
            color=color,
        )
    axis.set_yticks([])
    axis.set_xticks([40, 50, 60, 70])
    axis.set_xlabel("Accuracy (%)", labelpad=2)
    axis.spines[["top", "right", "left"]].set_visible(False)


def draw_did_forest(axis: plt.Axes, factorial: pd.DataFrame, seeds: pd.DataFrame) -> None:
    frame = factorial.set_index("learner").loc[LEARNERS]
    y = np.asarray([2.0, 1.0, 0.0])
    axis.axvline(0, color=INK, linewidth=0.85)
    axis.grid(axis="x", color=PALE, linewidth=0.65)
    for position, learner in zip(y, LEARNERS, strict=True):
        row = frame.loc[learner]
        estimate, low, high = 100 * row.estimate, 100 * row.ci_low, 100 * row.ci_high
        axis.errorbar(
            estimate,
            position,
            xerr=[[estimate - low], [high - estimate]],
            fmt="o",
            color=COLORS[learner],
            markerfacecolor=WHITE,
            markeredgewidth=1.15,
            markersize=5.2,
            linewidth=1.1,
            capsize=2.3,
            zorder=4,
        )
    seed_wide = seeds.pivot_table(index="seed", columns=["evaluation_set", "boundary"], values="accuracy")
    seed_did = 100 * (
        seed_wide[("target", "naive")] - seed_wide[("target", "safe")]
        - seed_wide[("clean_validation", "naive")] + seed_wide[("clean_validation", "safe")]
    )
    axis.scatter(seed_did, [-0.15, 0.0, 0.15], s=13, marker="D", facecolor=WHITE, edgecolor=SEED, linewidth=0.7)
    axis.set_xlim(-5, 22)
    axis.set_ylim(-0.48, 2.48)
    axis.set_xticks([0, 10, 20])
    axis.set_yticks([])
    axis.set_xlabel("Target-minus-clean\neffect (pp)", fontsize=7.0)
    axis.spines[["top", "right", "left"]].set_visible(False)


def draw_interaction_forest(axis: plt.Axes, interactions: pd.DataFrame) -> None:
    frame = interactions.set_index("comparison").loc[["qwen_minus_word", "qwen_minus_character"]]
    y = np.asarray([1.0, 0.0])
    axis.axvline(0, color=INK, linewidth=0.85)
    axis.grid(axis="x", color=PALE, linewidth=0.65)
    for position, comparison, color in zip(y, frame.index, [BLUE, OCHRE], strict=True):
        row = frame.loc[comparison]
        estimate, low, high = 100 * row.estimate, 100 * row.ci_low, 100 * row.ci_high
        axis.errorbar(
            estimate,
            position,
            xerr=[[estimate - low], [high - estimate]],
            fmt="o",
            color=color,
            markerfacecolor=WHITE,
            markeredgewidth=1.15,
            markersize=5.2,
            linewidth=1.1,
            capsize=2.3,
        )
        axis.text(13.8, position, interval(row), ha="right", va="center", fontsize=7.0)
    axis.set_xlim(-21, 14)
    axis.set_ylim(-0.55, 1.55)
    axis.set_yticks(y, ["Qwen\n- word", "Qwen\n- character"])
    axis.set_xticks([-20, -15, -10, -5, 0])
    axis.set_xlabel("Difference in target-minus-clean effect (pp)")
    axis.tick_params(axis="y", length=0, pad=6)
    axis.spines[["top", "right", "left"]].set_visible(False)


def make_decomposition(data: dict[str, pd.DataFrame]) -> None:
    cells = data["aggregate_cells"]
    seeds = data["qwen_seed_cells"]
    figure = plt.figure(figsize=(7.35, 5.20))
    figure.text(0.04, 0.925, "A", ha="left", va="top", fontsize=11.0, fontweight="bold")
    figure.text(0.07, 0.925, "Accuracy by training condition", ha="left", va="top", fontsize=9.8, fontweight="semibold")

    figure.text(0.49, 0.844, "○ Ancestor excluded  →  ● Ancestor included", ha="center", va="center", fontsize=7.4, color=INK)
    left, width, gap = 0.245, 0.132, 0.028
    axes = [
        figure.add_axes([left + index * (width + gap), 0.430, width, 0.300])
        for index in range(3)
    ]
    did_axis = figure.add_axes([0.755, 0.430, 0.095, 0.300])
    headings = [
        (left + width / 2, "Exposed target"),
        (left + width + gap + width / 2, "Ancestry-clean\nvalidation"),
        (left + 2 * (width + gap) + width / 2, "Untouched evaluation"),
    ]
    for xpos, heading in headings:
        figure.text(xpos, 0.805, heading, ha="center", va="center", fontsize=7.7, fontweight="semibold", linespacing=1.0)
    for axis, evaluation, title in zip(
        axes,
        ["target", "clean_validation", "untouched_clean_test"],
        ["Exposed target", "Ancestry-clean validation", "Untouched evaluation"],
        strict=True,
    ):
        draw_decomposition_cell(axis, cells, seeds, evaluation)
    draw_did_forest(did_axis, data["factorial"], seeds)
    figure.text(0.855, 0.805, "Target-minus-clean effect\n95% interval", ha="center", va="center", fontsize=7.7, fontweight="semibold", linespacing=1.0)
    factorial = data["factorial"].set_index("learner")
    for ypos, learner in zip([0.686, 0.582, 0.478], LEARNERS, strict=True):
        figure.text(0.855, ypos, interval(factorial.loc[learner]), ha="left", va="center", fontsize=6.2)

    learner_rows = [0.679, 0.575, 0.471]
    for learner, ypos in zip(LEARNERS, learner_rows, strict=True):
        figure.text(0.04, ypos, DISPLAY[learner], ha="left", va="center", fontsize=8.7, fontweight="bold")

    figure.add_artist(Line2D([0.04, 0.965], [0.305, 0.305], transform=figure.transFigure, color=LIGHT, linewidth=0.9))
    figure.text(0.04, 0.268, "B", ha="left", va="center", fontsize=11.0, fontweight="bold")
    figure.text(0.07, 0.268, "Cross-learner differences in target-minus-clean effect", ha="left", va="center", fontsize=9.4, fontweight="semibold")
    interaction_axis = figure.add_axes([0.34, 0.085, 0.55, 0.135])
    draw_interaction_forest(interaction_axis, data["interactions"])

    output = FIGURES / "controlled_cross_learner_factorial_main"
    metadata = {"Title": "Controlled cross-learner experiment", "Creator": "Matplotlib", "CreationDate": None, "ModDate": None}
    figure.savefig(output.with_suffix(".pdf"), metadata=metadata)
    figure.savefig(output.with_suffix(".png"), dpi=320)
    plt.close(figure)


def validate(data: dict[str, pd.DataFrame]) -> None:
    factorial = data["factorial"].set_index("learner")
    interactions = data["interactions"].set_index("comparison")
    assert np.isclose(factorial.loc["controlled_word", "estimate"], 0.15625)
    assert np.isclose(factorial.loc["controlled_character", "estimate"], 0.0859375)
    assert np.isclose(factorial.loc["controlled_neural", "estimate"], -0.0022786458333333)
    assert np.isclose(interactions.loc["qwen_minus_word", "estimate"], -0.1585286458333333)
    assert np.isclose(interactions.loc["qwen_minus_character", "estimate"], -0.0882161458333333)
    output = FIGURES / "controlled_cross_learner_factorial_main.pdf"
    if output.stat().st_size < 10_000:
        raise RuntimeError("Cross-learner PDF was not generated correctly")


def main() -> None:
    configure()
    data = data_source.load_data()
    make_decomposition(data)
    validate(data)


if __name__ == "__main__":
    main()
