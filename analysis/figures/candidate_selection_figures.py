#!/usr/bin/env python3
"""Generate candidate-selection figures from released evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import candidate_selection_data as data_source


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
FIGURES = PROJECT / "figures"
CANDIDATES = [1, 2, 3, 4, 5, 6]
CODES = {
    1: "WU / C=1",
    2: "W1-2 / C=1",
    3: "W1-2 / C=0.1",
    4: "C3-5 / C=1",
    5: "C3-5 / C=0.1",
    6: "W+C / C=0.1",
}

INK = "#20262E"
MID = "#737C88"
NEUTRAL = "#BBC2CA"
LIGHT = "#DCE1E7"
PALE = "#F2F4F7"
WHITE = "#FFFFFF"
BLUE = "#2368B8"
OCHRE = "#B97914"
TEAL = "#287D78"


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


def rank_order(frame: pd.DataFrame, metric: str) -> list[int]:
    return frame.sort_values([metric, "candidate"], ascending=[False, True]).candidate.astype(int).tolist()


def candidate_color(candidate: int) -> str:
    if candidate == 1:
        return BLUE
    if candidate == 4:
        return OCHRE
    if candidate == 2:
        return TEAL
    return NEUTRAL


def draw_rank_reversal(figure: plt.Figure, frame: pd.DataFrame, summary: dict[str, object]) -> None:
    axis = figure.add_axes([0.055, 0.145, 0.570, 0.690])
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0.45, 6.85)
    figure.text(0.055, 0.925, "A", ha="left", va="top", fontsize=11.0, fontweight="bold")
    figure.text(0.085, 0.925, "Model rankings by validation set", ha="left", va="top", fontsize=9.8, fontweight="semibold")

    exposed_order = rank_order(frame, "exposed_accuracy")
    clean_order = rank_order(frame, "clean_accuracy")
    exposed_rank = {candidate: rank for rank, candidate in enumerate(exposed_order, start=1)}
    clean_rank = {candidate: rank for rank, candidate in enumerate(clean_order, start=1)}
    ypos = {rank: 6.5 - rank for rank in range(1, 7)}
    left_x, right_x = 0.40, 0.74
    axis.plot([left_x, left_x], [0.5, 5.5], color=LIGHT, linewidth=1.0)
    axis.plot([right_x, right_x], [0.5, 5.5], color=LIGHT, linewidth=1.0)
    for rank in range(1, 7):
        axis.axhline(ypos[rank], xmin=0.02, xmax=0.90, color=PALE, linewidth=0.7)

    for candidate in CANDIDATES:
        le, ri = exposed_rank[candidate], clean_rank[candidate]
        color = candidate_color(candidate) if candidate in (1, 4) else NEUTRAL
        linewidth = 2.6 if candidate in (1, 4) else 0.9
        axis.plot([left_x, right_x], [ypos[le], ypos[ri]], color=color, linewidth=linewidth, zorder=2)
        axis.scatter([left_x, right_x], [ypos[le], ypos[ri]], s=33 if candidate in (1, 4) else 22, color=color, edgecolor=INK if candidate in (1, 4) else color, linewidth=0.6, zorder=3)

    lookup = frame.set_index("candidate")
    for rank, candidate in enumerate(exposed_order, start=1):
        row = lookup.loc[candidate]
        weight = "bold" if candidate in (1, 4) else "normal"
        color = INK if candidate in (1, 4) else MID
        axis.text(left_x - 0.025, ypos[rank], f"{rank}  {CODES[candidate]}  {100 * row.exposed_accuracy:.2f}%", ha="right", va="center", fontsize=6.4, color=color, fontweight=weight)
    for rank, candidate in enumerate(clean_order, start=1):
        row = lookup.loc[candidate]
        weight = "bold" if candidate in (1, 4) else "normal"
        color = INK if candidate in (1, 4) else MID
        axis.text(right_x + 0.025, ypos[rank], f"{rank}  {CODES[candidate]}  {100 * row.clean_accuracy:.2f}%", ha="left", va="center", fontsize=6.4, color=color, fontweight=weight)

    axis.text(left_x, 6.78, "Exposed-target\nvalidation", ha="center", va="bottom", fontsize=8.5, fontweight="semibold", linespacing=1.0)
    axis.text(right_x, 6.78, "Ancestry-clean\nvalidation", ha="center", va="bottom", fontsize=8.5, fontweight="semibold", linespacing=1.0)
    axis.text(left_x, 6.43, "Exposed-target\nselection", ha="center", va="center", fontsize=6.2, color=BLUE, linespacing=1.0, bbox={"boxstyle": "round,pad=0.20", "facecolor": WHITE, "edgecolor": BLUE, "linewidth": 0.8})
    axis.text(right_x, 6.43, "Ancestry-clean\nselection", ha="center", va="center", fontsize=6.2, color=OCHRE, linespacing=1.0, bbox={"boxstyle": "round,pad=0.20", "facecolor": WHITE, "edgecolor": OCHRE, "linewidth": 0.8})
    axis.text(left_x, 6.13, f"Margin +{100 * summary['exposed_winner_margin']:.2f} pp\nSelected in {100 * summary['exposed_winner_bootstrap_frequency']:.1f}% of\nbootstrap resamples", ha="center", va="top", fontsize=5.0, color=MID, linespacing=1.15)
    axis.text(right_x, 6.13, f"Margin +{100 * summary['clean_winner_margin']:.2f} pp\nSelected in {100 * summary['clean_winner_bootstrap_frequency']:.1f}% of\nbootstrap resamples", ha="center", va="top", fontsize=5.0, color=MID, linespacing=1.15)
    axis.text(0.0, 3.35, "Candidate rank (1 = best)", ha="center", va="center", rotation=90, fontsize=7.2, color=MID)


def draw_sealed_consequence(figure: plt.Figure, summary: dict[str, object]) -> None:
    untouched = summary["untouched"]
    estimate = 100 * float(untouched["clean_minus_exposed"])
    low = 100 * float(untouched["ci_low"])
    high = 100 * float(untouched["ci_high"])
    figure.text(0.675, 0.925, "B", ha="left", va="top", fontsize=11.0, fontweight="bold")
    figure.text(0.705, 0.925, "Untouched evaluation", ha="left", va="top", fontsize=9.4, fontweight="semibold", linespacing=1.0)
    raw = figure.add_axes([0.72, 0.650, 0.245, 0.105])
    raw.plot([47.4609375, 50.5859375], [0, 0], color=INK, linewidth=1.2)
    raw.scatter(47.4609375, 0, s=48, color=BLUE, edgecolor=INK, linewidth=0.7, zorder=3)
    raw.scatter(50.5859375, 0, s=48, color=OCHRE, edgecolor=INK, linewidth=0.7, zorder=3)
    raw.text(47.15, -0.04, "Exposed-selected\n47.46%", ha="right", va="top", fontsize=6.5, color=BLUE, linespacing=1.05)
    raw.text(51.30, 0.31, "Ancestry-clean-selected\n50.59%", ha="right", va="bottom", fontsize=5.8, color=OCHRE, linespacing=1.05)
    raw.set_xlim(45.5, 52.3)
    raw.set_ylim(-0.55, 0.55)
    raw.set_yticks([])
    raw.set_xticks([46, 48, 50, 52])
    raw.set_xlabel("Untouched accuracy (%)", labelpad=2)
    raw.spines[["top", "right", "left"]].set_visible(False)

    forest = figure.add_axes([0.72, 0.420, 0.245, 0.090])
    forest.axvline(0, color=INK, linewidth=0.8)
    forest.errorbar(estimate, 0, xerr=[[estimate - low], [high - estimate]], fmt="o", markersize=5.3, markerfacecolor=WHITE, markeredgecolor=INK, markeredgewidth=1.1, color=INK, capsize=2.3, linewidth=1.05)
    forest.set_xlim(-2, 8)
    forest.set_ylim(-0.5, 0.5)
    forest.set_yticks([])
    forest.set_xticks([-2, 0, 2, 4, 6, 8])
    forest.set_xlabel("Ancestry-clean-selected -\nexposed-selected (pp)", fontsize=5.9, labelpad=1)
    forest.spines[["top", "right", "left"]].set_visible(False)
    figure.text(0.842, 0.545, "+3.13 pp [-0.98, 7.23]", ha="center", va="center", fontsize=7.0, fontweight="semibold")

    bars = figure.add_axes([0.72, 0.160, 0.245, 0.145])
    bars.axvline(0, color=INK, linewidth=0.8)
    bars.barh([0], [-51], color=BLUE, height=0.45)
    bars.barh([0], [67], color=OCHRE, height=0.45)
    bars.text(-53.5, 0, "51", ha="right", va="center", fontsize=7.2, fontweight="bold")
    bars.text(69.5, 0, "67", ha="left", va="center", fontsize=7.2, fontweight="bold")
    bars.text(-43, 0.56, "Exposed-selected\nonly correct", ha="center", va="bottom", fontsize=5.8, color=BLUE)
    bars.text(47, 0.56, "Ancestry-clean-selected\nonly correct", ha="center", va="bottom", fontsize=5.6, color=OCHRE)
    bars.text(40, -0.58, "Exact McNemar p = 0.167", ha="center", va="top", fontsize=6.2, bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 0.2})
    bars.set_xlim(-80, 80)
    bars.set_ylim(-0.85, 0.85)
    bars.set_yticks([])
    bars.set_xticks([-60, -30, 0, 30, 60], ["60", "30", "0", "30", "60"])
    bars.set_xlabel("Discordant contexts", labelpad=2)
    bars.spines[["top", "right", "left"]].set_visible(False)


def make_ranking(data: dict[str, object]) -> None:
    figure = plt.figure(figsize=(7.3, 4.55))
    draw_rank_reversal(figure, data["candidate_summary"], data["result_summary"])
    draw_sealed_consequence(figure, data["result_summary"])
    figure.text(0.055, 0.025, "Codes: WU = word unigram; W1-2 = word 1-2 gram; C3-5 = character 3-5 gram; W+C = word-character union.", ha="left", va="bottom", fontsize=5.9, color=MID)
    output = FIGURES / "candidate_selection_main"
    metadata = {"Title": "Model rankings by validation set", "Creator": "Matplotlib", "CreationDate": None, "ModDate": None}
    figure.savefig(output.with_suffix(".pdf"), metadata=metadata)
    figure.savefig(output.with_suffix(".png"), dpi=320)
    plt.close(figure)


def draw_sensitivity(figure: plt.Figure, frame: pd.DataFrame, probabilities: pd.DataFrame, regions: pd.DataFrame) -> None:
    figure.text(0.055, 0.925, "A", ha="left", va="top", fontsize=11.0, fontweight="bold")
    figure.text(0.085, 0.925, "Sensitivity to validation weighting", ha="left", va="top", fontsize=9.6, fontweight="semibold")
    exposed = frame.set_index("candidate").exposed_accuracy
    clean = frame.set_index("candidate").clean_accuracy
    strip = figure.add_axes([0.11, 0.775, 0.57, 0.050])
    for _, row in regions.iterrows():
        candidate = int(row.candidate)
        strip.axvspan(row.start_weight, row.end_weight, color=candidate_color(candidate), alpha=0.88)
        strip.text((row.start_weight + row.end_weight) / 2, 0.5, CODES[candidate].split(" /")[0], ha="center", va="center", fontsize=6.6, color=WHITE, fontweight="bold")
    strip.set_xlim(0, 1)
    strip.set_ylim(0, 1)
    strip.set_axis_off()

    axis = figure.add_axes([0.11, 0.555, 0.57, 0.195])
    weights = np.linspace(0, 1, 101)
    for candidate in CANDIDATES:
        values = 100 * ((1 - weights) * clean.loc[candidate] + weights * exposed.loc[candidate])
        color = candidate_color(candidate) if candidate in (1, 2, 4) else NEUTRAL
        width = 2.0 if candidate in (1, 2, 4) else 0.8
        axis.plot(weights, values, color=color, linewidth=width, zorder=2 if candidate in (1, 2, 4) else 1)
        if candidate in (1, 2, 4):
            axis.text(0.98, values[-1] + {1: 2.2, 2: -2.0, 4: 1.0}[candidate], CODES[candidate], ha="right", va="center", fontsize=6.1, color=color)
    boundaries = regions.end_weight.to_numpy()[:-1]
    for boundary in boundaries:
        axis.axvline(boundary, color=INK, linestyle=":", linewidth=0.8)
        axis.text(boundary, 46.5, f"w={boundary:.3f}", ha="center", va="top", fontsize=6.1, color=INK)
    axis.set_xlim(0, 1)
    axis.set_ylim(44, 66)
    axis.set_xlabel("Exposed-target validation weight, w")
    axis.set_ylabel("Weighted validation accuracy (%)")
    axis.grid(color=PALE, linewidth=0.65)
    axis.spines[["top", "right"]].set_visible(False)

    figure.text(0.11, 0.425, "B", ha="left", va="center", fontsize=11.0, fontweight="bold")
    figure.text(0.14, 0.425, "Bootstrap selection probability", ha="left", va="center", fontsize=9.1, fontweight="semibold")
    heat = figure.add_axes([0.11, 0.170, 0.57, 0.185])
    matrix = probabilities.pivot(index="candidate", columns="exposed_weight", values="winner_probability").reindex(index=CANDIDATES, columns=weights).to_numpy()
    cmap = LinearSegmentedColormap.from_list("selection_probability", [WHITE, "#BBD2E9", BLUE])
    image = heat.pcolormesh(np.linspace(0, 1, matrix.shape[1] + 1), np.arange(0.5, 7.0), matrix, vmin=0, vmax=1, cmap=cmap, shading="flat", rasterized=False)
    heat.set_ylim(6.5, 0.5)
    key_weights = [0.0, float(boundaries[0]), float(boundaries[1]), 1.0]
    for boundary in boundaries:
        heat.axvline(boundary, color=INK, linestyle=":", linewidth=0.8)
    for weight in key_weights:
        nearest = int(np.argmin(np.abs(weights - weight)))
        x = float(weights[nearest])
        align = "left" if x == 0 else "right" if x == 1 else "center"
        for candidate in CANDIDATES:
            value = matrix[candidate - 1, nearest]
            heat.text(x, candidate, f"{100 * value:.0f}%", ha=align, va="center", fontsize=5.2, color=WHITE if value > 0.52 else INK)
    heat.set_xlim(0, 1)
    heat.set_yticks(CANDIDATES, [CODES[candidate] for candidate in CANDIDATES])
    heat.set_xticks(key_weights, ["0", "w=0.184", "w=0.611", "1"])
    heat.tick_params(axis="y", length=0, pad=4, labelsize=6.2)
    heat.spines[:].set_visible(False)
    colorbar = figure.colorbar(image, ax=heat, orientation="horizontal", fraction=0.065, pad=0.25)
    colorbar.solids.set_rasterized(False)
    colorbar.set_label("Bootstrap selection probability", fontsize=6.4, labelpad=1)
    colorbar.ax.tick_params(labelsize=5.9, length=2)


def make_sensitivity(data: dict[str, object]) -> None:
    figure = plt.figure(figsize=(7.3, 5.0))
    draw_sensitivity(figure, data["candidate_summary"], data["winner_probabilities"], data["winner_regions"])
    figure.add_artist(Line2D([0.705, 0.705], [0.12, 0.91], transform=figure.transFigure, color=LIGHT, linewidth=1.0))
    figure.text(0.725, 0.925, "C", ha="left", va="top", fontsize=11.0, fontweight="bold")
    figure.text(0.755, 0.925, "Untouched\nevaluation", ha="left", va="top", fontsize=9.2, fontweight="semibold", linespacing=1.0)
    figure.text(0.725, 0.775, "Selection rule", fontsize=6.5, color=MID, fontweight="bold")
    figure.text(0.96, 0.775, "Accuracy", ha="right", fontsize=6.5, color=MID, fontweight="bold")
    figure.text(0.725, 0.715, "Exposed-target validation\nWU / C=1", ha="left", va="center", fontsize=6.8, linespacing=1.3)
    figure.text(0.96, 0.715, "47.46%", ha="right", va="center", fontsize=7.8, color=BLUE, fontweight="bold")
    figure.text(0.725, 0.620, "Ancestry-clean validation\nC3-5 / C=1", ha="left", va="center", fontsize=6.8, linespacing=1.3)
    figure.text(0.96, 0.620, "50.59%", ha="right", va="center", fontsize=7.8, color=OCHRE, fontweight="bold")
    forest = figure.add_axes([0.75, 0.390, 0.19, 0.120])
    untouched = data["result_summary"]["untouched"]
    estimate, low, high = 100 * untouched["clean_minus_exposed"], 100 * untouched["ci_low"], 100 * untouched["ci_high"]
    forest.axvline(0, color=INK, linewidth=0.8)
    forest.errorbar(estimate, 0, xerr=[[estimate - low], [high - estimate]], fmt="o", markersize=5.4, markerfacecolor=WHITE, markeredgecolor=INK, markeredgewidth=1.1, color=INK, capsize=2.3, linewidth=1.0)
    forest.set_xlim(-2, 8)
    forest.set_ylim(-0.5, 0.5)
    forest.set_yticks([])
    forest.set_xticks([-2, 0, 2, 4, 6, 8])
    forest.set_xlabel("Ancestry-clean-selected -\nexposed-selected (pp)", fontsize=5.9, labelpad=1)
    forest.spines[["top", "right", "left"]].set_visible(False)
    figure.text(0.845, 0.535, "+3.13 pp [-0.98, 7.23]", ha="center", va="center", fontsize=7.0, fontweight="semibold")
    figure.text(0.725, 0.305, "67 contexts favor ancestry-clean-selected\n51 contexts favor exposed-selected\nExact McNemar p = 0.167", ha="left", va="top", fontsize=6.2, linespacing=1.4)
    output = FIGURES / "candidate_selection_supp"
    metadata = {"Title": "Sensitivity to validation weighting", "Creator": "Matplotlib", "CreationDate": None, "ModDate": None}
    figure.savefig(output.with_suffix(".pdf"), metadata=metadata)
    figure.savefig(output.with_suffix(".png"), dpi=320)
    plt.close(figure)


def validate(data: dict[str, object]) -> None:
    summary = data["result_summary"]
    assert summary["exposed_winner"] == 1
    assert summary["clean_winner"] == 4
    assert summary["bootstrap_replicates"] == 20_000
    assert np.isclose(summary["untouched"]["clean_minus_exposed"], 0.03125)
    for stem in ("candidate_selection_main", "candidate_selection_supp"):
        if (FIGURES / f"{stem}.pdf").stat().st_size < 10_000:
            raise RuntimeError("Candidate-selection PDF was not generated correctly")


def main() -> None:
    configure()
    data = data_source.load_and_analyze()
    make_ranking(data)
    make_sensitivity(data)
    validate(data)


if __name__ == "__main__":
    main()
