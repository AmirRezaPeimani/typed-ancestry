#!/usr/bin/env python3
"""Build the typed-ancestry and learning-dynamics figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

import learning_data as data_source


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
FIGURES = PROJECT / "figures"

INK = "#282D33"
MID = "#707985"
LIGHT = "#C8CED5"
PALE = "#F4F5F6"
TEAL = "#2A7F73"
RED = "#A34D49"
WORD = "#2F6BDE"
MINILM = "#C47A00"

LETTER_SIZE = 11.5
TITLE_SIZE = 10.3
FACET_SIZE = 9.5
AXIS_SIZE = 8.8
TICK_SIZE = 8.1
ANNOTATION_SIZE = 7.7
CELL_SIZE = 7.5

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": TICK_SIZE,
        "axes.labelsize": AXIS_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.75,
        "axes.unicode_minus": False,
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


def panel_header(ax: plt.Axes, letter: str, title: str, *, y: float = 1.075, title_size: float = TITLE_SIZE) -> None:
    ax.text(0.0, y, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=LETTER_SIZE, fontweight="bold")
    ax.annotate(title, xy=(0.0, y), xycoords=ax.transAxes, xytext=(23, 0), textcoords="offset points", ha="left", va="bottom", fontsize=title_size, fontweight="semibold")


def quiet_axis(ax: plt.Axes, *, grid_axis: str = "x") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color="#E0E4E8", linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)


def record_card(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    header: str,
    fields: list[str],
    header_fill: str,
) -> dict[str, tuple[float, float]]:
    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.016",
        facecolor="white",
        edgecolor=INK,
        linewidth=0.9,
        zorder=2,
    )
    ax.add_patch(card)
    header_height = 0.18 * height
    header_box = FancyBboxPatch(
        (x, y + height - header_height),
        width,
        header_height,
        boxstyle="round,pad=0.008,rounding_size=0.016",
        facecolor=header_fill,
        edgecolor=INK,
        linewidth=0.8,
        zorder=3,
    )
    ax.add_patch(header_box)
    ax.text(x + 0.045 * width, y + height - header_height / 2, header, ha="left", va="center", fontsize=8.5, fontweight="bold")

    body_top = y + height - header_height
    body_bottom = y
    row_step = (body_top - body_bottom) / len(fields)
    anchors: dict[str, tuple[float, float]] = {}
    for index, field in enumerate(fields):
        center = body_top - (index + 0.5) * row_step
        ax.text(x + 0.055 * width, center, field, ha="left", va="center", fontsize=7.8, color=MID)
        anchors[field] = (x + width, center)
        if index:
            rule_y = body_top - index * row_step
            ax.plot([x + 0.055 * width, x + 0.945 * width], [rule_y, rule_y], color="#DCE1E6", linewidth=0.65, zorder=2)
    return anchors


def draw_schematic(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_header(ax, "A", "Typed ancestry across dataset views", y=1.005)

    card_y, card_h = 0.27, 0.56
    left_x, right_x, card_w = 0.015, 0.625, 0.36
    left = record_card(
        ax,
        x=left_x,
        y=card_y,
        width=card_w,
        height=card_h,
        header="TRAIN · edit",
        fields=["context", "original response", "feedback", "edited response"],
        header_fill="#DCE5F3",
    )
    right = record_card(
        ax,
        x=right_x,
        y=card_y,
        width=card_w,
        height=card_h,
        header="VALIDATION · edit_quality",
        fields=["context", "original response", "feedback", "good edited response", "bad edited response"],
        header_fill="#ECE7DB",
    )

    ax.text(0.5, 0.883, "x", ha="right", va="center", fontsize=9.4, fontweight="bold", color=RED)
    ax.text(0.51, 0.883, "No complete-row match", ha="left", va="center", fontsize=ANNOTATION_SIZE, color=RED)

    connector_specs = [
        ("context", "context", "exact context"),
        ("original response", "original response", "exact response"),
        ("feedback", "feedback", "exact feedback"),
    ]
    for left_field, right_field, label in connector_specs:
        x0, y0 = left[left_field]
        _, y1 = right[right_field]
        ax.plot([x0, right_x], [y0, y1], color=MID, linewidth=0.85, zorder=1)
        ax.text(0.5, (y0 + y1) / 2 + 0.027, label, ha="center", va="bottom", fontsize=ANNOTATION_SIZE, color=MID, bbox=dict(facecolor="white", edgecolor="none", pad=0.4))

    x0, y0 = left["edited response"]
    _, y1 = right["good edited response"]
    ax.annotate(
        "",
        xy=(right_x, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=TEAL, linewidth=1.25, connectionstyle="arc3,rad=-0.08"),
    )
    ax.text(0.5, min(y0, y1) - 0.045, "exact directed-edit relation", ha="center", va="top", fontsize=ANNOTATION_SIZE, color=TEAL, bbox=dict(facecolor="white", edgecolor="none", pad=0.4))

    strip_x, strip_y, strip_w, strip_h = 0.18, 0.035, 0.64, 0.13
    strip = FancyBboxPatch(
        (strip_x, strip_y),
        strip_w,
        strip_h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=PALE,
        edgecolor="#E2E5E8",
        linewidth=0.65,
    )
    ax.add_patch(strip)
    ax.text(
        strip_x + strip_w / 2,
        strip_y + strip_h / 2,
        "0/163 complete-row matches · 123/163 context exposures · 103/163 directed-edit exposures",
        ha="center",
        va="center",
        fontsize=7.6,
        color=INK,
        fontweight="semibold",
    )


def draw_exposure_tensor(ax: plt.Axes, tensor: pd.DataFrame) -> None:
    data = data_source.prepare_tensor(tensor)
    matrix = data[["same_view", "cross_view_only", "any_view"]].to_numpy()
    cmap = LinearSegmentedColormap.from_list("neutral_exposure", ["#FFFFFF", "#D5DADF", "#58616A"])
    x_edges = np.arange(4) - 0.5
    y_edges = np.arange(14) - 0.5
    ax.pcolormesh(x_edges, y_edges, matrix, cmap=cmap, vmin=0, vmax=85, shading="flat")
    ax.axhspan(7.5, 12.5, color="white", alpha=0.11, zorder=2)
    ax.add_patch(Rectangle((-0.5, 7.5), 3, 5, fill=False, edgecolor="#8C9299", linewidth=0.9, zorder=4))

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            ax.text(column_index, row_index, f"{value:.1f}", ha="center", va="center", fontsize=CELL_SIZE, color="white" if value > 45 else INK, zorder=5)

    ax.set_xticks([0, 1, 2], ["Same view", "Cross-view\nonly", "Any view"])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_yticks([])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(12.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    groups = [
        ("Preference", 0.5, ["record", "context"]),
        ("Feedback", 2.5, ["record", "context"]),
        ("Edit", 4.5, ["record", "context"]),
        ("Principle", 6.5, ["record", "context"]),
        ("edit_quality", 10.0, ["record", "context", "response", "feedback", "directed edit"]),
    ]
    row_index = 0
    for group, center, sublabels in groups:
        if group != "edit_quality":
            ax.text(-0.82, center, group, ha="right", va="center", fontsize=7.2, color=INK, fontweight="semibold", clip_on=False, linespacing=1.2)
        else:
            ax.text(-1.36, center, group, ha="right", va="center", fontsize=6.9, color=INK, fontweight="bold", family="DejaVu Sans Mono", clip_on=False)
        for sublabel in sublabels:
            ax.text(-0.61, row_index, sublabel, ha="right", va="center", fontsize=TICK_SIZE, color=INK, clip_on=False)
            row_index += 1

    panel_header(ax, "B", "Exposure across validation views", y=1.16)

    cax = ax.inset_axes([0.14, -0.145, 0.72, 0.035])
    color_edges = np.linspace(0, 85, 18)
    color_values = ((color_edges[:-1] + color_edges[1:]) / 2)[None, :]
    cax.pcolormesh(color_edges, [0, 1], color_values, cmap=cmap, vmin=0, vmax=85, shading="flat")
    cax.set_xlim(0, 85)
    cax.set_yticks([])
    cax.set_xticks([0, 20, 40, 60, 80])
    cax.tick_params(axis="x", labelsize=CELL_SIZE, length=2, pad=2)
    cax.set_xlabel("% exposed", fontsize=CELL_SIZE, labelpad=1)


def draw_exposure_zoom(ax: plt.Axes, endpoints: pd.DataFrame) -> None:
    data = data_source.focal_exposure(endpoints)
    labels = ["Complete record", "Context", "Response", "Feedback", "Directed edit"]
    y_positions = np.arange(len(labels))[::-1]
    quiet_axis(ax, grid_axis="x")
    ax.axvline(0, color=INK, linewidth=0.75, zorder=1)
    ax.axvline(88, color="#D9DDE1", linewidth=0.65, zorder=1)

    for y_value, label in zip(y_positions, labels, strict=True):
        same = data[(data.label == label) & (data.scope == "same_view")].iloc[0]
        any_view = data[(data.label == label) & (data.scope == "any_view")].iloc[0]
        if label == "Complete record":
            ax.scatter(0, y_value, s=44, facecolor="white", edgecolor=MID, linewidth=1.0, zorder=4)
            ax.scatter(0, y_value, s=13, facecolor=INK, edgecolor=INK, linewidth=0.7, zorder=5)
            ax.text(10.0, y_value, "both 0", ha="left", va="center", fontsize=ANNOTATION_SIZE, color=MID)
        else:
            ax.plot([same.estimate, any_view.estimate], [y_value, y_value], color="#9DA5AE", linewidth=1.2, zorder=2)
            ax.scatter(same.estimate, y_value, s=35, facecolor="white", edgecolor="#9DA5AE", linewidth=1.0, zorder=4)
            ax.errorbar(
                any_view.estimate,
                y_value,
                xerr=[[any_view.estimate - any_view.ci_low], [any_view.ci_high - any_view.estimate]],
                fmt="o",
                markersize=5.4,
                markerfacecolor=INK,
                markeredgecolor=INK,
                color=INK,
                ecolor=INK,
                elinewidth=0.9,
                capsize=2.0,
                zorder=5,
            )
        ax.text(94.0, y_value, f"{int(any_view['count'])}/{int(any_view.n)}", ha="left", va="center", fontsize=ANNOTATION_SIZE, color=INK)

    display_labels = ["Complete\nrecord", "Context", "Response", "Feedback", "Directed edit"]
    ax.set_yticks(y_positions, display_labels)
    ax.set_xlim(-2.5, 112)
    ax.set_ylim(-0.55, 4.65)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel("Validation records exposed (%)")
    ax.text(102.0, 4.55, "Count", ha="center", va="bottom", fontsize=CELL_SIZE, color=MID, fontweight="semibold")
    panel_header(ax, "C", r"$\mathtt{edit\_quality}$ exposure", y=1.16, title_size=9.0)

    legend = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="#9DA5AE", markersize=5.2, label="Same view"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=INK, markeredgecolor=INK, markersize=5.2, label="Any view"),
    ]
    ax.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.98, 1.085), frameon=False, ncol=2, columnspacing=0.8, handletextpad=0.35, fontsize=CELL_SIZE, borderaxespad=0)


def accuracy_cells(cells: pd.DataFrame, model: str) -> pd.DataFrame:
    return data_source.accuracy_cell_table(cells).query("model == @model").set_index(["evaluation_set", "boundary"])


def draw_interaction_facet(ax: plt.Axes, cells: pd.DataFrame, *, model: str, color: str, title: str) -> dict[str, np.ndarray]:
    part = accuracy_cells(cells, model)
    x = np.asarray([0.0, 1.0])
    target = 100 * np.asarray([part.loc[("target", "safe")].accuracy, part.loc[("target", "naive")].accuracy])
    clean = 100 * np.asarray([part.loc[("clean_validation", "safe")].accuracy, part.loc[("clean_validation", "naive")].accuracy])
    untouched = 100 * np.asarray([part.loc[("untouched_clean_test", "safe")].accuracy, part.loc[("untouched_clean_test", "naive")].accuracy])

    ax.axhline(50, color=MID, linewidth=0.75, linestyle=":", zorder=1)
    marker = "o" if model == "word_tfidf_bt" else "s"
    fill = color if model == "word_tfidf_bt" else "white"
    ax.plot(x, target, color=color, linewidth=1.45, marker=marker, markerfacecolor=fill, markeredgecolor=color, markersize=5.0, label="Target", zorder=3)
    ax.plot(x, clean, color=color, linewidth=1.15, linestyle="--", marker=marker, markerfacecolor="white", markeredgecolor=color, markersize=4.6, label="Clean", zorder=3)
    ax.set_xlim(-0.10, 1.10)
    ax.set_ylim(44, 67)
    ax.set_xticks([0, 1], ["Ancestry-safe", "Naïve"])
    ax.set_title(title, loc="left", fontsize=FACET_SIZE, fontweight="semibold", color=color, pad=8)
    quiet_axis(ax, grid_axis="y")
    return {"target": target, "clean": clean, "untouched": untouched}


def draw_value_matrix(ax: plt.Axes, values: dict[str, np.ndarray], *, model: str, color: str, did: str) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    marker = "o" if model == "word_tfidf_bt" else "s"
    fill = color if model == "word_tfidf_bt" else "white"
    ax.text(0.02, 0.97, f"Target-minus-clean DiD\n{did}", ha="left", va="top", fontsize=ANNOTATION_SIZE, color=color, linespacing=1.12, bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor=color, linewidth=0.7))

    ax.add_patch(FancyBboxPatch((0.0, 0.04), 0.98, 0.58, boxstyle="round,pad=0.012,rounding_size=0.02", facecolor=PALE, edgecolor="#DDE1E5", linewidth=0.6))
    ax.plot([0.55, 0.55], [0.08, 0.57], color="#D8DDE2", linewidth=0.6)
    ax.text(0.68, 0.54, "Safe", ha="center", va="center", fontsize=CELL_SIZE, fontweight="semibold")
    ax.text(0.93, 0.54, "Naïve", ha="center", va="center", fontsize=CELL_SIZE, fontweight="semibold")
    rows = [("Target", values["target"], 0.41, "-"), ("Clean", values["clean"], 0.27, "--")]
    for label, pair, y_value, linestyle in rows:
        ax.plot([0.025, 0.13], [y_value, y_value], color=color, linewidth=1.15 if label == "Target" else 0.9, linestyle=linestyle)
        ax.scatter(0.078, y_value, marker=marker, s=22, facecolor=fill if label == "Target" else "white", edgecolor=color, linewidth=0.8)
        ax.text(0.17, y_value, label, ha="left", va="center", fontsize=CELL_SIZE)
        ax.text(0.68, y_value, f"{pair[0]:.1f}", ha="center", va="center", fontsize=CELL_SIZE, family="DejaVu Sans Mono")
        ax.text(0.93, y_value, f"{pair[1]:.1f}", ha="center", va="center", fontsize=CELL_SIZE, family="DejaVu Sans Mono")
    ax.plot([0.025, 0.13], [0.14, 0.14], color=color, linewidth=0.9)
    ax.text(0.17, 0.14, "Untouched", ha="left", va="center", fontsize=CELL_SIZE)
    ax.text(0.68, 0.075, f"{values['untouched'][0]:.1f}", ha="center", va="center", fontsize=CELL_SIZE, family="DejaVu Sans Mono")
    ax.text(0.93, 0.075, f"{values['untouched'][1]:.1f}", ha="center", va="center", fontsize=CELL_SIZE, family="DejaVu Sans Mono")


def draw_untouched_strip(ax: plt.Axes, cells: pd.DataFrame, *, model: str, color: str) -> None:
    part = accuracy_cells(cells, model)
    values = 100 * np.asarray([part.loc[("untouched_clean_test", "safe")].accuracy, part.loc[("untouched_clean_test", "naive")].accuracy])
    marker = "o" if model == "word_tfidf_bt" else "s"
    fill = color if model == "word_tfidf_bt" else "white"
    ax.axhline(50, color=MID, linewidth=0.7, linestyle=":")
    ax.plot([0, 1], values, color=color, linewidth=1.0, marker=marker, markerfacecolor=fill, markeredgecolor=color, markersize=4.2)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(44, 53)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([45, 50])
    ax.set_ylabel("Untouched")
    quiet_axis(ax, grid_axis="y")


def draw_checkpoint_trajectory(ax: plt.Axes, trajectory: pd.DataFrame) -> None:
    quiet_axis(ax, grid_axis="y")
    ax.axhline(0, color=INK, linewidth=0.8, zorder=1)
    for model, color, marker, linestyle, name, filled in [
        ("word_tfidf_bt", WORD, "o", "-", "Word", True),
        ("minilm_frozen_bt", MINILM, "s", "--", "MiniLM", False),
    ]:
        data = trajectory[trajectory.model == model].sort_values("fraction")
        x = 100 * data.fraction.to_numpy()
        y = 100 * data.did.to_numpy()
        low = 100 * data.ci_low.to_numpy()
        high = 100 * data.ci_high.to_numpy()
        ax.plot(x, y, color=color, linewidth=1.2, linestyle=linestyle, zorder=2)
        ax.errorbar(
            x,
            y,
            yerr=[y - low, high - y],
            fmt=marker,
            markersize=5.0,
            markerfacecolor=color if filled else "white",
            markeredgecolor=color,
            color=color,
            ecolor=color,
            elinewidth=0.9,
            capsize=2.0,
            zorder=3,
        )
        ax.text(102.0, y[-1], name, ha="left", va="center", fontsize=ANNOTATION_SIZE, color=color, fontweight="semibold")
    ax.set_xlim(16, 107)
    ax.set_ylim(-4, 21)
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.set_xlabel("Training completed (%)")
    ax.set_ylabel("Target-minus-clean DiD (pp)")
    panel_header(ax, "B", "Validation distortion during training", y=1.10)


def controlled_summary_rows(trajectory: pd.DataFrame, arrival: dict[str, object]) -> list[dict[str, object]]:
    final = trajectory[trajectory.fraction == 1.0].set_index("model")
    word = arrival["model_summaries"]["word_tfidf_bt"]
    return [
        {
            "group": "WORD TF-IDF",
            "label": "Target-minus-clean DiD",
            "color": WORD,
            "marker": "o",
            "y": 3.6,
            "estimate": 100 * final.loc["word_tfidf_bt", "did"],
            "low": 100 * final.loc["word_tfidf_bt", "ci_low"],
            "high": 100 * final.loc["word_tfidf_bt", "ci_high"],
        },
        {
            "group": "WORD TF-IDF",
            "label": "Target-minus-placebo arrival contrast",
            "color": WORD,
            "marker": "D",
            "y": 2.7,
            "estimate": 100 * word["target_minus_placebo_correct_mean_contrast"],
            "low": 100 * word["target_minus_placebo_correct_mean_contrast_95ci"][0],
            "high": 100 * word["target_minus_placebo_correct_mean_contrast_95ci"][1],
        },
        {
            "group": "FROZEN MINILM",
            "label": "Target-minus-clean DiD",
            "color": MINILM,
            "marker": "o",
            "y": 1.2,
            "estimate": 100 * final.loc["minilm_frozen_bt", "did"],
            "low": 100 * final.loc["minilm_frozen_bt", "ci_low"],
            "high": 100 * final.loc["minilm_frozen_bt", "ci_high"],
        },
    ]


def draw_controlled_summary(label_ax: plt.Axes, forest_ax: plt.Axes, value_ax: plt.Axes, trajectory: pd.DataFrame, arrival: dict[str, object]) -> pd.DataFrame:
    rows = controlled_summary_rows(trajectory, arrival)
    for axis in (label_ax, value_ax):
        axis.set_axis_off()
        axis.set_xlim(0, 1)
        axis.set_ylim(-0.25, 4.65)
    quiet_axis(forest_ax, grid_axis="x")
    forest_ax.spines["left"].set_visible(False)
    forest_ax.axvline(0, color=INK, linewidth=0.8, zorder=1)
    for row in rows:
        forest_ax.errorbar(
            row["estimate"],
            row["y"],
            xerr=[[row["estimate"] - row["low"]], [row["high"] - row["estimate"]]],
            fmt=row["marker"],
            markersize=5.2,
            markerfacecolor=row["color"] if row["color"] == WORD else "white",
            markeredgecolor=row["color"],
            color=row["color"],
            ecolor=row["color"],
            elinewidth=0.9,
            capsize=2.0,
            zorder=3,
        )
        display_label = {
            "Target-minus-placebo arrival contrast": "Target-minus-placebo\narrival contrast",
            "Target-minus-clean DiD": "Target-minus-clean\nDiD",
        }.get(row["label"], row["label"])
        label_ax.text(
            0.0,
            row["y"],
            display_label,
            ha="left",
            va="center",
            fontsize=ANNOTATION_SIZE,
            color=INK,
            linespacing=1.05,
        )
        value_ax.text(0.02, row["y"], f"{row['estimate']:+.1f} [{row['low']:.1f}, {row['high']:.1f}]", ha="left", va="center", fontsize=CELL_SIZE, color=row["color"], family="DejaVu Sans Mono")

    label_ax.text(0.0, 4.25, "WORD TF-IDF", ha="left", va="bottom", fontsize=CELL_SIZE, fontweight="bold", color=WORD)
    label_ax.text(0.0, 1.85, "FROZEN MINILM", ha="left", va="bottom", fontsize=CELL_SIZE, fontweight="bold", color=MINILM)
    forest_ax.set_xlim(-5, 20)
    forest_ax.set_ylim(-0.25, 4.65)
    forest_ax.set_yticks([])
    forest_ax.set_xlabel("Effect (pp)")
    return pd.DataFrame(rows).drop(columns=["color"])


def draw_event_panel(ax: plt.Axes, count_ax: plt.Axes, event: pd.DataFrame) -> None:
    quiet_axis(ax, grid_axis="y")
    ax.axhline(0, color=INK, linewidth=0.8, zorder=1)
    ax.axvline(0, color="#59616A", linewidth=0.85, linestyle="--", zorder=1)
    x = event.event_center.to_numpy()
    y = event.estimate.to_numpy()
    low = event.ci_low.to_numpy()
    high = event.ci_high.to_numpy()

    ax.plot(x[:3], y[:3], color=WORD, linewidth=0.9, zorder=2)
    ax.plot(x[3:], y[3:], color=WORD, linewidth=0.9, zorder=2)
    ax.errorbar(x, y, yerr=[y - low, high - y], fmt="D", color=WORD, markerfacecolor=WORD, markeredgecolor=WORD, markersize=5.3, ecolor=WORD, elinewidth=0.95, capsize=2.2, zorder=3)
    ax.set_xlim(-0.82, 0.82)
    ax.set_ylim(-5.5, 22.5)
    ax.set_xticks(data_source.EVENT_CENTERS)
    ax.set_xticklabels([])
    ax.set_ylabel("Target-minus-placebo effect (pp)")
    ax.text(-0.40, 20.5, "Before ancestor arrival", ha="center", va="center", fontsize=ANNOTATION_SIZE, color=MID)
    ax.text(0.40, 20.5, "After ancestor arrival", ha="center", va="center", fontsize=ANNOTATION_SIZE, color=MID)
    panel_header(ax, "D", "Effect aligned to ancestor arrival", y=1.10)

    count_ax.set_xlim(-0.82, 0.82)
    count_ax.set_ylim(0, 1)
    count_ax.set_yticks([])
    count_ax.set_xticks(data_source.EVENT_CENTERS, ["-0.6", "-0.3", "-0.1", "+0.1", "+0.3", "+0.6"])
    count_ax.set_xlabel("Training progress relative to assigned ancestor arrival")
    count_ax.tick_params(axis="x", length=0, pad=10)
    for spine in count_ax.spines.values():
        spine.set_visible(False)
    count_ax.text(-0.805, 0.67, "n", ha="left", va="center", fontsize=CELL_SIZE, color=MID, fontweight="semibold")
    for x_value, observations in zip(x, event.observations, strict=True):
        count_ax.text(x_value, 0.67, f"{int(observations)}", ha="center", va="center", fontsize=CELL_SIZE, color=MID)


def save_figure(fig: plt.Figure, stem: str) -> None:
    metadata = {"Title": stem.replace("_", " "), "Creator": "Matplotlib", "CreationDate": None, "ModDate": None}
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", format="pdf", metadata=metadata, bbox_inches=None)
    fig.savefig(FIGURES / f"{stem}.png", format="png", dpi=320, bbox_inches=None)
    plt.close(fig)


def make_figure_1(evidence: dict[str, object]) -> None:
    fig = plt.figure(figsize=(7.3, 7.05), constrained_layout=False)
    ax_a = fig.add_axes([0.055, 0.595, 0.925, 0.345])
    ax_b = fig.add_axes([0.235, 0.155, 0.375, 0.345])
    ax_c = fig.add_axes([0.725, 0.155, 0.255, 0.345])
    draw_schematic(ax_a)
    draw_exposure_tensor(ax_b, evidence["tensor"])
    draw_exposure_zoom(ax_c, evidence["endpoints"])
    save_figure(fig, "learning_dynamics_main")


def make_figure_2(evidence: dict[str, object], event: pd.DataFrame) -> pd.DataFrame:
    fig = plt.figure(figsize=(7.3, 7.65), constrained_layout=False)
    fig.text(0.08, 0.972, "A", ha="left", va="top", fontsize=LETTER_SIZE, fontweight="bold")
    fig.text(0.112, 0.972, "Boundary × evaluation interaction", ha="left", va="top", fontsize=TITLE_SIZE, fontweight="semibold")

    ax_word = fig.add_axes([0.08, 0.755, 0.18, 0.145])
    ax_word_values = fig.add_axes([0.27, 0.755, 0.19, 0.145])
    ax_mini = fig.add_axes([0.55, 0.755, 0.18, 0.145])
    ax_mini_values = fig.add_axes([0.74, 0.755, 0.19, 0.145])
    ax_word_strip = fig.add_axes([0.08, 0.685, 0.18, 0.045])
    ax_mini_strip = fig.add_axes([0.55, 0.685, 0.18, 0.045])
    word_values = draw_interaction_facet(ax_word, evidence["cells"], model="word_tfidf_bt", color=WORD, title="Word TF-IDF")
    mini_values = draw_interaction_facet(ax_mini, evidence["cells"], model="minilm_frozen_bt", color=MINILM, title="Frozen MiniLM")
    draw_value_matrix(ax_word_values, word_values, model="word_tfidf_bt", color=WORD, did="+14.3 pp [10.2, 18.4]")
    draw_value_matrix(ax_mini_values, mini_values, model="minilm_frozen_bt", color=MINILM, did="+1.6 pp [-1.6, 4.7]")
    draw_untouched_strip(ax_word_strip, evidence["cells"], model="word_tfidf_bt", color=WORD)
    draw_untouched_strip(ax_mini_strip, evidence["cells"], model="minilm_frozen_bt", color=MINILM)

    ax_b = fig.add_axes([0.08, 0.405, 0.36, 0.19])
    fig.text(0.54, 0.618, "C", ha="left", va="bottom", fontsize=LETTER_SIZE, fontweight="bold")
    fig.text(0.584, 0.618, "Controlled-effect summary", ha="left", va="bottom", fontsize=TITLE_SIZE, fontweight="semibold")
    ax_c_labels = fig.add_axes([0.54, 0.405, 0.16, 0.19])
    ax_c_forest = fig.add_axes([0.70, 0.405, 0.12, 0.19])
    ax_c_values = fig.add_axes([0.825, 0.405, 0.16, 0.19])
    draw_checkpoint_trajectory(ax_b, evidence["trajectory"])
    summary = draw_controlled_summary(ax_c_labels, ax_c_forest, ax_c_values, evidence["trajectory"], evidence["arrival"])

    ax_d = fig.add_axes([0.09, 0.155, 0.87, 0.155])
    count_ax = fig.add_axes([0.09, 0.075, 0.87, 0.055])
    draw_event_panel(ax_d, count_ax, event)
    save_figure(fig, "learning_dynamics_acquisition")
    return summary


def validate(effects: pd.DataFrame, event: pd.DataFrame) -> None:
    for name in ["learning_dynamics_main.pdf", "learning_dynamics_acquisition.pdf"]:
        path = FIGURES / name
        if not path.exists() or path.stat().st_size < 10_000:
            raise RuntimeError(f"invalid figure: {path}")
    word_did = effects[(effects.group == "WORD TF-IDF") & (effects.label == "Target-minus-clean DiD")].iloc[0]
    word_arrival = effects[(effects.group == "WORD TF-IDF") & (effects.label == "Target-minus-placebo arrival contrast")].iloc[0]
    if not np.isclose(word_did.estimate, 14.2578125) or not np.isclose(word_arrival.estimate, 11.040202454200598):
        raise RuntimeError("controlled-effect summary does not match frozen evidence")
    if len(event) != 6 or int(event.observations.sum()) != 6144 or int(event.observations.min()) < 50:
        raise RuntimeError("event-aligned counts failed validation")


def main() -> None:
    evidence = data_source.load_evidence()
    event = pd.read_csv(data_source.ARRIVAL_DIR / "event_aligned_estimates.csv")
    make_figure_1(evidence)
    summary = make_figure_2(evidence, event)
    validate(summary, event)


if __name__ == "__main__":
    main()
