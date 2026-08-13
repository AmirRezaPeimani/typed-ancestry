#!/usr/bin/env python3
"""Write compact LaTeX value macros from controlled result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}"


def command(name: str, value: str | int) -> str:
    return rf"\newcommand{{\{name}}}{{{value}\xspace}}"


def write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def simple_macros(summary: dict[str, Any]) -> list[str]:
    lines = ["% Generated from controlled_sparse_factorial/summary.json."]
    for learner, prefix in (
        ("controlled_word", "ControlledWord"),
        ("controlled_character", "ControlledChar"),
    ):
        block = summary["models"][learner]
        primary = block["primary"]
        untouched = block["untouched_clean_consequence"]
        lines.extend(
            [
                command(
                    f"{prefix}Did",
                    percent(primary["difference_in_differences"]),
                ),
                command(
                    f"{prefix}DidLow",
                    percent(primary["hierarchical_bootstrap_95"][0]),
                ),
                command(
                    f"{prefix}DidHigh",
                    percent(primary["hierarchical_bootstrap_95"][1]),
                ),
                command(
                    f"{prefix}Untouched",
                    percent(untouched["naive_minus_safe_accuracy"]),
                ),
                command(
                    f"{prefix}UntouchedLow",
                    percent(untouched["hierarchical_bootstrap_95"][0]),
                ),
                command(
                    f"{prefix}UntouchedHigh",
                    percent(untouched["hierarchical_bootstrap_95"][1]),
                ),
                command(
                    f"{prefix}UniquePredictionVectors",
                    block["unique_prediction_vectors"],
                ),
            ]
        )
    return lines


def neural_macros(summary: dict[str, Any]) -> list[str]:
    primary = summary["primary"]
    untouched = summary["untouched_clean_consequence"]
    competence = summary["competence"]
    word = summary["interactions"]["qwen_minus_word"]
    character = summary["interactions"]["qwen_minus_character"]
    return [
        "% Generated from controlled_qwen15_factorial/summary.json.",
        command(
            "ControlledNeuralDid",
            percent(primary["difference_in_differences"]),
        ),
        command(
            "ControlledNeuralDidLow",
            percent(primary["hierarchical_bootstrap_95"][0]),
        ),
        command(
            "ControlledNeuralDidHigh",
            percent(primary["hierarchical_bootstrap_95"][1]),
        ),
        command(
            "ControlledNeuralUntouched",
            percent(untouched["naive_minus_safe_accuracy"]),
        ),
        command(
            "ControlledNeuralUntouchedLow",
            percent(untouched["hierarchical_bootstrap_95"][0]),
        ),
        command(
            "ControlledNeuralUntouchedHigh",
            percent(untouched["hierarchical_bootstrap_95"][1]),
        ),
        command(
            "ControlledNeuralUniquePredictionVectors",
            summary["unique_prediction_vectors"],
        ),
        command("ControlledNeuralCompetencePairs", competence["pairs"]),
        command("ControlledNeuralCompetenceCorrect", competence["correct"]),
        command(
            "ControlledNeuralCompetenceIncorrect",
            competence["incorrect"],
        ),
        command(
            "ControlledNeuralCompetenceTies",
            competence["exact_ties"],
        ),
        command(
            "ControlledNeuralCompetenceTieAdjustedAccuracy",
            percent(competence["tie_adjusted_accuracy"]),
        ),
        command(
            "ControlledNeuralCompetenceStrictAccuracy",
            percent(competence["strict_accuracy"]),
        ),
        command(
            "ControlledNeuralCompetenceStrictWilsonLow",
            percent(competence["strict_wilson_95"][0]),
        ),
        command(
            "ControlledNeuralCompetenceStrictWilsonHigh",
            percent(competence["strict_wilson_95"][1]),
        ),
        command(
            "ControlledNeuralMinusWord",
            percent(word["estimate"]),
        ),
        command(
            "ControlledNeuralMinusWordLow",
            percent(word["hierarchical_paired_bootstrap_95"][0]),
        ),
        command(
            "ControlledNeuralMinusWordHigh",
            percent(word["hierarchical_paired_bootstrap_95"][1]),
        ),
        command(
            "ControlledNeuralMinusChar",
            percent(character["estimate"]),
        ),
        command(
            "ControlledNeuralMinusCharLow",
            percent(character["hierarchical_paired_bootstrap_95"][0]),
        ),
        command(
            "ControlledNeuralMinusCharHigh",
            percent(character["hierarchical_paired_bootstrap_95"][1]),
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root)
    args = parser.parse_args()
    resolved = args.root.resolve()

    sparse = read_json(
        resolved / "results" / "controlled_sparse_factorial" / "summary.json"
    )
    neural = read_json(
        resolved / "results" / "controlled_qwen15_factorial" / "summary.json"
    )
    sparse_lines = simple_macros(sparse)
    neural_lines = neural_macros(neural)
    write(
        resolved / "macros" / "controlled_sparse_macros.tex",
        sparse_lines,
    )
    write(
        resolved / "macros" / "controlled_neural_macros.tex",
        neural_lines,
    )
    write(
        resolved / "macros" / "controlled_cross_learner_macros.tex",
        [
            "% Self-contained controlled cross-learner value macros.",
            *sparse_lines,
            *neural_lines,
        ],
    )
    print("wrote controlled macro files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
