"""Regression tests for the controlled release results."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_controlled_release",
    ROOT / "analysis" / "verify_controlled_release.py",
)
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def test_competence_counts_and_scoring() -> None:
    result = VERIFY.verify_competence(ROOT)
    assert result["correct"] == 317
    assert result["incorrect"] == 82
    assert result["exact_ties"] == 1
    assert result["tie_adjusted_accuracy"] == 0.79375
    assert result["strict_accuracy"] == 0.7925


def test_cross_learner_macros_are_self_contained() -> None:
    text = (
        ROOT / "macros" / "controlled_cross_learner_macros.tex"
    ).read_text(encoding="utf-8")
    assert r"\input{" not in text
    assert r"\newcommand{\ControlledWordDid}" in text
    required = (
        r"\newcommand{\ControlledNeuralCompetenceTieAdjustedAccuracy}"
        r"{79.38\xspace}"
    )
    assert required in text


def test_factorial_point_estimates_and_vector_counts() -> None:
    result = VERIFY.verify_factorials(ROOT)
    assert result["controlled_word"]["unique_prediction_vectors"] == 1
    assert result["controlled_character"]["unique_prediction_vectors"] == 1
    assert result["controlled_neural"]["unique_prediction_vectors"] == 3


def test_evidence_registry_paths_and_hashes() -> None:
    assert VERIFY.verify_registry(ROOT) >= 20


def test_required_figures_exist() -> None:
    assert len(VERIFY.verify_figures(ROOT)) == 5
