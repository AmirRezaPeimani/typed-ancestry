#!/usr/bin/env python3
"""Build a portable claim-to-evidence registry with file checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence(root: Path, path: str, role: str) -> dict[str, Any]:
    target = root / path
    if not target.is_file():
        raise FileNotFoundError(target)
    return {
        "path": path,
        "role": role,
        "bytes": target.stat().st_size,
        "sha256": sha256(target),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    root_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root_default)
    args = parser.parse_args()
    root = args.root.resolve()

    claims = [
        {
            "claim_id": "controlled_sparse_distortion",
            "claim": (
                "The controlled target-minus-clean distortion is positive "
                "for the two sparse learners."
            ),
            "evidence": [
                evidence(
                    root,
                    "results/controlled_sparse_factorial/summary.json",
                    "point estimates and intervals",
                ),
                evidence(
                    root,
                    "results/controlled_sparse_factorial/predictions.csv.gz",
                    "row-level saved predictions",
                ),
                evidence(
                    root,
                    "tables/controlled_cross_learner_factorial.csv",
                    "table-ready estimates",
                ),
            ],
        },
        {
            "claim_id": "controlled_neural_null",
            "claim": (
                "The controlled Qwen estimate is near zero and differs from "
                "both sparse learner estimates under the fixed protocol."
            ),
            "evidence": [
                evidence(
                    root,
                    "results/controlled_qwen15_factorial/summary.json",
                    "point estimate, interval, and interactions",
                ),
                evidence(
                    root,
                    "results/controlled_qwen15_factorial/predictions.csv.gz",
                    "three-seed row-level saved predictions",
                ),
                evidence(
                    root,
                    "tables/controlled_learner_interactions.csv",
                    "table-ready interaction estimates",
                ),
                evidence(
                    root,
                    "figures/controlled_cross_learner_factorial_main.pdf",
                    "publication figure",
                ),
            ],
        },
        {
            "claim_id": "neural_competence_scoring",
            "claim": (
                "The selected checkpoint scored 317 correct, 82 incorrect, "
                "and one exact tie on 400 pairs: 79.38% with half tie credit "
                "and 79.25% strictly."
            ),
            "evidence": [
                evidence(
                    root,
                    "results/controlled_qwen15_factorial/competence_predictions.csv",
                    "all 400 scored pairs",
                ),
                evidence(
                    root,
                    "results/controlled_qwen15_factorial/competence.json",
                    "explicit scoring summary",
                ),
                evidence(
                    root,
                    "tables/controlled_neural_competence.csv",
                    "table-ready scoring summary",
                ),
                evidence(
                    root,
                    "macros/controlled_neural_macros.tex",
                    "paper value macros",
                ),
            ],
        },
        {
            "claim_id": "normalization_and_selection",
            "claim": (
                "Typed exposure is stable under the audited exact "
                "normalizations, while exposed and clean validation select "
                "different predefined candidates."
            ),
            "evidence": [
                evidence(
                    root,
                    "results/ancestry_robustness/normalization_results.csv",
                    "normalization sensitivity rows",
                ),
                evidence(
                    root,
                    "results/model_selection/development_candidates.csv",
                    "candidate validation scores",
                ),
                evidence(
                    root,
                    "results/model_selection/uncertainty_validation.json",
                    "held-out uncertainty audit",
                ),
                evidence(
                    root,
                    "figures/candidate_selection_main.pdf",
                    "prospective candidate-selection figure",
                ),
                evidence(
                    root,
                    "figures/candidate_selection_supp.pdf",
                    "validation-weighting sensitivity figure",
                ),
            ],
        },
        {
            "claim_id": "learning_dynamics",
            "claim": (
                "The lexical boundary effect grows over training and the "
                "randomized-arrival analysis separates target ancestry from "
                "a clean placebo contrast."
            ),
            "evidence": [
                evidence(
                    root,
                    "results/learning_dynamics/learning_dynamics.csv",
                    "checkpoint-level estimates",
                ),
                evidence(
                    root,
                    "results/ancestor_arrival/summary.json",
                    "randomized-arrival estimates",
                ),
                evidence(
                    root,
                    "figures/learning_dynamics_main.pdf",
                    "typed-boundary measurement figure",
                ),
                evidence(
                    root,
                    "figures/learning_dynamics_acquisition.pdf",
                    "controlled acquisition figure",
                ),
                evidence(
                    root,
                    "results/ancestor_arrival/event_aligned_estimates.csv",
                    "event-aligned estimates",
                ),
            ],
        },
        {
            "claim_id": "dataset_design",
            "claim": (
                "The controlled design holds the shared base and evaluation "
                "sets fixed while replacing a 512-pair ancestor block with "
                "a component-disjoint filler block."
            ),
            "evidence": [
                evidence(
                    root,
                    "tables/dataset_split_design.csv",
                    "table-ready split design",
                ),
                evidence(
                    root,
                    "results/dataset_design/summary.json",
                    "assignment counts and data hashes",
                ),
                evidence(
                    root,
                    "figures/learning_dynamics_main.pdf",
                    "typed-ancestry design and exposure figure",
                ),
            ],
        },
    ]
    files: dict[str, dict[str, Any]] = {}
    for claim in claims:
        for item in claim["evidence"]:
            files[item["path"]] = {
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
    payload = {
        "status": "complete",
        "path_policy": "All paths are relative to the artifact root.",
        "claims": claims,
        "files": files,
    }
    output = root / "evidence" / "evidence_registry.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
