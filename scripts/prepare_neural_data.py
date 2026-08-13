#!/usr/bin/env python3
"""Prepare component-disjoint construct-valid data for the neural study."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from ancestry_audit.factorial import (
    context_hash,
    edit_quality_pair,
    load_jsonl_gz,
    raw_record_hash,
    stable_rank,
)


SEEDS = (20260727, 20260728, 20260729)
SHARED_N = 4000
TASK_BASE_N = 1000
RETAINED_EDIT_N = 3000
SELECTOR_TRAIN_N = 1000
DEVELOPMENT_N = 400
SELECTION_SEED = 20260801


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                zipped.write(
                    (
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                )


def enrich_quality(row: dict[str, Any]) -> dict[str, Any]:
    pair = edit_quality_pair(row)
    pair["original_response"] = row["original_response"]
    pair["feedback"] = row["feedback"]
    pair["construct_fields"] = [
        "context",
        "original_response",
        "feedback",
        "candidate_edit",
    ]
    return pair


def read_components(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            record_hash = row["record_hash"]
            component = row["component_id"]
            previous = mapping.setdefault(record_hash, component)
            if previous != component:
                raise ValueError(f"inconsistent component for {record_hash}")
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edit-quality-train", type=Path, required=True)
    parser.add_argument("--factorial-data", type=Path, required=True)
    parser.add_argument("--component-assignments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    components = read_components(args.component_assignments)
    evaluation_files = [
        args.factorial_data / "target_512.jsonl.gz",
        args.factorial_data / "clean_validation_512.jsonl.gz",
        args.factorial_data / "untouched_clean_test_512.jsonl.gz",
    ]
    excluded_hashes: set[str] = set()
    excluded_contexts: set[str] = set()
    for path in evaluation_files:
        for row in load_jsonl_gz(path):
            excluded_hashes.add(row["pair_id"])
            excluded_contexts.add(context_hash(row))

    old_runs: dict[tuple[int, str], list[dict[str, Any]]] = {}
    boundary_positions: dict[int, list[int]] = {}
    shared_pair_ids: set[str] | None = None
    shared_rows_by_id: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        naive = load_jsonl_gz(
            args.factorial_data / f"train_naive_seed{seed}.jsonl.gz"
        )
        safe = load_jsonl_gz(
            args.factorial_data / f"train_safe_seed{seed}.jsonl.gz"
        )
        if len(naive) != SHARED_N + 512 or len(safe) != SHARED_N + 512:
            raise ValueError(f"unexpected training size for seed {seed}")
        positions = [
            index
            for index, (left, right) in enumerate(zip(naive, safe, strict=True))
            if left["pair_id"] != right["pair_id"]
        ]
        if len(positions) != 512:
            raise ValueError(
                f"expected 512 boundary positions for seed {seed}; "
                f"found {len(positions)}"
            )
        for index in positions:
            for row in (naive[index], safe[index]):
                excluded_hashes.add(row["pair_id"])
                excluded_contexts.add(context_hash(row))
        seed_shared = {
            left["pair_id"]
            for left, right in zip(naive, safe, strict=True)
            if left["pair_id"] == right["pair_id"]
        }
        if len(seed_shared) != SHARED_N:
            raise ValueError(f"unexpected shared-pair count for seed {seed}")
        if shared_pair_ids is None:
            shared_pair_ids = seed_shared
        elif seed_shared != shared_pair_ids:
            raise ValueError("shared training examples differ across seeds")
        for row in naive:
            if row["pair_id"] in seed_shared:
                shared_rows_by_id.setdefault(row["pair_id"], row)
        old_runs[(seed, "naive")] = naive
        old_runs[(seed, "safe")] = safe
        boundary_positions[seed] = positions

    excluded_components = {
        components[record_hash]
        for record_hash in excluded_hashes
        if record_hash in components
    }
    missing_excluded = sorted(
        record_hash
        for record_hash in excluded_hashes
        if record_hash not in components
    )
    if missing_excluded:
        raise KeyError(
            f"{len(missing_excluded)} excluded records lack component assignments"
        )

    by_component_context: dict[str, dict[str, dict[str, Any]]] = {}
    raw_rows = load_jsonl_gz(args.edit_quality_train)
    for row in raw_rows:
        record_hash = raw_record_hash(row)
        if record_hash not in components:
            raise KeyError(f"source row lacks component assignment: {record_hash}")
        component = components[record_hash]
        if component in excluded_components or context_hash(row) in excluded_contexts:
            continue
        normalized_context = context_hash(row)
        component_rows = by_component_context.setdefault(component, {})
        incumbent = component_rows.get(normalized_context)
        if incumbent is None or stable_rank(
            SELECTION_SEED, "within_component", record_hash
        ) < stable_rank(
            SELECTION_SEED, "within_component", raw_record_hash(incumbent)
        ):
            component_rows[normalized_context] = row

    if shared_pair_ids is None:
        raise AssertionError("shared pair set was not initialized")
    shared_components = {components[pair_id] for pair_id in shared_pair_ids}
    context_rows = [
        row
        for rows in by_component_context.values()
        for row in rows.values()
    ]
    development_candidates = [
        row
        for row in context_rows
        if components[raw_record_hash(row)] not in shared_components
    ]
    development_candidates.sort(
        key=lambda row: stable_rank(
            SELECTION_SEED, "task_matched_development", raw_record_hash(row)
        )
    )
    if len(development_candidates) < DEVELOPMENT_N:
        raise RuntimeError(
            f"need {DEVELOPMENT_N} clean development rows; "
            f"found {len(development_candidates)}"
        )
    development_source = development_candidates[:DEVELOPMENT_N]
    development_component_ids = {
        components[raw_record_hash(row)] for row in development_source
    }

    task_candidates = [
        row
        for row in context_rows
        if components[raw_record_hash(row)] not in development_component_ids
    ]
    task_candidates.sort(
        key=lambda row: stable_rank(
            SELECTION_SEED, "task_matched_training", raw_record_hash(row)
        )
    )
    if len(task_candidates) < TASK_BASE_N:
        raise RuntimeError(
            f"need {TASK_BASE_N} task training rows; found {len(task_candidates)}"
        )
    base_source = task_candidates[:TASK_BASE_N]
    base_component_ids = {
        components[raw_record_hash(row)] for row in base_source
    }
    if development_component_ids & (
        base_component_ids | shared_components
    ):
        raise AssertionError("development component occurs in training")

    base = [enrich_quality(row) for row in base_source]
    development = [enrich_quality(row) for row in development_source]
    base_components = {components[row["pair_id"]] for row in base}
    development_components = {
        components[row["pair_id"]] for row in development
    }
    if base_components & development_components:
        raise AssertionError("task base and development components overlap")
    if (base_components | development_components) & excluded_components:
        raise AssertionError("selector data overlap excluded factorial components")

    args.output.mkdir(parents=True)
    files: dict[str, dict[str, Any]] = {}

    def save(name: str, rows: list[dict[str, Any]], role: str) -> None:
        path = args.output / name
        write_jsonl_gz(path, rows)
        files[name] = {
            "role": role,
            "rows": len(rows),
            "sha256": sha256(path),
            "pair_order_sha256": hashlib.sha256(
                "\n".join(row["pair_id"] for row in rows).encode("utf-8")
            ).hexdigest(),
        }

    save(
        "selector_train_1000.jsonl.gz",
        base[:SELECTOR_TRAIN_N],
        "checkpoint_selector_train",
    )
    save(
        "selector_development_400.jsonl.gz",
        development,
        "checkpoint_selector_development",
    )
    save("task_base_1000.jsonl.gz", base, "task_factorial_base")

    retained_edit_ids = sorted(
        shared_pair_ids,
        key=lambda pair_id: stable_rank(
            SELECTION_SEED, "retained_edit_training", pair_id
        ),
    )[:RETAINED_EDIT_N]
    retained_edit = [shared_rows_by_id[pair_id] for pair_id in retained_edit_ids]
    save(
        "retained_edit_base_3000.jsonl.gz",
        retained_edit,
        "retained_edit_factorial_base",
    )
    shared_mixture = base + retained_edit
    if len(shared_mixture) != SHARED_N:
        raise AssertionError("shared mixture size changed")

    for seed in SEEDS:
        ordered_base = sorted(
            shared_mixture,
            key=lambda row: stable_rank(
                seed, "task_matched_base_order", row["pair_id"]
            ),
        )
        boundary_set = set(boundary_positions[seed])
        shared_positions = [
            index for index in range(SHARED_N + 512) if index not in boundary_set
        ]
        if len(shared_positions) != SHARED_N:
            raise AssertionError("shared position count changed")
        for boundary in ("naive", "safe"):
            rebuilt = [dict(row) for row in old_runs[(seed, boundary)]]
            for index, row in zip(shared_positions, ordered_base, strict=True):
                rebuilt[index] = row
            other = "safe" if boundary == "naive" else "naive"
            for index in boundary_positions[seed]:
                if (
                    rebuilt[index]["pair_id"]
                    != old_runs[(seed, boundary)][index]["pair_id"]
                ):
                    raise AssertionError("boundary-specific row changed")
                if (
                    old_runs[(seed, boundary)][index]["pair_id"]
                    == old_runs[(seed, other)][index]["pair_id"]
                ):
                    raise AssertionError("boundary intervention disappeared")
            save(
                f"train_{boundary}_seed{seed}.jsonl.gz",
                rebuilt,
                f"{boundary}_factorial_train",
            )

    for path in evaluation_files:
        rows = load_jsonl_gz(path)
        save(path.name, rows, "sealed_evaluation")

    manifest = {
        "status": "complete",
        "protocol": (
            "docs/controlled_factorial_protocol.md"
        ),
        "selection_seed": SELECTION_SEED,
        "source": {
            "dataset": "nvidia/HelpSteer3",
            "view": "edit_quality",
            "split": "train",
            "path": str(args.edit_quality_train),
            "sha256": sha256(args.edit_quality_train),
            "rows": len(raw_rows),
        },
        "component_assignments": {
            "path": str(args.component_assignments),
            "sha256": sha256(args.component_assignments),
        },
        "selection": {
            "eligible_components": len(by_component_context),
            "eligible_unique_contexts": len(context_rows),
            "shared_base_pairs": SHARED_N,
            "task_base_pairs": TASK_BASE_N,
            "retained_edit_pairs": RETAINED_EDIT_N,
            "selector_train_pairs": SELECTOR_TRAIN_N,
            "development_pairs": DEVELOPMENT_N,
            "excluded_components": len(excluded_components),
            "base_components": len(base_component_ids),
            "development_components": len(development_component_ids),
            "base_development_component_overlap": 0,
            "task_factorial_component_overlap": 0,
        },
        "intervention": {
            "shared_positions_per_seed": SHARED_N,
            "boundary_positions_per_seed": 512,
            "boundary_rows_and_positions_preserved": True,
            "evaluation_rows_preserved": True,
        },
        "files": files,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
