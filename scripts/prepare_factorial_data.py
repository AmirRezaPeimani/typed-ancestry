#!/usr/bin/env python3
"""Create the frozen cross-view exposure intervention without model outcomes."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ancestry_audit.canonical import normalize_text, raw_record_hash
from ancestry_audit.factorial import (
    allocate_quotas,
    assignment_hash,
    context_hash,
    deduplicate_rows,
    directed_edit_hash,
    edit_link_hash,
    edit_quality_pair,
    edit_training_pair,
    length_features,
    load_jsonl_gz,
    match_stratum,
    one_row_per_context,
    stable_rank,
    standardized_mean_difference,
    total_variation,
    write_jsonl_gz,
)


SEED = 20260727
TRAINING_SEEDS = (20260727, 20260728, 20260729)
TARGET_N = 512
CLEAN_VALIDATION_N = 512
CLEAN_TEST_N = 512
BASE_N = 4000
FILLER_N = 512


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def read_release(directory: Path, view: str) -> tuple[list[dict[str, Any]], list[Path]]:
    paths = []
    for split in ("train", "validation"):
        nested = directory / view / f"{split}.jsonl.gz"
        flattened = directory / f"{view}_{split}.jsonl.gz"
        paths.append(nested if nested.is_file() else flattened)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(load_jsonl_gz(path))
    return rows, paths


def nearest_clean_pair(
    target: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_features = length_features(target)

    def distance(row: dict[str, Any]) -> tuple[float, str]:
        features = length_features(row)
        squared = sum(
            (left - right) ** 2
            for left, right in zip(target_features, features, strict=True)
        )
        return (
            squared,
            stable_rank(seed, "clean_nearest_tie", raw_record_hash(row)),
        )

    ordered = sorted(candidates, key=distance)
    if len(ordered) < 2:
        raise ValueError("A target stratum ran out of clean matches")
    return ordered[0], ordered[1]


def external_pairs(
    helpsteer2_validation: Path, contract_csv: Path
) -> list[dict[str, Any]]:
    rows = load_jsonl_gz(helpsteer2_validation)
    pairs: list[dict[str, Any]] = []
    with contract_csv.open(newline="", encoding="utf-8") as handle:
        for contract in csv.DictReader(handle):
            left = rows[int(contract["response1_index"])]
            right = rows[int(contract["response2_index"])]
            if normalize_text(left["prompt"]) != normalize_text(right["prompt"]):
                raise AssertionError("External pair prompt mismatch")
            label = int(contract["label_response2_preferred"])
            pairs.append(
                {
                    "pair_id": contract["pair_id"],
                    "source_view": "helpsteer2",
                    "domain": "general",
                    "language": "english",
                    "context": [{"role": "user", "content": left["prompt"]}],
                    "response1": left["response"],
                    "response2": right["response"],
                    "overall_preference": 1 if label else -1,
                }
            )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--helpsteer3-dir", type=Path, required=True)
    parser.add_argument("--helpsteer2-validation", type=Path, required=True)
    parser.add_argument("--helpsteer2-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    edit_rows_raw, edit_paths = read_release(args.helpsteer3_dir, "edit")
    quality_rows_raw, quality_paths = read_release(
        args.helpsteer3_dir, "edit_quality"
    )
    edit_rows = deduplicate_rows(edit_rows_raw)
    quality_rows = one_row_per_context(
        deduplicate_rows(quality_rows_raw),
        seed=SEED,
        namespace="quality_context_representative",
    )

    edits_by_link: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    edit_contexts: set[str] = set()
    for row in edit_rows:
        edits_by_link[edit_link_hash(row, "edited_response")].append(row)
        edit_contexts.add(context_hash(row))
    for rows in edits_by_link.values():
        rows.sort(key=raw_record_hash)

    target_by_stratum: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    clean_by_stratum: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    target_donor: dict[str, dict[str, Any]] = {}
    donor_ids_seen: set[str] = set()

    target_candidates = sorted(
        (
            row
            for row in quality_rows
            if edit_link_hash(row, "good_edited_response") in edits_by_link
        ),
        key=lambda row: stable_rank(
            SEED, "target_candidate", raw_record_hash(row)
        ),
    )
    for row in target_candidates:
        row_id = raw_record_hash(row)
        donors = edits_by_link[edit_link_hash(row, "good_edited_response")]
        donor = min(
            donors,
            key=lambda item: stable_rank(
                SEED, f"donor_for:{row_id}", raw_record_hash(item)
            ),
        )
        donor_id = raw_record_hash(donor)
        if donor_id in donor_ids_seen:
            continue
        donor_ids_seen.add(donor_id)
        target_donor[row_id] = donor
        target_by_stratum[match_stratum(row)].append(row)

    for row in quality_rows:
        if (
            context_hash(row) not in edit_contexts
            and edit_link_hash(row, "good_edited_response") not in edits_by_link
            and edit_link_hash(row, "bad_edited_response") not in edits_by_link
        ):
            clean_by_stratum[match_stratum(row)].append(row)

    capacities = {
        stratum: min(
            len(target_by_stratum[stratum]),
            len(clean_by_stratum[stratum]) // 2,
        )
        for stratum in set(target_by_stratum) | set(clean_by_stratum)
    }
    quotas = allocate_quotas(capacities, TARGET_N)

    targets: list[dict[str, Any]] = []
    clean_validation: list[dict[str, Any]] = []
    clean_test: list[dict[str, Any]] = []
    for stratum in sorted(quotas):
        target_pool = sorted(
            target_by_stratum[stratum],
            key=lambda row: stable_rank(
                SEED, f"target:{stratum}", raw_record_hash(row)
            ),
        )
        clean_pool = list(clean_by_stratum[stratum])
        selected_targets = target_pool[: quotas[stratum]]
        for target in selected_targets:
            left, right = nearest_clean_pair(target, clean_pool, seed=SEED)
            clean_pool.remove(left)
            clean_pool.remove(right)
            targets.append(target)
            if (
                int(stable_rank(SEED, "clean_arm", raw_record_hash(target))[:8], 16)
                % 2
            ):
                clean_validation.append(left)
                clean_test.append(right)
            else:
                clean_validation.append(right)
                clean_test.append(left)

    targets.sort(key=raw_record_hash)
    clean_validation.sort(key=raw_record_hash)
    clean_test.sort(key=raw_record_hash)
    if not (
        len(targets)
        == len(clean_validation)
        == len(clean_test)
        == TARGET_N
    ):
        raise AssertionError("Evaluation assignment size mismatch")

    evaluation_contexts = {
        context_hash(row)
        for row in targets + clean_validation + clean_test
    }
    eligible_training = [
        row for row in edit_rows if context_hash(row) not in evaluation_contexts
    ]
    eligible_training.sort(
        key=lambda row: stable_rank(SEED, "eligible_train", raw_record_hash(row))
    )
    if len(eligible_training) < BASE_N + FILLER_N:
        raise ValueError("Insufficient component-disjoint training records")
    base = eligible_training[:BASE_N]
    filler = eligible_training[BASE_N : BASE_N + FILLER_N]
    donors = [target_donor[raw_record_hash(row)] for row in targets]

    # The controlled difference is exactly the 512-row boundary block.
    base_ids = {raw_record_hash(row) for row in base}
    filler_ids = {raw_record_hash(row) for row in filler}
    donor_ids = {raw_record_hash(row) for row in donors}
    if base_ids & (filler_ids | donor_ids) or filler_ids & donor_ids:
        raise AssertionError("Training blocks are not disjoint")

    target_pairs = [edit_quality_pair(row) for row in targets]
    clean_validation_pairs = [
        edit_quality_pair(row) for row in clean_validation
    ]
    clean_test_pairs = [edit_quality_pair(row) for row in clean_test]
    external = external_pairs(
        args.helpsteer2_validation, args.helpsteer2_contract
    )

    target_features = [length_features(row) for row in targets]
    clean_features = [length_features(row) for row in clean_validation]
    balance = {
        "domain_total_variation": total_variation(
            [str(row.get("domain", "unknown")) for row in targets],
            [str(row.get("domain", "unknown")) for row in clean_validation],
        ),
        "preferred_log_length_smd": standardized_mean_difference(
            [features[0] for features in target_features],
            [features[0] for features in clean_features],
        ),
        "rejected_log_length_smd": standardized_mean_difference(
            [features[1] for features in target_features],
            [features[1] for features in clean_features],
        ),
    }
    balance["meets_balance_criteria"] = (
        balance["domain_total_variation"] <= 0.05
        and abs(balance["preferred_log_length_smd"]) <= 0.15
        and abs(balance["rejected_log_length_smd"]) <= 0.15
    )
    if not balance["meets_balance_criteria"]:
        raise RuntimeError(
            "Data-only balance criteria were not met before model fitting: "
            + json.dumps(balance, sort_keys=True)
        )

    args.output.mkdir(parents=True)
    write_jsonl_gz(args.output / "target_512.jsonl.gz", target_pairs)
    write_jsonl_gz(
        args.output / "clean_validation_512.jsonl.gz",
        clean_validation_pairs,
    )
    write_jsonl_gz(
        args.output / "untouched_clean_test_512.jsonl.gz", clean_test_pairs
    )
    write_jsonl_gz(args.output / "external_helpsteer2.jsonl.gz", external)

    base_pairs = [edit_training_pair(row) for row in base]
    donor_pairs = [edit_training_pair(row) for row in donors]
    filler_pairs = [edit_training_pair(row) for row in filler]
    slot_ids = [f"base:{index}" for index in range(BASE_N)] + [
        f"boundary:{index}" for index in range(FILLER_N)
    ]
    training_manifests: dict[str, Any] = {}
    for training_seed in TRAINING_SEEDS:
        slots = sorted(
            slot_ids,
            key=lambda slot: stable_rank(training_seed, "train_slot", slot),
        )
        naive: list[dict[str, Any]] = []
        safe: list[dict[str, Any]] = []
        boundary_seen_at_fraction: Counter[str] = Counter()
        for position, slot in enumerate(slots):
            block, raw_index = slot.split(":")
            index = int(raw_index)
            if block == "base":
                naive.append(base_pairs[index])
                safe.append(base_pairs[index])
            else:
                naive.append(donor_pairs[index])
                safe.append(filler_pairs[index])
                fraction = min(5, (position * 5) // len(slots) + 1)
                boundary_seen_at_fraction[str(fraction * 20)] += 1
        naive_path = args.output / f"train_naive_seed{training_seed}.jsonl.gz"
        safe_path = args.output / f"train_safe_seed{training_seed}.jsonl.gz"
        write_jsonl_gz(naive_path, naive)
        write_jsonl_gz(safe_path, safe)
        training_manifests[str(training_seed)] = {
            "naive_file": naive_path.name,
            "safe_file": safe_path.name,
            "naive_assignment_sha256": assignment_hash(naive),
            "safe_assignment_sha256": assignment_hash(safe),
            "naive_file_sha256": sha256(naive_path),
            "safe_file_sha256": sha256(safe_path),
            "boundary_rows_by_checkpoint_interval": dict(
                sorted(boundary_seen_at_fraction.items(), key=lambda item: int(item[0]))
            ),
        }

    context_sets = {
        "target": {context_hash(row) for row in targets},
        "clean_validation": {context_hash(row) for row in clean_validation},
        "untouched_clean_test": {context_hash(row) for row in clean_test},
        "base": {context_hash(row) for row in base},
        "filler": {context_hash(row) for row in filler},
        "donor": {context_hash(row) for row in donors},
    }
    crossings: dict[str, int] = {}
    names = sorted(context_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            crossings[f"{left}__{right}"] = len(
                context_sets[left] & context_sets[right]
            )
    allowed_crossings = {"base__filler", "donor__target"}
    unexpected_crossings = {
        key: value
        for key, value in crossings.items()
        if value and key not in allowed_crossings
    }
    if unexpected_crossings:
        raise AssertionError(
            f"Unexpected context crossing: {unexpected_crossings}"
        )
    if crossings.get("donor__target") != TARGET_N:
        raise AssertionError("Every target must have exactly one donor context")

    source_paths = edit_paths + quality_paths + [
        args.helpsteer2_validation,
        args.helpsteer2_contract,
    ]
    manifest = {
        "status": "complete",
        "protocol": "docs/controlled_factorial_protocol.md",
        "randomization_seed": SEED,
        "training_seeds": list(TRAINING_SEEDS),
        "source_files": {
            str(path.resolve()): sha256(path) for path in source_paths
        },
        "source_counts": {
            "edit_rows_raw": len(edit_rows_raw),
            "edit_rows_unique": len(edit_rows),
            "edit_quality_rows_raw": len(quality_rows_raw),
            "edit_quality_context_representatives": len(quality_rows),
            "target_candidates_with_unique_donors": sum(
                len(rows) for rows in target_by_stratum.values()
            ),
            "strict_clean_candidates": sum(
                len(rows) for rows in clean_by_stratum.values()
            ),
        },
        "assignment_counts": {
            "target": len(target_pairs),
            "clean_validation": len(clean_validation_pairs),
            "untouched_clean_test": len(clean_test_pairs),
            "external": len(external),
            "shared_base": len(base_pairs),
            "naive_ancestor_block": len(donor_pairs),
            "safe_filler_block": len(filler_pairs),
            "training_per_condition": BASE_N + FILLER_N,
        },
        "stratum_capacities": dict(sorted(capacities.items())),
        "stratum_quotas": dict(sorted(quotas.items())),
        "balance": balance,
        "context_crossings": crossings,
        "training_files": training_manifests,
        "evaluation_files": {},
    }
    for path in sorted(args.output.glob("*.jsonl.gz")):
        if not path.name.startswith("train_"):
            manifest["evaluation_files"][path.name] = sha256(path)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
