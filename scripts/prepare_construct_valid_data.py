#!/usr/bin/env python3
"""Recover construct-defining fields without changing a controlled split."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from ancestry_audit.canonical import raw_record_hash
from ancestry_audit.factorial import (
    edit_quality_pair,
    edit_training_pair,
    load_jsonl_gz,
)


FILES = {
    "train_naive_seed20260727.jsonl.gz": "edit",
    "train_safe_seed20260727.jsonl.gz": "edit",
    "train_naive_seed20260728.jsonl.gz": "edit",
    "train_safe_seed20260728.jsonl.gz": "edit",
    "train_naive_seed20260729.jsonl.gz": "edit",
    "train_safe_seed20260729.jsonl.gz": "edit",
    "target_512.jsonl.gz": "edit_quality",
    "clean_validation_512.jsonl.gz": "edit_quality",
    "untouched_clean_test_512.jsonl.gz": "edit_quality",
}


def source_path(root: Path, view: str, split: str) -> Path:
    """Resolve either the Hub layout or the earlier flat download layout."""
    nested = root / view / f"{split}.jsonl.gz"
    flat = root / f"{view}_{split}.jsonl.gz"
    if nested.is_file():
        return nested
    if flat.is_file():
        return flat
    raise FileNotFoundError(
        f"missing {view}/{split}.jsonl.gz under {root}"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def index_source(paths: list[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl_gz(path):
            identifier = raw_record_hash(row)
            previous = rows.setdefault(identifier, row)
            if previous != row:
                raise AssertionError(f"hash collision or inconsistent duplicate: {identifier}")
    return rows


def deterministic_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                line = (
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                zipped.write(line.encode("utf-8"))


def validate_construct_fields(row: dict[str, Any], identifier: str) -> None:
    context = row.get("context")
    original = row.get("original_response")
    feedback = row.get("feedback")
    if not isinstance(context, list) or not context:
        raise ValueError(f"{identifier}: missing structured context")
    if not isinstance(original, str) or not original.strip():
        raise ValueError(f"{identifier}: missing original response")
    if not isinstance(feedback, list) or not feedback:
        raise ValueError(f"{identifier}: missing feedback entries")
    if any(not isinstance(item, str) or not item.strip() for item in feedback):
        raise ValueError(f"{identifier}: empty or nontext feedback entry")


def recover(
    source_pairs: list[dict[str, Any]],
    index: dict[str, dict[str, Any]],
    view: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    transform = edit_training_pair if view == "edit" else edit_quality_pair
    for pair in source_pairs:
        identifier = pair["pair_id"]
        if identifier not in index:
            raise KeyError(f"{view} source row not found for pair {identifier}")
        source = index[identifier]
        validate_construct_fields(source, identifier)
        recreated = transform(source)
        if recreated != pair:
            raise AssertionError(f"frozen pair changed during recovery: {identifier}")
        augmented = dict(pair)
        augmented["original_response"] = source["original_response"]
        augmented["feedback"] = source["feedback"]
        augmented["construct_fields"] = [
            "context",
            "original_response",
            "feedback",
            "candidate_edit",
        ]
        output.append(augmented)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--helpsteer3-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    edit_paths = [
        source_path(args.helpsteer3_dir, "edit", split)
        for split in ("train", "validation")
    ]
    quality_paths = [
        source_path(args.helpsteer3_dir, "edit_quality", split)
        for split in ("train", "validation")
    ]
    edit_index = index_source(edit_paths)
    quality_index = index_source(quality_paths)

    args.output.mkdir(parents=True)
    manifest_files: dict[str, Any] = {}
    pair_ids_by_file: dict[str, list[str]] = {}
    for filename, view in FILES.items():
        input_path = args.input_dir / filename
        source_pairs = load_jsonl_gz(input_path)
        augmented = recover(
            source_pairs,
            edit_index if view == "edit" else quality_index,
            view,
        )
        output_path = args.output / filename
        deterministic_jsonl_gz(output_path, augmented)
        reloaded = load_jsonl_gz(output_path)
        if reloaded != augmented:
            raise AssertionError(f"round-trip mismatch: {filename}")
        pair_ids_by_file[filename] = [row["pair_id"] for row in augmented]
        manifest_files[filename] = {
            "view": view,
            "rows": len(augmented),
            "input_sha256": sha256(input_path),
            "output_sha256": sha256(output_path),
            "pair_order_sha256": hashlib.sha256(
                "\n".join(pair_ids_by_file[filename]).encode("utf-8")
            ).hexdigest(),
            "exact_frozen_pair_fields_match": True,
            "construct_fields_complete": True,
        }

    manifest = {
        "status": "complete",
        "protocol": "docs/controlled_factorial_protocol.md",
        "input_source": str(args.input_dir),
        "source_files": {
            str(path.relative_to(args.helpsteer3_dir)): sha256(path)
            for path in edit_paths + quality_paths
        },
        "source_index_rows": {
            "edit": len(edit_index),
            "edit_quality": len(quality_index),
        },
        "files": manifest_files,
        "guarantees": {
            "pair_id": "identical",
            "label_and_candidate_orientation": "identical",
            "file_order": "identical",
            "training_boundary_and_seed": "identical",
            "new_fields": ["original_response", "feedback"],
        },
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
