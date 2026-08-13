"""Command-line interface."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

from . import __version__
from .adapters import helpsteer3_sources, records_from_source
from .exposure import any_view_summary, exposure_rows, summarize_exposure
from .graph import AncestryGraph
from .io import file_sha256, write_csv, write_json
from .split import assert_component_disjoint, component_folds


def _scan_helpsteer3(args: argparse.Namespace) -> None:
    output: Path = args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)

    sources = helpsteer3_sources(args.dataset_dir, args.revision)
    records = []
    files = []
    for source in sources:
        records.extend(records_from_source(source))
        files.append(
            {
                "view": source.view,
                "split": source.split,
                "path": str(source.path.resolve()),
                "bytes": source.path.stat().st_size,
                "sha256": file_sha256(source.path),
            }
        )
    graph = AncestryGraph(records)
    rows = exposure_rows(graph)
    summary = summarize_exposure(graph, rows)
    any_summary = any_view_summary(graph, rows)

    write_csv(output / "exposure_by_source_view.csv", summary)
    write_csv(output / "exposure_same_any_view.csv", any_summary)
    write_json(output / "component_summary.json", graph.component_summary())

    folds = component_folds(graph, folds=args.folds, seed=args.seed)
    assert_component_disjoint(graph, folds)
    assignment_rows = [
        {
            "record_id": record.record_id,
            "record_hash": record.record_hash,
            "view": record.view,
            "original_split": record.split,
            "source_index": record.source_index,
            "component_id": component_id,
            "component_fold": folds[component_id],
        }
        for record, component_id in zip(graph.records, graph.component_ids)
    ]
    write_csv(output / "component_assignments.csv", assignment_rows)

    # Deterministic, inspectable cross-view link queue. Atom values are not
    # copied into the queue; source indices resolve to the pinned files.
    queue = []
    for (atom_type, value_hash), occurrences in sorted(graph.atom_index.items()):
        train = [
            (index, field)
            for index, field in occurrences
            if graph.records[index].split == "train"
        ]
        validation = [
            (index, field)
            for index, field in occurrences
            if graph.records[index].split == "validation"
        ]
        if not train or not validation:
            continue
        for validation_index, validation_field in validation:
            source_views = sorted(
                {
                    graph.records[train_index].view
                    for train_index, _ in train
                    if graph.records[train_index].view
                    != graph.records[validation_index].view
                }
            )
            if not source_views:
                continue
            cross_view_train = [
                item
                for item in train
                if graph.records[item[0]].view
                != graph.records[validation_index].view
            ]
            first_train_index, first_train_field = min(
                cross_view_train,
                key=lambda item: (
                    graph.records[item[0]].view,
                    graph.records[item[0]].source_index,
                    item[1],
                ),
            )
            queue.append(
                {
                    "atom_type": atom_type,
                    "value_hash": value_hash,
                    "validation_view": graph.records[validation_index].view,
                    "validation_source_index": graph.records[
                        validation_index
                    ].source_index,
                    "validation_field": validation_field,
                    "source_views": "|".join(source_views),
                    "example_train_view": graph.records[first_train_index].view,
                    "example_train_source_index": graph.records[
                        first_train_index
                    ].source_index,
                    "example_train_field": first_train_field,
                }
            )
    queue.sort(
        key=lambda row: (
            row["atom_type"],
            row["validation_view"],
            row["value_hash"],
        )
    )
    if queue:
        write_csv(output / "cross_view_inspection_queue.csv", queue)

    output_files = sorted(output.glob("*"))
    manifest = {
        "tool": "multiview-ancestry-audit",
        "tool_version": __version__,
        "dataset": "nvidia/HelpSteer3",
        "revision": args.revision,
        "seed": args.seed,
        "folds": args.folds,
        "python": sys.version,
        "platform": platform.platform(),
        "source_files": files,
        "records": len(graph.records),
        "outputs": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in output_files
            if path.name != "manifest.json"
        ],
    }
    write_json(output / "manifest.json", manifest)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ancestry-audit")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    scan = subparsers.add_parser("scan", help="scan a pinned multi-view release")
    scan.add_argument(
        "--adapter", choices=("helpsteer3",), default="helpsteer3"
    )
    scan.add_argument("--dataset-dir", type=Path, required=True)
    scan.add_argument("--revision", required=True)
    scan.add_argument("--output", type=Path, required=True)
    scan.add_argument("--seed", type=int, default=20260725)
    scan.add_argument("--folds", type=int, default=20)
    scan.set_defaults(func=_scan_helpsteer3)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
