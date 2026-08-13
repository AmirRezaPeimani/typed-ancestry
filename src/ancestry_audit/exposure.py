"""Exposure tensors and row-level ancestry paths."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .graph import AncestryGraph


@dataclass(frozen=True)
class ExposureRow:
    evaluation_record_index: int
    evaluation_view: str
    atom_type: str
    source_view: str
    exposed: bool
    matching_atoms: int
    matching_training_records: int


def exposure_rows(
    graph: AncestryGraph,
    *,
    training_split: str = "train",
    evaluation_split: str = "validation",
) -> tuple[ExposureRow, ...]:
    train_index: dict[tuple[str, str], dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    views: set[str] = set()
    atom_types: set[str] = set()

    for index, record in enumerate(graph.records):
        views.add(record.view)
        for atom in record.atoms:
            atom_types.add(atom.atom_type)
            if record.split == training_split:
                train_index[(atom.atom_type, atom.value_hash)][record.view].add(index)

    output = []
    for index, record in enumerate(graph.records):
        if record.split != evaluation_split:
            continue
        by_type: dict[str, list] = defaultdict(list)
        for atom in record.atoms:
            by_type[atom.atom_type].append(atom)
        for atom_type in sorted(atom_types):
            atoms = by_type.get(atom_type, [])
            for source_view in sorted(views):
                matching_atoms = 0
                matching_records: set[int] = set()
                for atom in atoms:
                    matches = train_index.get(
                        (atom.atom_type, atom.value_hash), {}
                    ).get(source_view, set())
                    if matches:
                        matching_atoms += 1
                        matching_records.update(matches)
                output.append(
                    ExposureRow(
                        evaluation_record_index=index,
                        evaluation_view=record.view,
                        atom_type=atom_type,
                        source_view=source_view,
                        exposed=bool(matching_records),
                        matching_atoms=matching_atoms,
                        matching_training_records=len(matching_records),
                    )
                )
    return tuple(output)


def summarize_exposure(
    graph: AncestryGraph, rows: Iterable[ExposureRow]
) -> list[dict]:
    rows = tuple(rows)
    record_counts = Counter(
        record.view for record in graph.records if record.split == "validation"
    )
    buckets: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    matching_atoms: Counter[tuple[str, str, str]] = Counter()
    matching_records: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        key = (row.evaluation_view, row.source_view, row.atom_type)
        if row.exposed:
            buckets[key].add(row.evaluation_record_index)
        matching_atoms[key] += row.matching_atoms
        matching_records[key] += row.matching_training_records

    output = []
    keys = sorted(
        {
            (row.evaluation_view, row.source_view, row.atom_type)
            for row in rows
        }
    )
    for evaluation_view, source_view, atom_type in keys:
        key = (evaluation_view, source_view, atom_type)
        denominator = record_counts[evaluation_view]
        count = len(buckets[key])
        output.append(
            {
                "evaluation_view": evaluation_view,
                "source_view": source_view,
                "atom_type": atom_type,
                "validation_rows": denominator,
                "exposed_rows": count,
                "exposure_rate": count / denominator if denominator else None,
                "matching_atoms": matching_atoms[key],
                "matching_training_records": matching_records[key],
            }
        )
    return output


def any_view_summary(
    graph: AncestryGraph, rows: Iterable[ExposureRow]
) -> list[dict]:
    rows = tuple(rows)
    record_counts = Counter(
        record.view for record in graph.records if record.split == "validation"
    )
    any_exposed: dict[tuple[str, str], set[int]] = defaultdict(set)
    same_exposed: dict[tuple[str, str], set[int]] = defaultdict(set)
    all_keys = {
        (row.evaluation_view, row.atom_type)
        for row in rows
    }
    for row in rows:
        if not row.exposed:
            continue
        key = (row.evaluation_view, row.atom_type)
        any_exposed[key].add(row.evaluation_record_index)
        if row.source_view == row.evaluation_view:
            same_exposed[key].add(row.evaluation_record_index)
    output = []
    for key in sorted(all_keys):
        evaluation_view, atom_type = key
        denominator = record_counts[evaluation_view]
        any_ids = any_exposed[key]
        same_ids = same_exposed[key]
        cross_only = any_ids - same_ids
        output.append(
            {
                "evaluation_view": evaluation_view,
                "atom_type": atom_type,
                "validation_rows": denominator,
                "same_view_exposed_rows": len(same_ids),
                "same_view_exposure_rate": len(same_ids) / denominator,
                "any_view_exposed_rows": len(any_ids),
                "any_view_exposure_rate": len(any_ids) / denominator,
                "cross_view_only_rows": len(cross_only),
                "cross_view_only_rate": len(cross_only) / denominator,
            }
        )
    return output
