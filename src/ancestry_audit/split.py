"""Deterministic component-disjoint split utility."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

from .graph import AncestryGraph


def component_folds(
    graph: AncestryGraph, *, folds: int = 20, seed: int = 20260725
) -> dict[str, int]:
    """Assign whole ancestry components to approximately balanced folds.

    Components are greedily assigned largest-first. The objective balances
    total records and per-view counts. Hashes provide deterministic tie breaks.
    """

    if folds < 2:
        raise ValueError("folds must be at least 2")
    component_records: dict[str, list[int]] = defaultdict(list)
    for index, component_id in enumerate(graph.component_ids):
        component_records[component_id].append(index)

    component_views = {
        component_id: Counter(graph.records[index].view for index in indices)
        for component_id, indices in component_records.items()
    }
    totals = [0] * folds
    view_totals: list[Counter[str]] = [Counter() for _ in range(folds)]
    targets = Counter(record.view for record in graph.records)
    for view in targets:
        targets[view] /= folds
    total_target = len(graph.records) / folds

    def tie_hash(component_id: str, fold: int) -> str:
        return hashlib.sha256(f"{seed}:{component_id}:{fold}".encode()).hexdigest()

    assignments = {}
    ordered = sorted(
        component_records,
        key=lambda key: (-len(component_records[key]), key),
    )
    for component_id in ordered:
        size = len(component_records[component_id])
        views = component_views[component_id]
        candidates = []
        for fold in range(folds):
            size_penalty = ((totals[fold] + size - total_target) / total_target) ** 2
            view_penalty = 0.0
            for view, count in views.items():
                target = max(float(targets[view]), 1.0)
                view_penalty += (
                    (view_totals[fold][view] + count - target) / target
                ) ** 2
            candidates.append((size_penalty + view_penalty, tie_hash(component_id, fold), fold))
        selected = min(candidates)[2]
        assignments[component_id] = selected
        totals[selected] += size
        view_totals[selected].update(views)
    return assignments


def assert_component_disjoint(
    graph: AncestryGraph, assignments: dict[str, int]
) -> None:
    seen = {}
    for component_id in graph.component_ids:
        fold = assignments[component_id]
        if component_id in seen and seen[component_id] != fold:
            raise AssertionError(f"component {component_id} crosses folds")
        seen[component_id] = fold
