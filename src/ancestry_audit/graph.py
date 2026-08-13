"""Typed ancestry graph and connected-component construction."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .canonical import Atom, hash_value


@dataclass(frozen=True)
class RecordRef:
    dataset: str
    revision: str
    view: str
    split: str
    source_index: int
    record_hash: str
    atoms: tuple[Atom, ...]
    domain: str = "unknown"
    language: str = "unknown"
    label: str = "unknown"

    @property
    def record_id(self) -> str:
        return hash_value(
            {
                "dataset": self.dataset,
                "revision": self.revision,
                "view": self.view,
                "split": self.split,
                "source_index": self.source_index,
                "record_hash": self.record_hash,
            }
        )


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            next_value = self.parent[value]
            self.parent[value] = root
            value = next_value
        return root

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


class AncestryGraph:
    """In-memory exact typed ancestry graph.

    Atom indices use ``(atom_type, value_hash)``. Field names remain on edges
    for inspection but do not block cross-schema equality.
    """

    def __init__(self, records: Iterable[RecordRef]) -> None:
        self.records = tuple(records)
        self.atom_index: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(
            list
        )
        for record_index, record in enumerate(self.records):
            for atom in record.atoms:
                self.atom_index[(atom.atom_type, atom.value_hash)].append(
                    (record_index, atom.field)
                )
        self._component_ids: tuple[str, ...] | None = None

    @property
    def component_ids(self) -> tuple[str, ...]:
        if self._component_ids is None:
            self._component_ids = self._build_components()
        return self._component_ids

    def _build_components(self) -> tuple[str, ...]:
        union_find = _UnionFind(len(self.records))
        linkable = {
            (atom.atom_type, atom.value_hash)
            for record in self.records
            for atom in record.atoms
            if atom.linkable
        }
        for key in sorted(linkable):
            members = self.atom_index[key]
            if len(members) < 2:
                continue
            anchor = members[0][0]
            for record_index, _ in members[1:]:
                union_find.union(anchor, record_index)
        root_members: dict[int, list[str]] = defaultdict(list)
        for index, record in enumerate(self.records):
            root_members[union_find.find(index)].append(record.record_id)
        root_id = {
            root: hash_value(sorted(member_ids))
            for root, member_ids in root_members.items()
        }
        return tuple(root_id[union_find.find(i)] for i in range(len(self.records)))

    def component_summary(self) -> dict:
        sizes = Counter(self.component_ids)
        view_counts: dict[str, Counter[str]] = defaultdict(Counter)
        split_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for record, component_id in zip(self.records, self.component_ids):
            view_counts[component_id][record.view] += 1
            split_counts[component_id][record.split] += 1
        crossing = [
            component_id
            for component_id, counts in split_counts.items()
            if counts.get("train", 0) and counts.get("validation", 0)
        ]
        multi_view = [
            component_id
            for component_id, counts in view_counts.items()
            if len(counts) >= 2
        ]
        histogram = Counter(sizes.values())
        return {
            "records": len(self.records),
            "atoms": len(self.atom_index),
            "components": len(sizes),
            "largest_component_records": max(sizes.values(), default=0),
            "cross_split_components": len(crossing),
            "records_in_cross_split_components": sum(sizes[key] for key in crossing),
            "multi_view_components": len(multi_view),
            "records_in_multi_view_components": sum(sizes[key] for key in multi_view),
            "component_size_histogram": {
                str(size): count for size, count in sorted(histogram.items())
            },
        }

    def occurrences(
        self, atom_type: str, value_hash: str
    ) -> tuple[tuple[int, str], ...]:
        return tuple(self.atom_index.get((atom_type, value_hash), ()))
