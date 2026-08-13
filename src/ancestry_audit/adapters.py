"""Dataset adapters with immutable source metadata."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .canonical import extract_atoms, raw_record_hash
from .graph import RecordRef


HELPSTEER3_VIEWS = (
    "preference",
    "feedback",
    "edit",
    "principle",
    "edit_quality",
)


@dataclass(frozen=True)
class SourceFile:
    dataset: str
    revision: str
    view: str
    split: str
    path: Path


def iter_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def helpsteer3_sources(
    dataset_dir: Path, revision: str
) -> tuple[SourceFile, ...]:
    sources = []
    for view in HELPSTEER3_VIEWS:
        for split in ("train", "validation"):
            nested = dataset_dir / view / f"{split}.jsonl.gz"
            flattened = dataset_dir / f"{view}_{split}.jsonl.gz"
            path = nested if nested.is_file() else flattened
            if not path.is_file():
                raise FileNotFoundError(
                    f"expected {nested} or {flattened}"
                )
            sources.append(
                SourceFile(
                    dataset="nvidia/HelpSteer3",
                    revision=revision,
                    view=view,
                    split=split,
                    path=path,
                )
            )
    return tuple(sources)


def records_from_source(source: SourceFile) -> Iterator[RecordRef]:
    for source_index, row in enumerate(iter_jsonl_gz(source.path)):
        yield RecordRef(
            dataset=source.dataset,
            revision=source.revision,
            view=source.view,
            split=source.split,
            source_index=source_index,
            record_hash=raw_record_hash(row),
            atoms=extract_atoms(source.view, row),
            domain=str(row.get("domain", "unknown")),
            language=str(row.get("language", "unknown")),
            label=_label(source.view, row),
        )


def _label(view: str, row: dict[str, Any]) -> str:
    if view == "preference":
        value = int(row.get("overall_preference", 0))
        return "tie" if value == 0 else ("response2" if value > 0 else "response1")
    if view == "principle":
        return str(row.get("fulfilment", "unknown"))
    return "unlabeled"
