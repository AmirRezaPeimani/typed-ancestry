"""Public adapter interface for manifest-defined multi-view datasets."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

from .canonical import Atom, hash_value, normalize_text, raw_record_hash
from .graph import RecordRef


@dataclass(frozen=True)
class SourceSpec:
    dataset: str
    revision: str
    view: str
    split: str
    path: Path
    context_field: str | None
    response_fields: tuple[str, ...]
    feedback_fields: tuple[str, ...]
    edit_pairs: tuple[tuple[str, str], ...]
    domain_field: str | None = None
    language_field: str | None = None
    label_field: str | None = None


class DatasetAdapter(Protocol):
    """Minimal interface consumed by ancestry scanners."""

    @property
    def dataset(self) -> str: ...

    @property
    def revision(self) -> str: ...

    def sources(self) -> tuple[SourceSpec, ...]: ...

    def records(self, source: SourceSpec) -> Iterator[RecordRef]: ...


def _atom(
    atom_type: str, value: Any, field: str, *, linkable: bool = True
) -> Atom:
    return Atom(
        atom_type=atom_type,
        value_hash=hash_value(value),
        field=field,
        linkable=linkable,
    )


def _string_items(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        yield from (item for item in value if isinstance(item, str))


def extract_manifest_atoms(
    row: dict[str, Any], source: SourceSpec
) -> tuple[Atom, ...]:
    """Extract typed atoms under an explicit, reviewable field mapping."""

    atoms: set[Atom] = {_atom("record", row, "__record__")}
    if source.context_field:
        value = row.get(source.context_field)
        if value not in (None, "", []):
            atoms.add(_atom("context", value, source.context_field))
    for field in source.response_fields:
        value = row.get(field)
        if isinstance(value, str) and normalize_text(value):
            atoms.add(_atom("response", value, field, linkable=False))
    for field in source.feedback_fields:
        for index, value in enumerate(_string_items(row.get(field))):
            if normalize_text(value):
                atoms.add(
                    _atom(
                        "feedback",
                        value,
                        f"{field}[{index}]",
                        linkable=False,
                    )
                )
    for original_field, edited_field in source.edit_pairs:
        original = row.get(original_field)
        edited = row.get(edited_field)
        if (
            isinstance(original, str)
            and isinstance(edited, str)
            and normalize_text(original)
            and normalize_text(edited)
        ):
            atoms.add(
                _atom(
                    "edit_link",
                    {"original": original, "edited": edited},
                    f"{original_field}->{edited_field}",
                )
            )
    return tuple(sorted(atoms))


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    if path.suffix == ".gz":
        handle = opener(path, "rt", encoding="utf-8")
    else:
        handle = opener(path, "r", encoding="utf-8")
    with handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"{path} contains a non-object JSON row")
                yield value


class ManifestAdapter:
    """Adapter loaded from a JSON field-mapping manifest."""

    def __init__(self, manifest_path: Path, dataset_dir: Path | None = None):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._dataset = str(payload["dataset"])
        self._revision = str(payload["revision"])
        root = dataset_dir or manifest_path.parent
        sources = []
        for item in payload["sources"]:
            mapping = item.get("mapping", {})
            sources.append(
                SourceSpec(
                    dataset=self._dataset,
                    revision=self._revision,
                    view=str(item["view"]),
                    split=str(item["split"]),
                    path=root / str(item["path"]),
                    context_field=mapping.get("context"),
                    response_fields=tuple(mapping.get("responses", ())),
                    feedback_fields=tuple(mapping.get("feedback", ())),
                    edit_pairs=tuple(
                        (str(pair["original"]), str(pair["edited"]))
                        for pair in mapping.get("edits", ())
                    ),
                    domain_field=mapping.get("domain"),
                    language_field=mapping.get("language"),
                    label_field=mapping.get("label"),
                )
            )
        self._sources = tuple(sources)
        missing = [str(source.path) for source in self._sources if not source.path.is_file()]
        if missing:
            raise FileNotFoundError("missing manifest sources: " + ", ".join(missing))

    @property
    def dataset(self) -> str:
        return self._dataset

    @property
    def revision(self) -> str:
        return self._revision

    def sources(self) -> tuple[SourceSpec, ...]:
        return self._sources

    def records(self, source: SourceSpec) -> Iterator[RecordRef]:
        for source_index, row in enumerate(_iter_jsonl(source.path)):
            yield RecordRef(
                dataset=source.dataset,
                revision=source.revision,
                view=source.view,
                split=source.split,
                source_index=source_index,
                record_hash=raw_record_hash(row),
                atoms=extract_manifest_atoms(row, source),
                domain=(
                    str(row.get(source.domain_field, "unknown"))
                    if source.domain_field
                    else "unknown"
                ),
                language=(
                    str(row.get(source.language_field, "unknown"))
                    if source.language_field
                    else "unknown"
                ),
                label=(
                    str(row.get(source.label_field, "unknown"))
                    if source.label_field
                    else "unknown"
                ),
            )
