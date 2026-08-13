"""Canonicalization and typed atom extraction.

Exact ancestry is intentionally conservative and inspectable. Similarity
search must be implemented as a separate analysis and may not be silently
merged into these hashes.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable


_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, order=True)
class Atom:
    """A canonical datum that can connect records across schemas."""

    atom_type: str
    value_hash: str
    field: str
    linkable: bool = True


def normalize_text(value: str) -> str:
    """Normalize Unicode and whitespace without changing case or punctuation."""

    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value)).strip()


def normalize_value(value: Any) -> Any:
    """Recursively normalize strings while retaining container structure."""

    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): normalize_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        normalize_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def hash_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def raw_record_hash(row: dict[str, Any]) -> str:
    """Hash a normalized complete record."""

    return hash_value(row)


def literal_record_hash(row: dict[str, Any]) -> str:
    """Hash a parsed record without normalizing string contents."""

    payload = json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _string_items(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str):
                yield item


def _atom(
    atom_type: str, value: Any, field: str, *, linkable: bool = True
) -> Atom:
    return Atom(
        atom_type=atom_type,
        value_hash=hash_value(value),
        field=field,
        linkable=linkable,
    )


def extract_atoms(view: str, row: dict[str, Any]) -> tuple[Atom, ...]:
    """Extract exact typed atoms from supported feedback-data schemas.

    ``field`` is provenance metadata. Equality is defined by ``atom_type`` and
    ``value_hash``, allowing a response to connect across differently named
    fields while preventing a response from equaling a context by accident.
    """

    atoms: set[Atom] = {
        _atom("record", row, "__record__"),
    }

    context = row.get("context", row.get("prompt"))
    if context not in (None, "", []):
        atoms.add(_atom("context", context, "context" if "context" in row else "prompt"))

    response_fields = (
        "response",
        "response1",
        "response2",
        "original_response",
        "edited_response",
        "good_edited_response",
        "bad_edited_response",
    )
    for field in response_fields:
        value = row.get(field)
        if isinstance(value, str) and normalize_text(value):
            # Identical standalone responses can be generic ("Yes.", apology
            # templates, refusals) and are therefore evidence to report, but
            # are insufficient by themselves to merge source records.
            atoms.add(_atom("response", value, field, linkable=False))

    for field in ("feedback", "feedback1", "feedback2"):
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

    principle = row.get("principle")
    if isinstance(principle, str) and normalize_text(principle):
        # Principle vocabularies behave like reusable labels. They are
        # reportable exposure atoms but are not component-linking ancestry by
        # default.
        atoms.add(_atom("principle", principle, "principle", linkable=False))

    original = row.get("original_response")
    if isinstance(original, str) and normalize_text(original):
        for field in (
            "edited_response",
            "good_edited_response",
            "bad_edited_response",
        ):
            edited = row.get(field)
            if isinstance(edited, str) and normalize_text(edited):
                atoms.add(
                    _atom(
                        "edit_link",
                        {
                            "original": original,
                            "edited": edited,
                        },
                        f"original_response->{field}",
                    )
                )

    return tuple(sorted(atoms))
