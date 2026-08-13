"""Deterministic contracts for the controlled ancestry-exposure experiment."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from .canonical import hash_value, normalize_text, raw_record_hash


LENGTH_BOUNDARIES = (1000, 3000, 8000)


def stable_rank(seed: int, namespace: str, identifier: str) -> str:
    payload = f"{seed}\0{namespace}\0{identifier}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def context_hash(row: dict[str, Any]) -> str:
    return hash_value(row["context"])


def directed_edit_hash(original: str, edited: str) -> str:
    return hash_value(
        {"original": normalize_text(original), "edited": normalize_text(edited)}
    )


def edit_link_hash(row: dict[str, Any], edited_field: str) -> str:
    return directed_edit_hash(row["original_response"], row[edited_field])


def length_features(row: dict[str, Any]) -> tuple[float, float]:
    return (
        math.log1p(len(normalize_text(row["good_edited_response"]))),
        math.log1p(len(normalize_text(row["bad_edited_response"]))),
    )


def length_bin(row: dict[str, Any]) -> str:
    maximum = max(
        len(normalize_text(row["good_edited_response"])),
        len(normalize_text(row["bad_edited_response"])),
    )
    if maximum <= LENGTH_BOUNDARIES[0]:
        return "0000_1000"
    if maximum <= LENGTH_BOUNDARIES[1]:
        return "1001_3000"
    if maximum <= LENGTH_BOUNDARIES[2]:
        return "3001_8000"
    return "8001_plus"


def match_stratum(row: dict[str, Any]) -> str:
    return f"{row.get('domain', 'unknown')}|{length_bin(row)}"


def orient_pair(
    *,
    context: Any,
    preferred: str,
    rejected: str,
    source_id: str,
    source_view: str,
    domain: str,
    language: str,
) -> dict[str, Any]:
    """Create a binary pair with a deterministic, non-positional label."""

    reverse = int(stable_rank(0, "pair_orientation", source_id)[:16], 16) % 2
    if reverse:
        response1, response2, preference = preferred, rejected, -1
    else:
        response1, response2, preference = rejected, preferred, 1
    return {
        "pair_id": source_id,
        "source_view": source_view,
        "domain": str(domain),
        "language": str(language),
        "context": context,
        "response1": response1,
        "response2": response2,
        "overall_preference": preference,
    }


def edit_training_pair(row: dict[str, Any]) -> dict[str, Any]:
    source_id = raw_record_hash(row)
    return orient_pair(
        context=row["context"],
        preferred=row["edited_response"],
        rejected=row["original_response"],
        source_id=source_id,
        source_view="edit",
        domain=row.get("domain", "unknown"),
        language=row.get("language", "unknown"),
    )


def edit_quality_pair(row: dict[str, Any]) -> dict[str, Any]:
    source_id = raw_record_hash(row)
    return orient_pair(
        context=row["context"],
        preferred=row["good_edited_response"],
        rejected=row["bad_edited_response"],
        source_id=source_id,
        source_view="edit_quality",
        domain=row.get("domain", "unknown"),
        language=row.get("language", "unknown"),
    )


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def load_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def deduplicate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(raw_record_hash(row), row)
    return [unique[key] for key in sorted(unique)]


def one_row_per_context(
    rows: Sequence[dict[str, Any]], *, seed: int, namespace: str
) -> list[dict[str, Any]]:
    chosen: dict[str, tuple[str, dict[str, Any]]] = {}
    for row in rows:
        key = context_hash(row)
        rank = stable_rank(seed, namespace, raw_record_hash(row))
        previous = chosen.get(key)
        if previous is None or rank < previous[0]:
            chosen[key] = (rank, row)
    return [chosen[key][1] for key in sorted(chosen)]


def allocate_quotas(
    capacities: dict[str, int], total: int
) -> dict[str, int]:
    """Allocate exactly ``total`` slots without exceeding stratum capacities."""

    usable = {key: value for key, value in capacities.items() if value > 0}
    if sum(usable.values()) < total:
        raise ValueError(
            f"Only {sum(usable.values())} matched slots are available; need {total}"
        )
    weight_total = sum(usable.values())
    raw = {key: total * value / weight_total for key, value in usable.items()}
    quota = {key: min(usable[key], int(math.floor(raw[key]))) for key in usable}
    remaining = total - sum(quota.values())
    order = sorted(
        usable,
        key=lambda key: (
            -(raw[key] - math.floor(raw[key])),
            -usable[key],
            key,
        ),
    )
    while remaining:
        progressed = False
        for key in order:
            if quota[key] < usable[key]:
                quota[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise AssertionError("Quota allocator became stuck")
    return quota


def total_variation(left: Sequence[str], right: Sequence[str]) -> float:
    left_counts, right_counts = Counter(left), Counter(right)
    keys = set(left_counts) | set(right_counts)
    left_n, right_n = len(left), len(right)
    if not left_n or not right_n:
        return 1.0
    return 0.5 * sum(
        abs(left_counts[key] / left_n - right_counts[key] / right_n)
        for key in keys
    )


def standardized_mean_difference(
    left: Sequence[float], right: Sequence[float]
) -> float:
    if len(left) < 2 or len(right) < 2:
        raise ValueError("SMD needs at least two observations per group")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_var = sum((value - left_mean) ** 2 for value in left) / (len(left) - 1)
    right_var = sum((value - right_mean) ** 2 for value in right) / (
        len(right) - 1
    )
    pooled = math.sqrt((left_var + right_var) / 2)
    if pooled == 0:
        return 0.0
    return (left_mean - right_mean) / pooled


def assignment_hash(rows: Sequence[dict[str, Any]]) -> str:
    payload = "\n".join(str(row["pair_id"]) for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
