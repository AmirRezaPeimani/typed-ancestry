#!/usr/bin/env python3
"""Verify reconstructed data files against a checked manifest."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import BinaryIO


def digest(handle: BinaryIO) -> str:
    value = hashlib.sha256()
    for block in iter(lambda: handle.read(1 << 20), b""):
        value.update(block)
    return value.hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return digest(handle)


def content_digest(path: Path) -> str:
    with gzip.open(path, "rb") as handle:
        return digest(handle)


def row_count(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    checks = []
    for name, expected in manifest["files"].items():
        path = args.data_dir / name
        observed = {
            "name": name,
            "exists": path.is_file(),
            "rows": row_count(path) if path.is_file() else None,
            "sha256": file_digest(path) if path.is_file() else None,
            "content_sha256": (
                content_digest(path) if path.is_file() else None
            ),
        }
        observed["valid"] = (
            observed["exists"]
            and observed["rows"] == expected["rows"]
            and (
                "sha256" not in expected
                or observed["sha256"] == expected["sha256"]
            )
            and (
                "content_sha256" not in expected
                or observed["content_sha256"]
                == expected["content_sha256"]
            )
        )
        checks.append(observed)

    payload = {"valid": all(row["valid"] for row in checks), "files": checks}
    print(json.dumps(payload, indent=2))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
