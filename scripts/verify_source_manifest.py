#!/usr/bin/env python3
"""Verify reconstructed HelpSteer3 files against the public study manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "manifests"
        / "helpsteer3_sources.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checks = []
    for expected in manifest["files"]:
        repo_path = expected.get("repo_path", expected["name"])
        nested = args.dataset_dir / repo_path
        flattened = args.dataset_dir / expected["name"]
        path = nested if nested.is_file() else flattened
        observed = {
            "name": expected["name"],
            "repo_path": repo_path,
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256(path) if path.is_file() else None,
        }
        observed["valid"] = (
            observed["exists"]
            and observed["bytes"] == expected["bytes"]
            and observed["sha256"] == expected["sha256"]
        )
        checks.append(observed)
    payload = {"valid": all(row["valid"] for row in checks), "files": checks}
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
