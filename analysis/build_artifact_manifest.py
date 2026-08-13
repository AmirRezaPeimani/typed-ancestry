#!/usr/bin/env python3
"""Build a deterministic SHA-256 manifest for a revision or release tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXCLUDED_DIR_NAMES = {
    ".git",
    ".cache",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "build",
    "checkpoints",
    "data",
    "dist",
    "outputs",
    "runs",
    "tmp",
    "wandb",
    "work",
}
EXCLUDED_FILE_NAMES = {
    ".DS_Store",
    "ARTIFACT_MANIFEST.json",
    "artifact_manifest.json",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def is_excluded(path: Path, root: Path, output: Path) -> bool:
    if path.resolve() == output.resolve():
        return True
    relative = path.relative_to(root)
    if path.name in EXCLUDED_FILE_NAMES or path.suffix == ".pyc":
        return True
    return any(
        part in EXCLUDED_DIR_NAMES or part.endswith(".egg-info")
        for part in relative.parts[:-1]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    output = (
        args.output.resolve()
        if args.output
        else root / "artifact_manifest.json"
    )
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if is_excluded(path, root, output):
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )

    payload = {
        "schema_version": "1.0",
        "root": ".",
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
