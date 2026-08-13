import json
from pathlib import Path

from ancestry_audit.adapter_api import ManifestAdapter
from ancestry_audit.adapters import HELPSTEER3_VIEWS, helpsteer3_sources
from ancestry_audit.graph import AncestryGraph


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_manifest_adapter_connects_cross_view_context_and_edit(tmp_path):
    shared_context = [{"role": "user", "content": "Improve this answer."}]
    write_jsonl(
        tmp_path / "edit_train.jsonl",
        [
            {
                "prompt": shared_context,
                "before": "Draft",
                "after": "Revised",
                "critique": ["Add supporting detail."],
            }
        ],
    )
    write_jsonl(
        tmp_path / "quality_validation.jsonl",
        [
            {
                "context": shared_context,
                "original": "Draft",
                "candidate": "Revised",
                "label": "good",
            }
        ],
    )
    manifest = {
        "dataset": "example/multiview",
        "revision": "0123456789abcdef",
        "sources": [
            {
                "view": "edit",
                "split": "train",
                "path": "edit_train.jsonl",
                "mapping": {
                    "context": "prompt",
                    "responses": ["before", "after"],
                    "feedback": ["critique"],
                    "edits": [{"original": "before", "edited": "after"}],
                },
            },
            {
                "view": "edit_quality",
                "split": "validation",
                "path": "quality_validation.jsonl",
                "mapping": {
                    "context": "context",
                    "responses": ["original", "candidate"],
                    "edits": [{"original": "original", "edited": "candidate"}],
                    "label": "label",
                },
            },
        ],
    }
    manifest_path = tmp_path / "adapter.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    adapter = ManifestAdapter(manifest_path)
    records = [
        record
        for source in adapter.sources()
        for record in adapter.records(source)
    ]
    graph = AncestryGraph(records)

    assert len(records) == 2
    assert records[1].label == "good"
    assert graph.component_ids[0] == graph.component_ids[1]
    assert graph.component_summary()["cross_split_components"] == 1


def test_manifest_adapter_rejects_missing_source(tmp_path):
    manifest = {
        "dataset": "example/missing",
        "revision": "0123456789abcdef",
        "sources": [
            {
                "view": "feedback",
                "split": "train",
                "path": "absent.jsonl",
                "mapping": {"context": "context"},
            }
        ],
    }
    path = tmp_path / "adapter.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    try:
        ManifestAdapter(path)
    except FileNotFoundError as error:
        assert "absent.jsonl" in str(error)
    else:
        raise AssertionError("missing source was accepted")


def test_helpsteer3_sources_support_hub_directory_layout(tmp_path):
    for view in HELPSTEER3_VIEWS:
        directory = tmp_path / view
        directory.mkdir()
        for split in ("train", "validation"):
            (directory / f"{split}.jsonl.gz").write_bytes(b"")

    sources = helpsteer3_sources(tmp_path, "0123456789abcdef")

    assert len(sources) == 10
    assert all(source.path.parent.name == source.view for source in sources)
