from ancestry_audit.canonical import extract_atoms, raw_record_hash
from ancestry_audit.exposure import any_view_summary, exposure_rows
from ancestry_audit.graph import AncestryGraph, RecordRef


def rec(view, split, index, row):
    return RecordRef(
        "fixture",
        "rev",
        view,
        split,
        index,
        raw_record_hash(row),
        extract_atoms(view, row),
    )


def test_cross_view_only_exposure_is_separate_from_same_view():
    graph = AncestryGraph(
        [
            rec(
                "edit",
                "train",
                0,
                {
                    "context": ["q"],
                    "original_response": "a",
                    "edited_response": "c",
                },
            ),
            rec(
                "edit_quality",
                "validation",
                0,
                {"context": ["q"], "original_response": "a"},
            ),
            rec(
                "edit_quality",
                "validation",
                1,
                {"context": ["clean"], "original_response": "b"},
            ),
        ]
    )
    summary = any_view_summary(graph, exposure_rows(graph))
    row = next(
        item
        for item in summary
        if item["evaluation_view"] == "edit_quality"
        and item["atom_type"] == "context"
    )
    assert row["same_view_exposed_rows"] == 0
    assert row["any_view_exposed_rows"] == 1
    assert row["cross_view_only_rows"] == 1
    assert row["any_view_exposure_rate"] == 0.5


def test_zero_exposure_types_remain_in_summary():
    graph = AncestryGraph(
        [
            rec(
                "edit",
                "train",
                0,
                {
                    "context": ["q"],
                    "original_response": "a",
                    "edited_response": "c",
                },
            ),
            rec("preference", "validation", 0, {"context": ["clean"], "response1": "b"}),
        ]
    )
    summary = any_view_summary(graph, exposure_rows(graph))
    row = next(
        item
        for item in summary
        if item["evaluation_view"] == "preference"
        and item["atom_type"] == "edit_link"
    )
    assert row["any_view_exposed_rows"] == 0
    assert row["any_view_exposure_rate"] == 0
