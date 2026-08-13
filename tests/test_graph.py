from ancestry_audit.canonical import extract_atoms, raw_record_hash
from ancestry_audit.graph import AncestryGraph, RecordRef


def record(view, split, index, row):
    return RecordRef(
        dataset="fixture",
        revision="abc",
        view=view,
        split=split,
        source_index=index,
        record_hash=raw_record_hash(row),
        atoms=extract_atoms(view, row),
    )


def test_cross_schema_context_and_response_join_one_component():
    graph = AncestryGraph(
        [
            record(
                "preference",
                "train",
                0,
                {"context": ["q"], "response1": "a", "response2": "b"},
            ),
            record(
                "edit",
                "validation",
                0,
                {
                    "context": ["q"],
                    "original_response": "a",
                    "edited_response": "c",
                },
            ),
        ]
    )
    assert graph.component_ids[0] == graph.component_ids[1]


def test_generic_principle_does_not_join_components():
    graph = AncestryGraph(
        [
            record(
                "principle",
                "train",
                0,
                {"context": ["q1"], "response": "a", "principle": "accuracy"},
            ),
            record(
                "principle",
                "validation",
                0,
                {"context": ["q2"], "response": "b", "principle": "accuracy"},
            ),
        ]
    )
    assert graph.component_ids[0] != graph.component_ids[1]


def test_component_summary_counts_cross_split_multiview_component():
    graph = AncestryGraph(
        [
            record("preference", "train", 0, {"context": ["q"], "response1": "a"}),
            record("feedback", "validation", 0, {"context": ["q"], "response1": "a"}),
            record("preference", "train", 1, {"context": ["other"], "response1": "z"}),
        ]
    )
    summary = graph.component_summary()
    assert summary["cross_split_components"] == 1
    assert summary["multi_view_components"] == 1
    assert summary["records_in_cross_split_components"] == 2
