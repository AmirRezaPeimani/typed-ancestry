from ancestry_audit.canonical import extract_atoms, raw_record_hash
from ancestry_audit.graph import AncestryGraph, RecordRef
from ancestry_audit.split import assert_component_disjoint, component_folds


def test_split_is_deterministic_and_keeps_components_whole():
    rows = []
    for index in range(20):
        row = {"context": [f"q{index // 2}"], "response1": f"a{index}"}
        rows.append(
            RecordRef(
                "fixture",
                "rev",
                "preference" if index % 2 == 0 else "feedback",
                "combined",
                index,
                raw_record_hash(row),
                extract_atoms("preference", row),
            )
        )
    graph = AncestryGraph(rows)
    first = component_folds(graph, folds=4, seed=7)
    second = component_folds(graph, folds=4, seed=7)
    assert first == second
    assert_component_disjoint(graph, first)
