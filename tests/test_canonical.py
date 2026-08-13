from ancestry_audit.canonical import (
    extract_atoms,
    hash_value,
    normalize_text,
    raw_record_hash,
)


def test_whitespace_and_unicode_are_normalized():
    assert normalize_text("A\u00a0  B\nC") == "A B C"
    assert hash_value("Ａ") == hash_value("A")


def test_key_order_does_not_change_record_hash():
    assert raw_record_hash({"b": 2, "a": "x"}) == raw_record_hash(
        {"a": "x", "b": 2}
    )


def test_response_connects_across_schema_fields():
    preference = extract_atoms(
        "preference",
        {"context": [{"role": "user", "content": "Q"}], "response1": "A"},
    )
    edit = extract_atoms(
        "edit",
        {
            "context": [{"role": "user", "content": "Q"}],
            "original_response": "A",
            "edited_response": "B",
        },
    )
    left = {(atom.atom_type, atom.value_hash) for atom in preference}
    right = {(atom.atom_type, atom.value_hash) for atom in edit}
    assert ("context", next(a.value_hash for a in preference if a.atom_type == "context")) in right
    assert any(key in right for key in left if key[0] == "response")


def test_principles_are_reportable_but_not_component_linking():
    atoms = extract_atoms("principle", {"principle": "accuracy"})
    principle = next(atom for atom in atoms if atom.atom_type == "principle")
    assert principle.linkable is False
