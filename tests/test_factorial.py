from ancestry_audit.factorial import (
    allocate_quotas,
    directed_edit_hash,
    edit_quality_pair,
    edit_training_pair,
    standardized_mean_difference,
)


def test_directed_edit_hash_preserves_direction():
    assert directed_edit_hash("a", "b") != directed_edit_hash("b", "a")
    assert directed_edit_hash("Ａ", "b  c") == directed_edit_hash("A", "b c")


def test_edit_and_quality_pairs_encode_the_preferred_candidate():
    context = [{"role": "user", "content": "revise"}]
    edit = edit_training_pair(
        {
            "domain": "general",
            "language": "english",
            "context": context,
            "original_response": "original",
            "edited_response": "better",
        }
    )
    quality = edit_quality_pair(
        {
            "domain": "general",
            "language": "english",
            "context": context,
            "original_response": "original",
            "good_edited_response": "better",
            "bad_edited_response": "worse",
        }
    )
    edit_preferred = (
        edit["response2"]
        if edit["overall_preference"] > 0
        else edit["response1"]
    )
    quality_preferred = (
        quality["response2"]
        if quality["overall_preference"] > 0
        else quality["response1"]
    )
    assert edit_preferred == "better"
    assert quality_preferred == "better"


def test_quota_allocator_is_exact_and_bounded():
    result = allocate_quotas({"a": 8, "b": 2, "c": 0}, 7)
    assert sum(result.values()) == 7
    assert result["a"] <= 8
    assert result["b"] <= 2


def test_smd_is_zero_for_equal_samples():
    assert standardized_mean_difference([1.0, 2.0], [1.0, 2.0]) == 0.0
