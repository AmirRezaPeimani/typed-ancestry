"""Shared pairwise reward-model serialization and labels."""

from __future__ import annotations

from typing import Any


OMISSION_MARKER = "\n[...MIDDLE_OMITTED...]\n"


def preferred_response(row: dict[str, Any]) -> tuple[str, str]:
    preference = float(row["overall_preference"])
    if preference == 0:
        raise ValueError("Pairwise rows cannot have a tied preference")
    if preference > 0:
        return str(row["response2"]), str(row["response1"])
    return str(row["response1"]), str(row["response2"])


def label_response2_preferred(row: dict[str, Any]) -> int:
    preference = float(row["overall_preference"])
    if preference == 0:
        raise ValueError("Pairwise rows cannot have a tied preference")
    return int(preference > 0)


def plain_role_sections(context: Any, response: str) -> tuple[str, str]:
    messages: list[tuple[str, str]] = []
    if isinstance(context, list):
        for item in context:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "user")).strip().upper() or "USER"
            messages.append((role, str(item.get("content", ""))))
    else:
        messages.append(("USER", str(context)))
    context_text = "\n".join(
        f"[{role}]\n{content}" for role, content in messages
    )
    return context_text, f"\n[CANDIDATE_RESPONSE]\n{response}"


def _head_tail(
    token_ids: list[int], budget: int, marker_ids: list[int]
) -> list[int]:
    if budget <= 0:
        return []
    if len(token_ids) <= budget:
        return token_ids
    if budget <= len(marker_ids) + 2:
        return token_ids[:budget]
    content_budget = budget - len(marker_ids)
    head = (content_budget + 1) // 2
    tail = content_budget - head
    return token_ids[:head] + marker_ids + token_ids[-tail:]


def raw_token_lengths(
    tokenizer: Any, context: Any, response: str
) -> tuple[int, int, int]:
    context_text, response_text = plain_role_sections(context, response)
    context_ids = tokenizer(
        context_text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )["input_ids"]
    response_ids = tokenizer(
        response_text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )["input_ids"]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    eos = int(eos_id is not None)
    return len(context_ids), len(response_ids), len(context_ids) + len(response_ids) + eos


def serialize_candidate(
    tokenizer: Any,
    context: Any,
    response: str,
    *,
    max_length: int,
    context_fraction: float = 0.5,
) -> dict[str, list[int]]:
    """Use fixed markers, recent context, and head-tail response retention."""

    if max_length < 16:
        raise ValueError("max_length must be at least 16")
    if not 0 < context_fraction < 1:
        raise ValueError("context_fraction must lie between zero and one")
    context_text, response_text = plain_role_sections(context, response)
    context_ids = list(
        tokenizer(
            context_text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )["input_ids"]
    )
    response_ids = list(
        tokenizer(
            response_text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )["input_ids"]
    )
    marker_ids = list(
        tokenizer(
            OMISSION_MARKER,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )["input_ids"]
    )
    eos_id = getattr(tokenizer, "eos_token_id", None)
    eos_ids = [] if eos_id is None else [int(eos_id)]
    available = max_length - len(eos_ids)
    context_cap = int(max_length * context_fraction)

    if len(context_ids) + len(response_ids) <= available:
        kept_context = context_ids
        kept_response = response_ids
    else:
        context_budget = min(len(context_ids), context_cap, available)
        response_budget = available - context_budget
        if len(response_ids) < response_budget:
            context_budget = min(
                len(context_ids), available - len(response_ids)
            )
            response_budget = available - context_budget
        kept_context = (
            context_ids[-context_budget:] if context_budget else []
        )
        kept_response = _head_tail(
            response_ids, response_budget, marker_ids
        )

    input_ids = kept_context + kept_response + eos_ids
    if len(input_ids) > max_length:
        raise AssertionError("Serialization exceeded max_length")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
    }
