"""Construct-complete serialization for the frozen neural experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


FIELD_MARKERS = {
    "context": "[CONTEXT]",
    "original_response": "[ORIGINAL_RESPONSE]",
    "feedback": "[FEEDBACK]",
    "candidate": "[CANDIDATE_EDIT]",
}
MINIMUM_BUDGETS = {
    "context": 64,
    "original_response": 80,
    "feedback": 144,
    "candidate": 160,
}
REALLOCATION_ORDER = ("candidate", "feedback", "original_response", "context")
OMISSION = "[...OMITTED...]"


@dataclass(frozen=True)
class SerializedCandidate:
    input_ids: list[int]
    attention_mask: list[int]
    metadata: dict[str, Any]


def _context_text(messages: list[dict[str, Any]]) -> str:
    rendered = []
    for message in messages:
        role = str(message.get("role", "unknown")).strip().upper()
        content = str(message.get("content", "")).strip()
        rendered.append(f"[ROLE {role}] {content}")
    return "\n".join(rendered)


class ConstructSerializer:
    def __init__(self, tokenizer: Any, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.marker_ids = {
            field: self._encode(marker) for field, marker in FIELD_MARKERS.items()
        }
        self.omission_ids = self._encode(OMISSION)
        # Encoder tokenizers commonly wrap sequences with CLS/SEP, whereas
        # decoder tokenizers may expose BOS/EOS or only EOS. Keep the frozen
        # field serialization unchanged and use the tokenizer's declared
        # boundary tokens when available.
        start_id = tokenizer.cls_token_id
        if start_id is None:
            start_id = tokenizer.bos_token_id
        end_id = tokenizer.sep_token_id
        if end_id is None:
            end_id = tokenizer.eos_token_id
        self.prefix_ids = [] if start_id is None else [int(start_id)]
        self.suffix_ids = [] if end_id is None else [int(end_id)]
        if not self.prefix_ids and not self.suffix_ids:
            raise ValueError(
                "tokenizer must define at least one CLS/BOS/SEP/EOS boundary token"
            )
        marker_cost = sum(len(ids) for ids in self.marker_ids.values())
        self.available_content = (
            max_length
            - marker_cost
            - len(self.prefix_ids)
            - len(self.suffix_ids)
        )
        if self.available_content < sum(MINIMUM_BUDGETS.values()):
            raise ValueError("maximum length cannot satisfy frozen field budgets")

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _truncate(self, ids: list[int], budget: int) -> list[int]:
        if len(ids) <= budget:
            return ids
        if budget <= 0:
            return []
        if budget <= len(self.omission_ids) + 1:
            return ids[:budget]
        available = budget - len(self.omission_ids)
        head = (available + 1) // 2
        tail = available - head
        return ids[:head] + self.omission_ids + (ids[-tail:] if tail else [])

    def _feedback_tokens(
        self, entries: list[str], budget: int
    ) -> tuple[list[int], list[int]]:
        headers = [
            self._encode(f"[ENTRY {index + 1}/{len(entries)}]")
            for index in range(len(entries))
        ]
        contents = [self._encode(str(entry).strip()) for entry in entries]
        if any(not content for content in contents):
            raise ValueError("feedback entries must contain text")
        minimum = sum(len(header) for header in headers) + len(contents)
        if budget < minimum:
            raise ValueError("feedback budget cannot retain every source entry")
        quotas = [1] * len(contents)
        remaining = budget - minimum
        while remaining:
            active = [
                index
                for index, content in enumerate(contents)
                if quotas[index] < len(content)
            ]
            if not active:
                break
            share = max(1, remaining // len(active))
            progressed = 0
            for index in active:
                addition = min(
                    share,
                    len(contents[index]) - quotas[index],
                    remaining,
                )
                quotas[index] += addition
                remaining -= addition
                progressed += addition
                if not remaining:
                    break
            if not progressed:
                raise AssertionError("feedback allocator became stuck")
        output: list[int] = []
        retained_content: list[int] = []
        for header, content, quota in zip(headers, contents, quotas, strict=True):
            truncated = self._truncate(content, quota)
            if not truncated:
                raise AssertionError("feedback entry lost all content")
            output.extend(header)
            output.extend(truncated)
            retained_content.append(len(truncated))
        return output, retained_content

    def serialize(
        self, row: dict[str, Any], candidate_field: str
    ) -> SerializedCandidate:
        context = row.get("context")
        original = row.get("original_response")
        feedback = row.get("feedback")
        candidate = row.get(candidate_field)
        if not isinstance(context, list) or not context:
            raise ValueError("missing structured context")
        if not isinstance(original, str) or not original.strip():
            raise ValueError("missing original response")
        if not isinstance(feedback, list) or not feedback:
            raise ValueError("missing feedback entries")
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError(f"missing candidate field {candidate_field}")

        raw = {
            "context": self._encode(_context_text(context)),
            "original_response": self._encode(original.strip()),
            "feedback": [],
            "candidate": self._encode(candidate.strip()),
        }
        full_feedback, _ = self._feedback_tokens(
            [str(entry) for entry in feedback],
            budget=10**9,
        )
        raw["feedback"] = full_feedback
        budgets = {
            field: min(len(raw[field]), minimum)
            for field, minimum in MINIMUM_BUDGETS.items()
        }
        spare = self.available_content - sum(budgets.values())
        for field in REALLOCATION_ORDER:
            addition = min(spare, len(raw[field]) - budgets[field])
            budgets[field] += addition
            spare -= addition

        retained = {
            "context": self._truncate(raw["context"], budgets["context"]),
            "original_response": self._truncate(
                raw["original_response"], budgets["original_response"]
            ),
            "candidate": self._truncate(raw["candidate"], budgets["candidate"]),
        }
        retained["feedback"], feedback_entry_lengths = self._feedback_tokens(
            [str(entry) for entry in feedback], budgets["feedback"]
        )
        if any(not retained[field] for field in retained):
            raise AssertionError("a construct field lost all retained content")

        input_ids = list(self.prefix_ids)
        for field in ("context", "original_response", "feedback", "candidate"):
            input_ids.extend(self.marker_ids[field])
            input_ids.extend(retained[field])
        input_ids.extend(self.suffix_ids)
        if len(input_ids) > self.max_length:
            raise AssertionError(
                f"serialization length {len(input_ids)} exceeds {self.max_length}"
            )
        metadata = {
            "length": len(input_ids),
            "raw_tokens": {field: len(tokens) for field, tokens in raw.items()},
            "retained_tokens": {
                field: len(tokens) for field, tokens in retained.items()
            },
            "budgets": budgets,
            "truncated": {
                field: len(retained[field]) < len(raw[field]) for field in raw
            },
            "feedback_entries": len(feedback),
            "feedback_entry_retained_tokens": feedback_entry_lengths,
            "prefix_special_tokens": len(self.prefix_ids),
            "suffix_special_tokens": len(self.suffix_ids),
            "all_markers_retained": True,
            "all_fields_nonempty": True,
            "all_feedback_entries_nonempty": all(
                length > 0 for length in feedback_entry_lengths
            ),
        }
        return SerializedCandidate(
            input_ids=input_ids,
            attention_mask=[1] * len(input_ids),
            metadata=metadata,
        )
