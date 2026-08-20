"""Prompt builders for the TISER experiment conditions."""

from __future__ import annotations

from typing import Any


TEMPORAL_CONTEXT_MARKER = "Temporal context:"
ANSWER_SECTION_MARKER = "### Answer:"


def extract_temporal_context(tiser_prompt: str) -> str:
    """Extract the context block from the dataset's TISER prompt."""
    prompt_text = str(tiser_prompt)

    try:
        context_start = prompt_text.index(TEMPORAL_CONTEXT_MARKER) + len(
            TEMPORAL_CONTEXT_MARKER
        )
    except ValueError as error:
        raise ValueError("TISER prompt is missing the temporal context marker.") from error

    try:
        context_end = prompt_text.index(ANSWER_SECTION_MARKER, context_start)
    except ValueError as error:
        raise ValueError("TISER prompt is missing the answer section marker.") from error

    return prompt_text[context_start:context_end].strip()


def build_standard_prompt(example: dict[str, Any]) -> str:
    """Build the plain context-and-question prompt used in the standard condition."""
    question = str(example["question"])
    temporal_context = extract_temporal_context(str(example["prompt"]))
    return (
        "You are an AI assistant that has to respond to questions given a context.\n\n"
        f"Question: {question}\n\n"
        f"Temporal Context: {temporal_context}"
    )


def build_tiser_prompt(example: dict[str, Any]) -> str:
    """Use the dataset's original TISER prompt without changing its formatting."""
    return str(example["prompt"])


def get_prompt_text(example: dict[str, Any], prompt_type: str) -> str:
    """Select the prompt format for an experiment condition."""
    if prompt_type == "standard":
        return build_standard_prompt(example)
    if prompt_type == "tiser":
        return build_tiser_prompt(example)
    raise ValueError(f"Unknown prompt type: {prompt_type}")
