"""Prompt templates for temporal reasoning tasks."""


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful temporal reasoning assistant. "
    "Answer using the provided context and make date/time assumptions explicit."
)


def build_prompt(question: str, context: str | None = None) -> str:
    """Build a prompt for a temporal reasoning question."""
    if context:
        return f"{DEFAULT_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion:\n{question}"
    return f"{DEFAULT_SYSTEM_PROMPT}\n\nQuestion:\n{question}"
