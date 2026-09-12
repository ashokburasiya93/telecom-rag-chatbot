"""Session conversation history (FR-45 – FR-50).

History lives in the UI's session state; this module is only about shaping it
for the two things that consume it — the RAG prompt and the refund agent. It
deliberately does not touch the retrieval query: retrieval still runs on the
question as typed (FR-49).
"""

from __future__ import annotations

from config import MAX_HISTORY_TURNS


def trim(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """Keep the most recent `max_turns` exchanges (FR-50).

    A turn is a user message plus its answer, so the cap is on messages, not
    exchanges — hence the doubling.
    """
    if max_turns <= 0:
        return []
    return history[-(max_turns * 2) :]


def as_messages(history: list[dict]) -> list[tuple[str, str]]:
    """History as LangChain ("human"|"ai", text) pairs for a prompt template."""
    role_map = {"user": "human", "assistant": "ai"}
    return [
        (role_map[m["role"]], m["content"])
        for m in trim(history)
        if m.get("role") in role_map and m.get("content")
    ]


def as_transcript(history: list[dict]) -> str:
    """History as plain text, for prompts that take context as a string."""
    lines = [
        f"{'Customer' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in trim(history)
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    return "\n".join(lines) if lines else "(no earlier messages)"
