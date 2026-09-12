"""Which route handles this message (FR-02, FR-03).

The one place the two branches are chosen between, so the Streamlit app and the
CLI cannot drift apart on it.
"""

from __future__ import annotations

from agent_log import log_step
from intent import REFUND_REQUEST, detect_intent

RAG = "rag"
REFUND = "refund"


def route(
    message: str, pending: dict | None = None, history: list[dict] | None = None
) -> str:
    """Return `RAG` or `REFUND` for one incoming message.

    A refund already in flight wins without asking the classifier: when we have
    just asked "shall I refund ₹499?", the answer to that question is part of
    the refund, and re-classifying "yes" on its own would be guesswork.
    """
    if pending:
        log_step("Intent", f"{REFUND_REQUEST} (continuing {pending.get('stage')})")
        return REFUND

    return REFUND if detect_intent(message, history) == REFUND_REQUEST else RAG
