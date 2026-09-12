"""Intent detection in front of the existing RAG pipeline (FR-01 – FR-05a).

Two intents, one classifier call, on the LLM the app is already configured with
(FR-04). The classifier returns a validated constant rather than whatever the
model happened to type (FR-05), and anything it cannot parse falls back to
`requirement_inquiry` — the route that cannot spend money.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent_log import log_step
from conversation import as_messages

REQUIREMENT_INQUIRY = "requirement_inquiry"
REFUND_REQUEST = "refund_request"
INTENTS = (REQUIREMENT_INQUIRY, REFUND_REQUEST)

CLASSIFIER_PROMPT = """You label one message from a telecom customer with exactly one \
intent.

refund_request — the customer wants money back for a recharge they have already \
paid for: a refund, a reversal, or cancelling a recharge that has gone through. \
Also use this when they are answering a question we asked them as part of a refund \
they already started, such as naming an amount.

requirement_inquiry — anything else: questions about plans, pricing, billing, \
roaming, data, SIM or eSIM, call quality, the app, troubleshooting, and questions \
about how something works or how to do something.

Decide by what the customer is asking us to DO, not by which words appear:

"I want a refund of 999" -> refund_request
"Cancel my last recharge and give my money back" -> refund_request
"Reverse the recharge I made yesterday" -> refund_request
"How do I cancel a recharge?" -> requirement_inquiry (asking how, not asking us to do it)
"What is your refund policy?" -> requirement_inquiry (asking about the rules)
"Why is my internet slow?" -> requirement_inquiry
"Why is my bill higher than usual?" -> requirement_inquiry

If you are not sure, answer requirement_inquiry.

Answer with one word and nothing else: requirement_inquiry or refund_request."""

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CLASSIFIER_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{message}"),
    ]
)


def _parse(raw: str) -> str | None:
    """Pull a known intent out of the model's reply.

    Substring matching rather than equality: a model that answers
    "refund_request." or wraps the word in backticks has still classified
    correctly, and a model that answers with a sentence containing neither
    intent has not — that is the case worth falling back on.
    """
    text = (raw or "").strip().lower()
    found = [intent for intent in INTENTS if intent in text]
    return found[0] if len(found) == 1 else None


def detect_intent(message: str, history: list[dict] | None = None) -> str:
    """Classify one message as `requirement_inquiry` or `refund_request`."""
    from chain import MissingAPIKey, get_llm

    try:
        response = (_PROMPT | get_llm()).invoke(
            {"message": message, "history": as_messages(history or [])}
        )
        intent = _parse(getattr(response, "content", ""))
    except MissingAPIKey:
        raise
    except Exception as exc:
        log_step("Intent", f"classifier failed ({exc}) — defaulting to inquiry")
        intent = None

    if intent is None:
        intent = REQUIREMENT_INQUIRY

    log_step("Intent", intent)
    return intent
