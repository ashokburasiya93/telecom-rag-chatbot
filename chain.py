"""Prompt, LLM and the LCEL chain that turns a question into a grounded answer.

The whole point of the system prompt is FR-10: the model answers from the
retrieved context or it says it cannot answer. It never falls back on what it
happens to know about telecoms.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from config import (
    ESCALATION_HINT,
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    REASONING_EFFORT,
)
from retriever import RetrievedDoc, format_context, retrieve

SYSTEM_PROMPT = """You are the NovaCell customer care assistant. You help customers with our plans, \
pricing and add-ons, and you resolve common support issues: mobile data and \
connectivity, billing questions, SIM and eSIM problems, roaming, voice calls, and \
account or app basics.

CONTEXT (retrieved from the published plan and pricing catalog, the company FAQ, \
resolved support tickets, and the technical guide — these are the only facts you may \
use):
---
{context}
---

RULES — follow all of them:
1. Answer using ONLY the CONTEXT above. Never use your own knowledge of telecom \
networks, products, prices or policies, even if you are confident it is correct.
2. Never invent or estimate a price, rate, data allowance, speed, threshold, timeframe \
or policy. If a number is not in the CONTEXT, do not state a number. You may do simple \
arithmetic on numbers that ARE in the CONTEXT — for example multiplying a daily rate by \
the number of days a customer gave you — as long as you say what you worked it out from.
2a. When you recommend or compare plans, quote the exact plan names and prices from the \
CONTEXT, and state any eligibility requirement (student, 55+, new accounts only) that \
applies to a plan you mention. Never present a restricted plan as if anyone can buy it.
2b. [PLANS] is the current, dated price list and it wins. Where a [FAQ], [TICKETS] or \
[GUIDES] entry names a plan, pass or price that [PLANS] does not, it is out of date — \
leave it out of your answer and quote [PLANS] instead.
3. If the CONTEXT does not contain the answer, say so plainly in one sentence and tell \
the customer to {escalation_hint}. Do not guess, and do not pad the answer with \
generic advice.
3a. You only help with NovaCell service: plans, pricing, add-ons, billing and technical \
support. Anything else — travel itineraries, general recommendations, other companies' \
products, advice unrelated to the phone service — is out of scope. Say in one sentence \
that it is not something you can help with, then offer the one NovaCell thing that is \
relevant if there is one.
4. You have no access to the customer's account. If they ask about their own balance, \
bill, plan or number, say you cannot see account details, explain the general rule from \
the CONTEXT if there is one, and point them to the MyTelecom app or 611.
5. Write in plain customer-facing language. Keep it under 150 words. Use a numbered \
list for anything that is a procedure.
6. Never mention "context", "documents", "sources" or "retrieval" — just answer.
7. Do not repeat internal notes verbatim. If the CONTEXT describes something an agent \
or engineer did internally, translate it into what the customer should do or expect.
"""

HUMAN_PROMPT = "{question}"

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
).partial(escalation_hint=ESCALATION_HINT)


class MissingAPIKey(RuntimeError):
    """Raised when GROQ_API_KEY is not configured."""


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """Build the Groq chat model (FR-12, FR-13)."""
    if not GROQ_API_KEY:
        raise MissingAPIKey(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://console.groq.com/keys"
        )

    kwargs = dict(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=GROQ_API_KEY)
    try:
        return ChatGroq(reasoning_effort=REASONING_EFFORT, **kwargs)
    except Exception:
        # Older langchain-groq releases don't expose reasoning_effort directly.
        return ChatGroq(model_kwargs={"reasoning_effort": REASONING_EFFORT}, **kwargs)


@lru_cache(maxsize=1)
def get_chain():
    """prompt | llm | parser — context is supplied per call."""
    return PROMPT | get_llm() | StrOutputParser()


def answer_stream(question: str) -> tuple[list[RetrievedDoc], Iterator[str]]:
    """Retrieve context, then stream the grounded answer token by token (FR-05).

    Returns the retrieved documents alongside the stream so the UI can render
    the Sources panel (FR-13a) next to the answer it produced.
    """
    docs = retrieve(question)
    payload = {"context": format_context(docs), "question": question}
    return docs, get_chain().stream(payload)


def answer(question: str) -> tuple[list[RetrievedDoc], str]:
    """Non-streaming variant, used by tests and scripted checks."""
    docs = retrieve(question)
    payload = {"context": format_context(docs), "question": question}
    return docs, get_chain().invoke(payload)
