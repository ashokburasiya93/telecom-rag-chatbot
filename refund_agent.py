"""The Refund Agent (§6.2, §7, §10) — a LangGraph workflow with hard edges.

The division of labour is the whole point of this module:

* The LLM is the *agent*. It reads the conversation and calls tools to find out
  what it is dealing with — `get_user_account` and `validate_refund`, the two
  read-only tools, are the only ones bound to it (FR-23a).
* The graph is the *control*. Eligibility, the ₹499 approval limit, whether a
  confirmation counts, and the two state-changing calls (`process_refund`,
  `submit_to_support`) are decided and made by code the model cannot reach
  (FR-29, FR-30, FR-36a, GR-01, GR-02, GR-05, GR-08).

So a model that hallucinates "your refund is approved" changes nothing: the
customer-facing text for every outcome is written here, from the tool results.

State is passed in and out on each turn rather than checkpointed, because the
Streamlit session already is the session store — see `RefundTurn.pending`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

import refund_tools
from accounts import APPROVAL_LIMIT, DEMO_USER_ID, REFUND_WINDOW_DAYS
from agent_log import log_step
from conversation import as_transcript
from refund_tools import format_rupees

# --- §10 workflow states -------------------------------------------------

REFUND_REQUESTED = "refund_requested"
REFUND_VALIDATED = "refund_validated"
AWAITING_CONFIRMATION = "awaiting_confirmation"
AWAITING_AMOUNT = "awaiting_amount"
REFUND_CONFIRMED = "refund_confirmed"
REFUND_PROCESSED = "refund_processed"
REFUND_REJECTED = "refund_rejected"
REFUND_CANCELLED = "refund_cancelled"
REFUND_SUBMITTED_FOR_REVIEW = "refund_submitted_for_review"

# Only these two mean "come back to me next turn"; everything else ends the
# refund and clears the pending state.
PENDING_STAGES = (AWAITING_CONFIRMATION, AWAITING_AMOUNT)

MAX_TOOL_ROUNDS = 4


# --- Amount extraction (FR-07, FR-44) ------------------------------------

_CURRENCY = r"₹|rs\.?|inr|rupees?"
_NON_MONEY_UNITS = r"days?|weeks?|months?|years?|hours?|gb|mb|tb|kb|%|percent|am|pm"

_AMOUNT_RE = re.compile(
    rf"(?P<cur>{_CURRENCY})?\s*"
    rf"(?<![\w.])(?P<num>\d{{1,3}}(?:,\d{{2,3}})+(?:\.\d{{1,2}})?|\d+(?:\.\d{{1,2}})?)(?![\d\w])"
    rf"\s*(?P<unit>{_CURRENCY}|{_NON_MONEY_UNITS})?",
    re.IGNORECASE,
)


def extract_amount(message: str) -> float | None:
    """Read a rupee amount out of what the customer typed, or return None.

    Deterministic on purpose. The agent may never invent an amount (FR-44), so
    the amount is taken from the customer's own words or asked for — it is
    never something the model decided looked plausible.

    Numbers carrying a non-money unit are ignored, so "my 2 day old recharge"
    and "5 GB left" are not read as amounts, and identifiers like `rch_457`
    are excluded by the word-boundary guards.
    """
    if not message:
        return None

    candidates: list[tuple[bool, float]] = []
    for match in _AMOUNT_RE.finditer(message):
        unit = (match.group("unit") or "").lower()
        if unit and not re.fullmatch(_CURRENCY, unit, re.IGNORECASE):
            continue  # "7 days", "5 gb" — a number, but not money
        value = float(match.group("num").replace(",", ""))
        has_marker = bool(match.group("cur")) or bool(unit)
        candidates.append((has_marker, value))

    if not candidates:
        return None

    # An explicitly marked amount ("₹499", "499 rupees") beats a bare number
    # elsewhere in the sentence; otherwise take the first number given.
    for marked, value in candidates:
        if marked:
            return value
    return candidates[0][1]


# --- Confirmation interpretation (FR-36b, FR-36c, FR-36e) ----------------

CONFIRMED = "confirmed"
DECLINED = "declined"
UNCLEAR = "unclear"

_NEGATIVE = {
    "no", "nope", "nah", "cancel", "stop", "dont", "not", "never",
    "nevermind", "abort", "reject", "decline", "wait", "hold",
}
_STRONG_YES = {
    "yes", "yeah", "yep", "yup", "y", "confirm", "confirmed", "ok", "okay",
    "proceed", "sure", "affirmative",
}
# Words that may keep a "yes" company without changing what it means.
_FILLER = {
    "please", "thanks", "thank", "you", "lets", "let", "do", "it", "go",
    "ahead", "the", "refund", "my", "money", "now", "and", "that", "this",
    "one", "with", "for", "back", "process", "a", "s",
}
_EXPLICIT_YES_PHRASES = {"go ahead", "do it", "go for it", "process it", "refund it"}
# Hedges are checked before anything else, because they read as a "no" word-by-word
# ("I'm not sure") while meaning "ask me again".
_HEDGES = ("not sure", "unsure", "maybe", "i think", "dont know", "do not know", "perhaps")


def interpret_confirmation(message: str) -> str:
    """Classify a reply to "do you want to proceed?" — in code, not the model.

    Deliberately strict. Anything that is not an unmistakable yes or no comes
    back `unclear` so the agent asks again (FR-36e) instead of spending money
    on a "maybe". A negative word anywhere wins outright: "yes, no wait" is a
    customer changing their mind mid-sentence.
    """
    normalized = re.sub(r"[^\w\s']", " ", (message or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return UNCLEAR

    if normalized in _EXPLICIT_YES_PHRASES:
        return CONFIRMED
    if any(hedge in normalized for hedge in _HEDGES):
        return UNCLEAR

    tokens = [t.replace("'", "") for t in normalized.split()]
    if any(token in _NEGATIVE for token in tokens):
        return DECLINED
    if any(token in _STRONG_YES for token in tokens) and all(
        token in _STRONG_YES or token in _FILLER for token in tokens
    ):
        return CONFIRMED
    return UNCLEAR


# --- Graph state ---------------------------------------------------------


class RefundState(TypedDict, total=False):
    user_id: str
    message: str
    history: list[dict]
    pending: dict | None
    decision: bool | None  # set by the UI's Confirm / Cancel buttons
    amount: float | None
    account: dict | None
    validation: dict | None
    outcome: dict | None
    stage: str
    reply: str
    tool_calls: list[str]


@dataclass
class RefundTurn:
    """What one turn of the refund agent produced."""

    reply: str
    stage: str
    pending: dict | None = None
    amount: float | None = None
    validation: dict | None = None
    outcome: dict | None = None
    tool_calls: list[str] = field(default_factory=list)

    @property
    def awaits_confirmation(self) -> bool:
        return self.stage == AWAITING_CONFIRMATION


# --- The two tools the model is allowed to call --------------------------

AGENT_PROMPT = """You are NovaCell's refund assistant, working on one refund request \
for the signed-in customer.

Your job in this step is only to gather facts with your tools. You do not decide \
anything and you do not write the reply to the customer.

Do exactly this:
1. Call get_user_account to retrieve the customer's recharges.
2. Call validate_refund with the amount under consideration.

The amount under consideration is: {amount}

Never guess account details, and never judge eligibility yourself — validate_refund \
is the only thing that decides it. When both tools have been called, answer with the \
single word DONE."""


def _build_tools(user_id: str, record: dict[str, Any]):
    """Wrap the read-only tools for the model, bound to this session's user.

    The customer's id is closed over rather than being a tool argument, so the
    model has no way to read or validate against somebody else's account, and
    nothing the customer types about their own recharges can be substituted for
    what the account store actually says (FR-16, GR-03).
    """

    @tool
    def get_user_account() -> dict:
        """Retrieve the signed-in customer's account, plan and recharge history."""
        result = refund_tools.get_user_account(user_id)
        record["account"] = result
        record.setdefault("tool_calls", []).append("get_user_account")
        log_step("Agent", "get_user_account")
        return result

    @tool
    def validate_refund(amount: float) -> dict:
        """Check whether a refund of `amount` is eligible under the refund rules.

        Returns the eligibility decision and the recharge it applies to. This is
        the only thing that decides eligibility.
        """
        result = refund_tools.validate_refund(user_id, amount)
        record["validation"] = result
        record.setdefault("tool_calls", []).append(f"validate_refund(amount={amount:g})")
        log_step("Agent", f"validate_refund(amount={amount:g})")
        log_step("Validation", f"eligible={result['eligible']} — {result['reason']}")
        return result

    return [get_user_account, validate_refund]


# --- Nodes ---------------------------------------------------------------


def _resolve_amount(state: RefundState) -> dict:
    """Work out which amount this turn is about (FR-07, FR-41)."""
    amount = extract_amount(state.get("message", ""))

    # Mid-flow, the customer may be answering "which amount?" with just a
    # number; the earlier request carried no amount, so there is nothing to
    # fall back to and None correctly means "ask again".
    if amount is None:
        log_step("Agent", "no amount in the request")
    else:
        log_step("Agent", f"amount={format_rupees(amount)}")
    return {"amount": amount, "stage": REFUND_REQUESTED}


def _offer_amount(state: RefundState) -> dict:
    """No amount given — offer or ask, never assume (FR-41 – FR-44)."""
    account = refund_tools.get_user_account(state["user_id"])
    log_step("Agent", "get_user_account")

    if not account.get("found"):
        return {
            "account": account,
            "stage": REFUND_REJECTED,
            "reply": "I could not find your account, so I cannot look at a refund. "
            "Please call 611 or use the MyTelecom app.",
        }

    eligible = refund_tools.refundable_recharges(account["recharges"])

    # FR-42: exactly one candidate, so offer that amount and carry on — the
    # normal flow still asks for confirmation before anything happens.
    if len(eligible) == 1:
        amount = eligible[0]["recharge_amount"]
        log_step("Agent", f"single refundable recharge — offering {format_rupees(amount)}")
        return {"account": account, "amount": amount}

    # FR-42a
    if len(eligible) > 1:
        amounts = [format_rupees(r["recharge_amount"]) for r in eligible]
        listed = " and ".join([", ".join(amounts[:-1]), amounts[-1]]).strip(", ")
        log_step("Agent", AWAITING_AMOUNT)
        return {
            "account": account,
            "stage": AWAITING_AMOUNT,
            "reply": f"I can see more than one recent recharge — {listed}. "
            "Which amount would you like refunded?",
        }

    # FR-43: nothing refundable, so there is no amount to offer.
    log_step("Agent", "no refundable recharges")
    return {
        "account": account,
        "stage": REFUND_REJECTED,
        "reply": "I can't see a recharge on your account that is still within our "
        f"{REFUND_WINDOW_DAYS}-day refund window and not already refunded. If you "
        "think that is wrong, please call 611 or use the MyTelecom app.",
    }


def _gather(state: RefundState) -> dict:
    """The agentic step: let the model call the read-only tools (FR-06, FR-08).

    Whatever the model does or fails to do, the node ends with a real account
    and a real validation for the amount in play — if the model skipped a tool,
    timed out, or validated the wrong number, the same tools are called here
    directly. The graph downstream reads only these results, so correctness
    never depends on the model having behaved.
    """
    user_id = state["user_id"]
    amount = state.get("amount")
    record: dict[str, Any] = {}

    try:
        from chain import get_llm

        tools = _build_tools(user_id, record)
        llm = get_llm().bind_tools(tools)
        by_name = {t.name: t for t in tools}

        messages: list = [
            SystemMessage(
                AGENT_PROMPT.format(
                    amount=format_rupees(amount) if amount is not None else "not yet known"
                )
            ),
            HumanMessage(
                f"Conversation so far:\n{as_transcript(state.get('history') or [])}\n\n"
                f"Latest customer message: {state.get('message', '')}"
            ),
        ]

        for _ in range(MAX_TOOL_ROUNDS):
            response = llm.invoke(messages)
            messages.append(response)
            calls = getattr(response, "tool_calls", None) or []
            if not calls:
                break
            for call in calls:
                target = by_name.get(call["name"])
                if target is None:
                    continue
                output = target.invoke(call["args"])
                messages.append(
                    ToolMessage(content=str(output), tool_call_id=call["id"])
                )
    except Exception as exc:
        # A refund must not depend on the model being reachable.
        log_step("Agent", f"tool loop unavailable ({exc}) — calling tools directly")

    account = record.get("account")
    if not account or not account.get("found"):
        account = refund_tools.get_user_account(user_id)
        log_step("Agent", "get_user_account (direct)")

    validation = record.get("validation")
    if not validation or validation.get("requested_amount") != amount:
        validation = refund_tools.validate_refund(user_id, amount)
        log_step("Agent", f"validate_refund(amount={amount:g}) (direct)")
        log_step("Validation", f"eligible={validation['eligible']} — {validation['reason']}")

    return {
        "account": account,
        "validation": validation,
        "stage": REFUND_VALIDATED,
        "tool_calls": record.get("tool_calls", []),
    }


def _decide(state: RefundState) -> dict:
    """Deterministic routing on the validation result and the ₹499 limit."""
    validation = state["validation"]
    amount = state["amount"]

    # FR-09, GR-01
    if not validation["eligible"]:
        log_step("Agent", REFUND_REJECTED)
        return {
            "stage": REFUND_REJECTED,
            "reply": f"I can't refund {format_rupees(amount)}. {validation['reason']}\n\n"
            "If you think that is wrong, our support team can take another look — "
            "call 611 or use the MyTelecom app.",
        }

    # FR-37 – FR-40a, GR-08, GR-09: above the limit this application cannot
    # refund at all, so there is nothing for the customer to approve.
    if amount > APPROVAL_LIMIT:
        outcome = refund_tools.submit_to_support(state["user_id"], amount)
        log_step("Agent", f"submit_to_support(amount={amount:g})")
        log_step("Support", f"submitted ticket={outcome.get('ticket_id')}")
        return {
            "outcome": outcome,
            "stage": REFUND_SUBMITTED_FOR_REVIEW,
            "reply": f"Your refund request for {format_rupees(amount)} has been "
            "submitted to our Support Team.\n\nThey will review the request and "
            "process it or take the necessary action.\n\n"
            f"Reference: `{outcome.get('ticket_id')}`",
        }

    # FR-11, FR-36: eligible and within the limit — ask, and stop here.
    log_step("Agent", AWAITING_CONFIRMATION)
    recharge_id = validation.get("recharge_id")
    age = validation.get("recharge_age_days")
    return {
        "stage": AWAITING_CONFIRMATION,
        "reply": f"**Refund eligible: {format_rupees(amount)}**\n\n"
        f"This is against your recharge `{recharge_id}` from "
        f"{age} day{'s' if age != 1 else ''} ago.\n\n"
        "Do you want to proceed with this refund?",
    }


def _confirm(state: RefundState) -> dict:
    """Handle the answer to the confirmation question (FR-12, FR-32, GR-02)."""
    pending = state.get("pending") or {}
    amount = pending.get("amount")
    decision = state.get("decision")

    if amount is None:
        # Nothing was ever offered, so there is nothing to confirm. Start over
        # rather than guess what the "yes" was for.
        log_step("Agent", "confirmation with no pending amount — restarting")
        return {
            "stage": REFUND_REJECTED,
            "reply": "I have lost track of which refund that was about. How much "
            "would you like refunded?",
        }

    if decision is None:
        verdict = interpret_confirmation(state.get("message", ""))
    else:
        verdict = CONFIRMED if decision else DECLINED

    if verdict == UNCLEAR:
        # FR-36e: ambiguity is not consent.
        log_step("User", "ambiguous reply — asking again")
        return {
            "amount": amount,
            "stage": AWAITING_CONFIRMATION,
            "reply": "Sorry, I need a clear answer before I touch anything. Should I "
            f"refund {format_rupees(amount)} — yes or no?",
        }

    if verdict == DECLINED:
        # FR-36d
        log_step("User", "declined")
        log_step("Agent", REFUND_CANCELLED)
        return {
            "amount": amount,
            "stage": REFUND_CANCELLED,
            "reply": f"No problem — I have not processed anything. Your "
            f"{format_rupees(amount)} recharge is unchanged. Let me know if you "
            "need anything else.",
        }

    log_step("User", "confirmed")

    # Revalidate at the moment of execution rather than trusting the decision
    # taken a turn ago: the account may have changed since, and this is the
    # call that spends money (GR-01, GR-04, GR-06).
    validation = refund_tools.validate_refund(state["user_id"], amount)
    log_step("Validation", f"eligible={validation['eligible']} — {validation['reason']}")
    if not validation["eligible"] or amount > APPROVAL_LIMIT:
        log_step("Agent", REFUND_REJECTED)
        return {
            "amount": amount,
            "validation": validation,
            "stage": REFUND_REJECTED,
            "reply": f"I could not process that refund after all. {validation['reason']}",
        }

    outcome = refund_tools.process_refund(state["user_id"], amount)
    log_step("Agent", f"process_refund(amount={amount:g})")

    if not outcome.get("success"):
        log_step("Refund", f"failed — {outcome.get('reason')}")
        return {
            "amount": amount,
            "outcome": outcome,
            "stage": REFUND_REJECTED,
            "reply": f"I could not process that refund. {outcome.get('reason')}",
        }

    log_step("Refund", f"success ref={outcome['refund_id']}")
    return {
        "amount": amount,
        "validation": validation,
        "outcome": outcome,
        "stage": REFUND_PROCESSED,
        "reply": f"Done — your {format_rupees(amount)} refund has been processed "
        f"against recharge `{outcome['recharge_id']}`.\n\n"
        f"Refund reference: `{outcome['refund_id']}`",
    }


# --- Graph ---------------------------------------------------------------


def _entry(state: RefundState) -> str:
    pending = state.get("pending") or {}
    if pending.get("stage") == AWAITING_CONFIRMATION:
        return "confirm"
    return "resolve_amount"


def _after_resolve(state: RefundState) -> str:
    return "gather" if state.get("amount") is not None else "offer_amount"


def _after_offer(state: RefundState) -> str:
    # `_offer_amount` either settled on an amount to pursue or already wrote
    # the reply that ends the turn.
    return "gather" if state.get("reply") is None else "end"


def _build_graph():
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(RefundState)
    builder.add_node("resolve_amount", _resolve_amount)
    builder.add_node("offer_amount", _offer_amount)
    builder.add_node("gather", _gather)
    builder.add_node("decide", _decide)
    builder.add_node("confirm", _confirm)

    builder.add_conditional_edges(
        START, _entry, {"confirm": "confirm", "resolve_amount": "resolve_amount"}
    )
    builder.add_conditional_edges(
        "resolve_amount", _after_resolve, {"gather": "gather", "offer_amount": "offer_amount"}
    )
    builder.add_conditional_edges(
        "offer_amount", _after_offer, {"gather": "gather", "end": END}
    )
    builder.add_edge("gather", "decide")
    builder.add_edge("decide", END)
    builder.add_edge("confirm", END)
    return builder.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


# --- Public entry point --------------------------------------------------


def run(
    message: str,
    *,
    pending: dict | None = None,
    history: list[dict] | None = None,
    user_id: str = DEMO_USER_ID,
    decision: bool | None = None,
) -> RefundTurn:
    """Run one turn of the refund workflow.

    `pending` is what the previous turn returned as `RefundTurn.pending`, and
    `decision` short-circuits the natural-language reading of a confirmation
    when the customer used the Confirm / Cancel buttons instead.
    """
    result = get_graph().invoke(
        {
            "user_id": user_id,
            "message": message,
            "history": history or [],
            "pending": pending,
            "decision": decision,
            "reply": None,
        }
    )

    stage = result.get("stage", REFUND_REJECTED)
    amount = result.get("amount")
    next_pending = (
        {"stage": stage, "amount": amount, "validation": result.get("validation")}
        if stage in PENDING_STAGES
        else None
    )

    return RefundTurn(
        reply=result.get("reply") or "Sorry — I could not complete that request.",
        stage=stage,
        pending=next_pending,
        amount=amount,
        validation=result.get("validation"),
        outcome=result.get("outcome"),
        tool_calls=result.get("tool_calls") or [],
    )
