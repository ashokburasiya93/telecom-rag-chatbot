"""The agent PRD's seven demonstration scenarios, as a runnable check (§8).

    python test_scenarios.py            # everything
    python test_scenarios.py --offline  # skip the two that need the LLM + index

Scenarios 1-4 and 6 drive the refund agent directly, so they exercise the real
graph, the real tools and the real guardrails. Scenarios 5 and 7 go through
intent detection and the RAG pipeline, which needs GROQ_API_KEY and an ingested
vector store.

`process_refund` is replaced by a spy throughout, because "was the refund
executed?" is the question most of these scenarios are actually asking.
"""

from __future__ import annotations

import copy
import sys
from contextlib import contextmanager

import accounts
import refund_agent
import refund_tools

PASSED: list[str] = []
FAILED: list[str] = []


def _make_output_crash_proof() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail and not condition else ""))


@contextmanager
def spy_on_process_refund():
    """Count calls to the one function that is allowed to move money."""
    original = refund_tools.process_refund
    calls: list[tuple] = []

    def wrapper(user_id, amount):
        calls.append((user_id, amount))
        return original(user_id, amount)

    refund_tools.process_refund = wrapper
    try:
        yield calls
    finally:
        refund_tools.process_refund = original


def seed(recharges: list[dict] | None = None) -> None:
    """Reset to the demo account, optionally with different recharges."""
    users = copy.deepcopy(accounts.SEED_USERS)
    if recharges is not None:
        users["user_123"]["recharges"] = copy.deepcopy(recharges)
    accounts.reset_accounts(users)


def recharge(user_id: str, recharge_id: str) -> dict:
    return accounts.find_recharge(user_id, recharge_id)


# --- Scenario 1 ----------------------------------------------------------


def scenario_1() -> None:
    print("\nScenario 1 — successful refund of ₹499")
    seed()
    with spy_on_process_refund() as calls:
        turn = refund_agent.run("I want a refund of ₹499.")
        check(
            "eligible refund at the limit asks for confirmation",
            turn.stage == refund_agent.AWAITING_CONFIRMATION,
            turn.stage,
        )
        check("nothing processed before confirmation", not calls, f"{calls}")

        history = [
            {"role": "user", "content": "I want a refund of ₹499."},
            {"role": "assistant", "content": turn.reply},
        ]
        done = refund_agent.run("Yes, proceed.", pending=turn.pending, history=history)
        check(
            "confirmation processes the refund",
            done.stage == refund_agent.REFUND_PROCESSED,
            done.stage,
        )
        check("process_refund called exactly once", len(calls) == 1, f"{calls}")
        check(
            "a refund reference comes back",
            bool((done.outcome or {}).get("refund_id")),
            f"{done.outcome}",
        )
        check(
            "simulated refund status is updated",
            recharge("user_123", "rch_457")["refund_status"] == "refunded",
        )
        check(
            "the other recharge is still refundable",
            refund_tools.validate_refund("user_123", 999)["eligible"],
        )
        check("no pending state is left behind", done.pending is None)


# --- Scenario 2 ----------------------------------------------------------


def scenario_2() -> None:
    print("\nScenario 2 — amount exceeds the recharge")
    seed()
    with spy_on_process_refund() as calls:
        turn = refund_agent.run("Give me a refund of ₹2,000.")
        check("rejected", turn.stage == refund_agent.REFUND_REJECTED, turn.stage)
        check(
            "reason names the excess",
            "more than" in turn.reply,
            turn.reply,
        )
        check("process_refund never called", not calls, f"{calls}")


# --- Scenario 3 ----------------------------------------------------------


def scenario_3() -> None:
    print("\nScenario 3 — recharge outside the 7-day window")
    seed(
        [
            {
                "recharge_id": "rch_457",
                "recharge_amount": 499,
                "recharge_status": "completed",
                "recharge_age_days": 15,
                "refund_status": "not_refunded",
            }
        ]
    )
    with spy_on_process_refund() as calls:
        turn = refund_agent.run("I want a refund of ₹499.")
        check("rejected", turn.stage == refund_agent.REFUND_REJECTED, turn.stage)
        check("reason names the window", "refund window" in turn.reply, turn.reply)
        check("process_refund never called", not calls, f"{calls}")


# --- Scenario 4 ----------------------------------------------------------


def scenario_4() -> None:
    print("\nScenario 4 — recharge already refunded")
    seed(
        [
            {
                "recharge_id": "rch_457",
                "recharge_amount": 499,
                "recharge_status": "completed",
                "recharge_age_days": 1,
                "refund_status": "refunded",
            }
        ]
    )
    with spy_on_process_refund() as calls:
        turn = refund_agent.run("I want a refund of ₹499.")
        check("rejected", turn.stage == refund_agent.REFUND_REJECTED, turn.stage)
        check("reason names the earlier refund", "already been refunded" in turn.reply)
        check("process_refund never called", not calls, f"{calls}")


# --- Scenario 5 ----------------------------------------------------------


def scenario_5() -> None:
    print("\nScenario 5 — an ordinary support question still goes to RAG")
    from router import RAG, route

    destination = route("Why is my mobile internet slow?")
    check("routed to the existing pipeline", destination == RAG, destination)

    from chain import answer

    docs, text = answer("Why is my mobile internet slow?")
    check("retrieval returned documents", len(docs) > 0, f"{len(docs)} docs")
    check("an answer came back", len(text.strip()) > 0)
    print(f"       → {text.strip()[:160]}…")


# --- Scenario 6 ----------------------------------------------------------


def scenario_6() -> None:
    print("\nScenario 6 — refund above the approval limit")
    seed()
    with spy_on_process_refund() as calls:
        turn = refund_agent.run("I want a refund of ₹999.")
        check(
            "submitted for review",
            turn.stage == refund_agent.REFUND_SUBMITTED_FOR_REVIEW,
            turn.stage,
        )
        check("process_refund never called", not calls, f"{calls}")
        check(
            "refund status unchanged",
            recharge("user_123", "rch_456")["refund_status"] == "not_refunded",
        )
        check("no confirmation step", turn.pending is None)
        lowered = turn.reply.lower()
        check("the reply says it was submitted", "submitted" in lowered, turn.reply)
        check(
            "no outcome is claimed on the Support Team's behalf",
            not any(
                word in lowered for word in ("approved", "rejected", "has been refunded")
            ),
            turn.reply,
        )


# --- Scenario 7 ----------------------------------------------------------


def scenario_7() -> None:
    print("\nScenario 7 — follow-up question resolved from history")
    from chain import answer
    from router import RAG, route

    first = "How do I activate international roaming?"
    _, first_answer = answer(first)
    history = [
        {"role": "user", "content": first},
        {"role": "assistant", "content": first_answer},
    ]

    follow_up = "Should I do that before I travel?"
    destination = route(follow_up, None, history)
    check("follow-up still routes to RAG", destination == RAG, destination)

    _, text = answer(follow_up, history)
    lowered = text.lower()
    check(
        "the answer is about roaming, not a request for clarification",
        "roam" in lowered or "travel" in lowered,
        text[:200],
    )
    print(f"       → {text.strip()[:200]}…")


# --- Extra guardrail checks ---------------------------------------------


def guardrails() -> None:
    print("\nGuardrails and the missing-amount path")
    seed()
    with spy_on_process_refund() as calls:
        ask = refund_agent.run("I want a refund.")
        check(
            "two eligible recharges means we ask which",
            ask.stage == refund_agent.AWAITING_AMOUNT,
            ask.stage,
        )
        check("both amounts offered", "₹999" in ask.reply and "₹499" in ask.reply, ask.reply)
        check("no amount invented", ask.amount is None)

        chosen = refund_agent.run("₹499 please", pending=ask.pending)
        check(
            "naming the amount moves on to confirmation",
            chosen.stage == refund_agent.AWAITING_CONFIRMATION,
            chosen.stage,
        )

        vague = refund_agent.run("Maybe, I'm not sure.", pending=chosen.pending)
        check(
            "an ambiguous answer asks again instead of executing",
            vague.stage == refund_agent.AWAITING_CONFIRMATION,
            vague.stage,
        )
        check("still nothing processed", not calls, f"{calls}")

        declined = refund_agent.run("No, cancel the refund.", pending=vague.pending)
        check("declining cancels", declined.stage == refund_agent.REFUND_CANCELLED, declined.stage)
        check("declining processed nothing", not calls, f"{calls}")
        check(
            "recharge untouched after cancelling",
            recharge("user_123", "rch_457")["refund_status"] == "not_refunded",
        )

    # GR-08: even called directly, the tool refuses above the limit.
    seed()
    blocked = refund_tools.process_refund("user_123", 999)
    check("process_refund refuses above the limit", not blocked["success"], f"{blocked}")
    check(
        "and changes nothing",
        recharge("user_123", "rch_456")["refund_status"] == "not_refunded",
    )

    # FR-23a: the model is only ever handed the two read-only tools.
    exposed = {t.name for t in refund_agent._build_tools("user_123", {})}
    check(
        "only read-only tools are bound to the LLM",
        exposed == {"get_user_account", "validate_refund"},
        f"{exposed}",
    )

    # FR-44 / FR-07: amounts come from the customer's words, never invented.
    check("no amount in 'I want a refund'", refund_agent.extract_amount("I want a refund") is None)
    check("₹1,000 parsed", refund_agent.extract_amount("refund ₹1,000 please") == 1000)
    check("2000 parsed", refund_agent.extract_amount("Give me a refund of 2000") == 2000)
    check(
        "units are not amounts",
        refund_agent.extract_amount("my 2 day old recharge with 5 GB left") is None,
    )
    check(
        "recharge ids are not amounts",
        refund_agent.extract_amount("refund rch_457") is None,
    )


def main() -> int:
    _make_output_crash_proof()
    offline = "--offline" in sys.argv

    scenario_1()
    scenario_2()
    scenario_3()
    scenario_4()
    if offline:
        print("\nScenarios 5 and 7 skipped (--offline)")
    else:
        scenario_5()
    scenario_6()
    if not offline:
        scenario_7()
    guardrails()

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for name in FAILED:
        print(f"  failed: {name}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
