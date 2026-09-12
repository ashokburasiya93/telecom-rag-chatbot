"""The four refund tools (FR-19 – FR-23a) and the rules behind them.

Plain functions, no LangChain: every rule in here is deterministic application
logic that the LLM can neither influence nor overrule (FR-29, FR-30, GR-05).
`refund_agent` wraps the two read-only tools so the model can call them; the two
that change something — `process_refund` and `submit_to_support` — are never
bound to the model (FR-23a) and are reached only from controller code that has
already checked validation, the approval limit and the user's confirmation.
"""

from __future__ import annotations

import uuid

from accounts import APPROVAL_LIMIT, REFUND_WINDOW_DAYS, get_account, mark_refunded

REFUNDED = "refunded"
NOT_REFUNDED = "not_refunded"
COMPLETED = "completed"


def format_rupees(amount: float | int) -> str:
    """₹499, ₹1,000, ₹49.50 — never a bare number in customer-facing text."""
    if float(amount) == int(amount):
        return f"₹{int(amount):,}"
    return f"₹{amount:,.2f}"


# --- FR-19 ---------------------------------------------------------------


def get_user_account(user_id: str) -> dict:
    """Retrieve the account and its recharges (FR-19, FR-22)."""
    account = get_account(user_id)
    if not account:
        return {"found": False, "reason": f"No account found for {user_id}."}
    return {"found": True, **account}


# --- FR-27a: which recharge does this refund apply to? -------------------


def _by_recency(recharges: list[dict]) -> list[dict]:
    """Most recent first, stable within a tie so the choice never depends on
    the order two equally-aged recharges happened to be stored in."""
    return [
        recharge
        for _, recharge in sorted(
            enumerate(recharges),
            key=lambda pair: (pair[1]["recharge_age_days"], pair[0]),
        )
    ]


def select_recharge(recharges: list[dict], amount: float) -> dict | None:
    """Pick the recharge a refund of `amount` applies to (FR-27a).

    Deterministic, in this order:

    1. An exact amount match wins, even if that recharge is already refunded —
       asking for ₹499 back twice has to come back "already refunded", not get
       quietly redirected to the ₹999 recharge.
    2. Otherwise the most recent recharge that is not yet refunded and is large
       enough to cover the amount.
    3. Otherwise no recharge can cover the amount, and the pick only decides
       which one the rejection reason quotes — so quote the largest, because
       "₹2,000 is more than the ₹999 recharge" tells the customer more than
       the same sentence about a ₹499 one.
    """
    if not recharges:
        return None

    ordered = _by_recency(recharges)

    exact = [r for r in ordered if r["recharge_amount"] == amount]
    if exact:
        return exact[0]

    coverable = [
        r
        for r in ordered
        if r["refund_status"] != REFUNDED and r["recharge_amount"] >= amount
    ]
    if coverable:
        return coverable[0]

    return max(ordered, key=lambda r: r["recharge_amount"])


def refundable_recharges(recharges: list[dict]) -> list[dict]:
    """Recharges that would pass validation for their own full amount.

    Used to work out what to offer when the user did not name an amount
    (FR-41, FR-42, FR-42a).
    """
    return [
        r
        for r in _by_recency(recharges)
        if r["recharge_status"] == COMPLETED
        and r["refund_status"] != REFUNDED
        and r["recharge_age_days"] <= REFUND_WINDOW_DAYS
    ]


# --- FR-20, FR-24 – FR-30 ------------------------------------------------


def _result(eligible: bool, reason: str, **extra) -> dict:
    return {"eligible": eligible, "reason": reason, **extra}


def validate_refund(user_id: str, amount: float) -> dict:
    """Decide whether a refund of `amount` is eligible (FR-20, FR-22).

    Eligibility only. Whether an eligible refund can be executed here is a
    separate question, answered by the approval limit in §6.7 — not by this
    function.
    """
    account = get_account(user_id)
    if not account:
        return _result(False, "We could not find your account.", requested_amount=amount)

    # FR-26
    if amount is None or amount <= 0:
        return _result(
            False,
            "A refund amount has to be greater than zero.",
            requested_amount=amount,
        )

    recharges = account.get("recharges") or []
    if not recharges:
        return _result(
            False, "There are no recharges on your account.", requested_amount=amount
        )

    recharge = select_recharge(recharges, amount)
    common = {
        "requested_amount": amount,
        "recharge_amount": recharge["recharge_amount"],
        "recharge_id": recharge["recharge_id"],
        "recharge_age_days": recharge["recharge_age_days"],
    }

    # FR-24
    if recharge["recharge_status"] != COMPLETED:
        return _result(
            False,
            f"That recharge is {recharge['recharge_status']}, not completed, so it "
            "cannot be refunded.",
            **common,
        )

    # FR-25
    if recharge["recharge_age_days"] > REFUND_WINDOW_DAYS:
        return _result(
            False,
            f"That recharge is {recharge['recharge_age_days']} days old, which is "
            f"outside our {REFUND_WINDOW_DAYS}-day refund window.",
            **common,
        )

    # FR-28
    if recharge["refund_status"] == REFUNDED:
        return _result(
            False,
            f"The {format_rupees(recharge['recharge_amount'])} recharge has already "
            "been refunded.",
            **common,
        )

    # FR-27
    if amount > recharge["recharge_amount"]:
        return _result(
            False,
            f"The requested amount of {format_rupees(amount)} is more than the "
            f"{format_rupees(recharge['recharge_amount'])} recharge it would come "
            "from.",
            **common,
        )

    return _result(True, "Refund is eligible.", **common)


# --- FR-21, FR-31 – FR-35: never bound to the LLM (FR-23a) ---------------


def process_refund(user_id: str, amount: float) -> dict:
    """Execute the simulated refund (FR-21, FR-23, FR-35).

    Revalidates and rechecks the approval limit before changing anything. The
    controller has already done both; doing them again here means no future
    caller can skip them (GR-01, GR-08).
    """
    validation = validate_refund(user_id, amount)
    if not validation["eligible"]:
        return {
            "success": False,
            "reason": validation["reason"],
            "amount": amount,
            "status": "rejected",
        }

    if amount > APPROVAL_LIMIT:
        return {
            "success": False,
            "reason": (
                f"{format_rupees(amount)} is above the "
                f"{format_rupees(APPROVAL_LIMIT)} limit this system can refund."
            ),
            "amount": amount,
            "status": "requires_support_review",
        }

    recharge_id = validation["recharge_id"]
    if not mark_refunded(user_id, recharge_id):
        return {
            "success": False,
            "reason": "That recharge has already been refunded.",
            "amount": amount,
            "status": "rejected",
        }

    # FR-33, FR-34
    return {
        "success": True,
        "refund_id": f"ref_{uuid.uuid4().hex[:6]}",
        "amount": amount,
        "recharge_id": recharge_id,
        "status": REFUNDED,
    }


def submit_to_support(user_id: str, amount: float) -> dict:
    """Hand a refund above the limit to the Support Team (FR-21a, FR-38).

    Records the submission and nothing else. The review and whatever follows it
    happen outside this application, so no refund status changes here and no
    outcome is invented (FR-40a, GR-09).
    """
    validation = validate_refund(user_id, amount)
    if not validation["eligible"]:
        return {
            "submitted": False,
            "reason": validation["reason"],
            "amount": amount,
            "status": "rejected",
        }

    return {
        "submitted": True,
        "ticket_id": f"sup_{uuid.uuid4().hex[:6]}",
        "amount": amount,
        "recharge_id": validation["recharge_id"],
        "status": "submitted_for_review",
    }
