"""Simulated account and recharge data for the refund agent (FR-14 – FR-18a).

There is no CRM and no billing system behind this. One hard-coded user stands
in for an authenticated session (FR-15, FR-16) and the only thing that ever
changes is a recharge's `refund_status`, which `refund_tools.process_refund`
flips when a simulated refund succeeds (FR-33).

The store is module-level, so a refund persists for the life of the server
process — that is what makes "already refunded" observable in a demo rather
than something you have to take on faith. `reset_accounts()` puts the seed
back.
"""

from __future__ import annotations

from copy import deepcopy

# Treated as if it arrived from authenticated session context. The user is
# never asked for it (FR-17).
DEMO_USER_ID = "user_123"

# Business constants. Both are enforced in code, never by the LLM (FR-29,
# FR-36a).
REFUND_WINDOW_DAYS = 7
APPROVAL_LIMIT = 499

SEED_USERS: dict[str, dict] = {
    "user_123": {
        "user_id": "user_123",
        "name": "Rahul",
        "plan": "Pro",
        "plan_amount": 999,
        "recharges": [
            {
                "recharge_id": "rch_456",
                "recharge_amount": 999,
                "recharge_status": "completed",
                "recharge_age_days": 2,
                "refund_status": "not_refunded",
            },
            {
                "recharge_id": "rch_457",
                "recharge_amount": 499,
                "recharge_status": "completed",
                "recharge_age_days": 1,
                "refund_status": "not_refunded",
            },
        ],
    }
}

_accounts: dict[str, dict] = deepcopy(SEED_USERS)


def get_account(user_id: str) -> dict | None:
    """Return a copy of the stored account, or None.

    A copy, not the live dict: callers pass this into prompts and UI state, and
    nothing outside this module should be able to edit account data by holding
    on to a reference (GR-03).
    """
    account = _accounts.get(user_id)
    return deepcopy(account) if account else None


def find_recharge(user_id: str, recharge_id: str) -> dict | None:
    account = _accounts.get(user_id)
    if not account:
        return None
    for recharge in account["recharges"]:
        if recharge["recharge_id"] == recharge_id:
            return deepcopy(recharge)
    return None


def mark_refunded(user_id: str, recharge_id: str) -> bool:
    """Flip one recharge to `refunded`. Returns False if it already was.

    The last line of defence against a double refund (GR-06): even if two
    confirmations raced, only the first one gets True back.
    """
    account = _accounts.get(user_id)
    if not account:
        return False
    for recharge in account["recharges"]:
        if recharge["recharge_id"] == recharge_id:
            if recharge["refund_status"] == "refunded":
                return False
            recharge["refund_status"] = "refunded"
            return True
    return False


def reset_accounts(users: dict[str, dict] | None = None) -> None:
    """Restore the seed data, or install a scenario's data in its place."""
    global _accounts
    _accounts = deepcopy(users if users is not None else SEED_USERS)
