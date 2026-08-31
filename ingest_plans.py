"""Ingest the plan and pricing catalog: one JSON plan becomes one document.

The catalog is the source of truth for anything with a price on it — plans,
family bundles, specialty plans and add-ons like roaming passes. Plan questions
were previously answered from the FAQ, tickets and guide, none of which carry
the current lineup, so the bot had nothing accurate to retrieve.

Each entry is rendered as a labelled block of plain text rather than dumped as
JSON: the embedding model was trained on prose, and "unlimited 5G data" matches
a customer asking for an unlimited plan far better than `"data_unlimited": true`.

Three summary documents are generated on top of the per-plan ones. Comparison
questions ("what is your *cheapest* unlimited plan?", "day pass or monthly
pass?") cannot be answered from any single plan document, because cheapest is a
fact about the lineup, not about one plan. The summaries hold those cross-plan
views in retrievable form, derived from the same JSON so they cannot drift
from it.

Run:  python ingest_plans.py [--reset]
"""

from __future__ import annotations

import argparse
import json

from langchain_core.documents import Document

from config import PLANS_COLLECTION, PLANS_JSON
from vectorstore import count, reset_collection, upsert

# --- Field rendering -----------------------------------------------------


def _money(amount: float, currency: str) -> str:
    """`45.0` -> `$45` — trailing `.0` reads like a typo in a price."""
    symbol = "$" if currency == "USD" else f"{currency} "
    text = f"{amount:.2f}".rstrip("0").rstrip(".")
    return f"{symbol}{text}"


def _price_line(plan: dict, currency: str) -> str:
    """Plans price monthly; add-ons carry their own unit (per day, per line)."""
    if plan.get("monthly_price") is not None:
        price = f"{_money(plan['monthly_price'], currency)} per month"
        if plan.get("price_per_line"):
            price += (
                f" total, which is {_money(plan['price_per_line'], currency)} per line "
                f"for {plan['lines_included']} lines"
            )
        elif plan.get("lines_included"):
            price += f" total for {plan['lines_included']} lines"
        return price
    if plan.get("price") is not None:
        return f"{_money(plan['price'], currency)} {plan.get('price_unit', 'per month')}"
    return "Contact us for pricing"


def _data_line(plan: dict) -> str | None:
    network = plan.get("network")
    if plan.get("data_unlimited"):
        return f"Unlimited {network} data" if network else "Unlimited data"
    allowance = plan.get("data_gb")
    if allowance is None:
        return None
    if isinstance(allowance, str):  # e.g. "uses your plan's high-speed data"
        return f"Data abroad: {allowance}"
    return f"{allowance} GB of high-speed data per month"


def _talk_text_line(plan: dict) -> str | None:
    talk, texts = plan.get("talk_minutes"), plan.get("texts")
    if talk in (0, None) and texts in (0, None):
        return "No talk or text included" if talk == 0 else None
    return f"Talk: {talk}. Texts: {texts}."


def plan_to_text(plan: dict, currency: str) -> str:
    """Render one catalog entry as the text that gets embedded."""
    kind = "add-on" if plan.get("type") == "add-on" else f"{plan.get('type')} plan"
    lines = [
        f"NovaCell {kind}: {plan['name']}",
        f"Price: {_price_line(plan, currency)}",
    ]

    for value in (_data_line(plan), _talk_text_line(plan)):
        if value:
            lines.append(value)

    hotspot = plan.get("hotspot_gb")
    if hotspot is not None:
        lines.append(
            f"Mobile hotspot: {hotspot} GB included"
            if hotspot
            else "Mobile hotspot: not included"
        )

    optional = (
        ("Network", plan.get("network")),
        ("Coverage", plan.get("coverage")),
        ("Contract", plan.get("contract")),
        ("Eligibility requirement", plan.get("eligibility")),
        ("Intro offer", plan.get("intro_offer")),
        ("Best for", plan.get("best_for")),
    )
    lines.extend(f"{label}: {value}" for label, value in optional if value)

    if plan.get("features"):
        lines.append("What is included:")
        lines.extend(f"- {feature}" for feature in plan["features"])

    lines.append(f"Category: {plan.get('category')}. Plan code: {plan['id']}.")
    return "\n".join(lines)


# --- Catalog overview ----------------------------------------------------


def _contract_note(plan: dict) -> str:
    contract = plan.get("contract")
    return "no contract" if contract == "none" else str(contract)


def _lines_note(plan: dict, currency: str) -> str:
    """Per-line price, so a multi-line plan is not judged on its total alone."""
    if plan.get("price_per_line"):
        return (
            f", covering {plan['lines_included']} lines at "
            f"{_money(plan['price_per_line'], currency)} per line"
        )
    if plan.get("lines_included"):
        return f", covering {plan['lines_included']} lines"
    return ""


def _eligibility_note(plan: dict) -> str:
    if plan.get("eligibility"):
        return f" — eligibility required: {plan['eligibility']}"
    return " — open to any customer, no eligibility requirement"


def build_summary_documents(catalog: dict) -> list[tuple[str, str, str]]:
    """Build the cross-plan summaries as `(id suffix, citation, text)`.

    Three narrow summaries rather than one catalog-wide document: a single
    overview covering everything from the cheapest prepaid plan to hotspot
    add-ons matches every plan question weakly and none of them strongly. Split
    by the comparison a customer is actually making, each one embeds close to
    the question that needs it.
    """
    currency = catalog.get("currency", "USD")
    plans = catalog["plans"]

    monthly = sorted(
        (p for p in plans if p.get("monthly_price") is not None),
        key=lambda p: p["monthly_price"],
    )
    add_ons = [p for p in plans if p.get("monthly_price") is None]
    unlimited = [p for p in monthly if p.get("data_unlimited")]

    summaries: list[tuple[str, str, str]] = []

    # 1. The whole lineup, price-ordered.
    lines = [
        f"NovaCell plan prices — every monthly plan we sell, cheapest first. "
        f"Prices in {currency}, last updated {catalog.get('last_updated')}.",
    ]
    for plan in monthly:
        data = "unlimited data" if plan.get("data_unlimited") else f"{plan.get('data_gb')} GB"
        lines.append(
            f"- {plan['name']}: {_money(plan['monthly_price'], currency)}/month, "
            f"{data}, {plan.get('network')}, {plan.get('type')}"
            f"{_lines_note(plan, currency)}{_eligibility_note(plan)}"
        )
    if monthly:
        lines.append(
            f"The lowest-priced plan of any kind is {monthly[0]['name']} at "
            f"{_money(monthly[0]['monthly_price'], currency)} per month."
        )
    summaries.append(("summary-prices", "Plan catalog — all plans by price", "\n".join(lines)))

    # 2. Unlimited plans, where "cheapest" needs the eligibility caveat.
    lines = ["NovaCell unlimited plans compared — every plan with unlimited data, cheapest first."]
    for plan in unlimited:
        lines.append(
            f"- {plan['name']}: {_money(plan['monthly_price'], currency)}/month, "
            f"{plan.get('type')}, {plan.get('hotspot_gb')} GB hotspot"
            f"{_lines_note(plan, currency)}{_eligibility_note(plan)}"
        )
    unrestricted = [p for p in unlimited if not p.get("eligibility")]
    if unrestricted:
        cheapest = unrestricted[0]
        lines.append(
            f"The cheapest unlimited plan any customer can buy, with no eligibility "
            f"requirement, is {cheapest['name']} at "
            f"{_money(cheapest['monthly_price'], currency)} per month "
            f"({_contract_note(cheapest)})."
        )
    restricted = [p for p in unlimited if p.get("eligibility")]
    for plan in restricted:
        if plan["monthly_price"] < (unrestricted[0]["monthly_price"] if unrestricted else 0):
            lines.append(
                f"{plan['name']} is cheaper still at "
                f"{_money(plan['monthly_price'], currency)} per month, but only for "
                f"customers who qualify: {plan['eligibility']}."
            )
    summaries.append(
        ("summary-unlimited", "Plan catalog — unlimited plans compared", "\n".join(lines))
    )

    # 3. Add-ons, including the two roaming passes customers weigh up.
    lines = [
        "NovaCell add-ons and passes — extras bought on top of a plan, "
        "including international roaming and calling.",
    ]
    for plan in add_ons:
        lines.append(
            f"- {plan['name']}: {_money(plan['price'], currency)} "
            f"{plan.get('price_unit')} — {plan.get('best_for')}"
        )
    day = next((p for p in add_ons if p["id"] == "ADDON-INTL-DAYPASS"), None)
    month = next((p for p in add_ons if p["id"] == "ADDON-INTL-MONTHLY"), None)
    if day and month:
        break_even = month["price"] / day["price"]
        lines.append(
            f"Choosing a roaming pass: the Day Pass costs "
            f"{_money(day['price'], currency)} for each day you use your phone abroad, "
            f"so a trip of {break_even:.0f} days or more costs at least as much as the "
            f"{_money(month['price'], currency)} Monthly Pass. Short trips favour the "
            f"Day Pass; trips of about a week or longer favour the Monthly Pass."
        )
    summaries.append(("summary-addons", "Plan catalog — add-ons and roaming passes", "\n".join(lines)))

    return summaries


# --- Loading -------------------------------------------------------------


def load_plan_documents() -> tuple[list[Document], list[str]]:
    if not PLANS_JSON.exists():
        raise FileNotFoundError(f"Plan catalog not found: {PLANS_JSON}")

    catalog = json.loads(PLANS_JSON.read_text(encoding="utf-8"))
    currency = catalog.get("currency", "USD")
    plans = catalog.get("plans") or []
    if not plans:
        raise ValueError(f"No plans found in {PLANS_JSON}")

    documents: list[Document] = []
    ids: list[str] = []

    for plan in plans:
        documents.append(
            Document(
                page_content=plan_to_text(plan, currency),
                metadata={
                    "source": "plans",
                    "plan_id": plan["id"],
                    "plan_name": plan["name"],
                    "category": plan.get("category", "plan"),
                    "type": plan.get("type", "plan"),
                    "citation": f"Plan catalog — {plan['name']}",
                },
            )
        )
        ids.append(f"plan-{plan['id']}")

    for suffix, citation, text in build_summary_documents(catalog):
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": "plans",
                    "plan_id": suffix,
                    "plan_name": citation,
                    "category": "summary",
                    "type": "summary",
                    "citation": citation,
                },
            )
        )
        ids.append(f"plan-{suffix}")

    return documents, ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the plan catalog into ChromaDB.")
    parser.add_argument("--reset", action="store_true", help="drop the collection first")
    args = parser.parse_args()

    if args.reset:
        reset_collection(PLANS_COLLECTION)

    documents, ids = load_plan_documents()
    upsert(PLANS_COLLECTION, documents, ids)
    print(
        f"plans: ingested {len(documents) - 3} plans and add-ons plus 3 catalog "
        f"summaries -> {count(PLANS_COLLECTION)} vectors in store"
    )


if __name__ == "__main__":
    main()
