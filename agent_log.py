"""Workflow tracing for the intent router and refund agent (§12).

One line per step, in the shape the PRD asks for:

    [Intent] refund_request
    [Agent] validate_refund(amount=499)
    [Validation] eligible=True
    [Refund] success

Steps go to the console, where whoever is running the demo can watch the flow,
and to the same JSONL log the chat interactions use, where they can be read
back later. Only workflow facts are recorded — never the model's private
reasoning.
"""

from __future__ import annotations

from interaction_log import log_event


def _print(line: str) -> None:
    """Print a step without ever letting the console encoding kill a request.

    A Windows console on cp1252 raises on `₹`, and a refund trace is full of
    them. The log file is UTF-8 and keeps the real text either way.
    """
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)


def log_step(stage: str, detail: str = "", **fields) -> None:
    """Record one workflow step. `stage` is the bracketed tag."""
    _print(f"[{stage}] {detail}".rstrip())
    log_event("agent_step", stage=stage, detail=detail, **fields)
