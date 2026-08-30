"""Append-only interaction log (FR-05a).

Every answered question and every 👍/👎 lands here as one JSON object per line,
which is enough for support ops to see what the bot is getting wrong and which
questions it could not answer.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from config import INTERACTION_LOG, LOG_DIR


def _write(record: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    with INTERACTION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_answer(
    *,
    interaction_id: str,
    question: str,
    answer: str,
    citations: list[str],
    latency_s: float,
    channel: str = "streamlit",
) -> None:
    _write(
        {
            "event": "answer",
            "interaction_id": interaction_id,
            "channel": channel,
            "question": question,
            "answer": answer,
            "citations": citations,
            "latency_s": round(latency_s, 3),
        }
    )


def log_feedback(*, interaction_id: str, question: str, rating: str) -> None:
    """rating is "up" or "down"."""
    _write(
        {
            "event": "feedback",
            "interaction_id": interaction_id,
            "question": question,
            "rating": rating,
        }
    )
