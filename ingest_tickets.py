"""Ingest resolved tickets (FR-15): one SQLite row becomes one vector document.

Only resolved tickets are indexed — an open ticket has no resolution to teach
the model, and half-finished troubleshooting would be worse than no answer.

Run:  python ingest_tickets.py [--reset]
"""

from __future__ import annotations

import argparse
import sqlite3

from langchain_core.documents import Document

from config import TICKETS_COLLECTION, TICKETS_DB
from vectorstore import count, reset_collection, upsert


def load_ticket_documents() -> tuple[list[Document], list[str]]:
    if not TICKETS_DB.exists():
        raise FileNotFoundError(f"Ticket database not found: {TICKETS_DB}")

    connection = sqlite3.connect(TICKETS_DB)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT ticket_id, category, issue_type, description, resolution, status
            FROM tickets
            WHERE lower(status) = 'resolved'
            ORDER BY id
            """
        ).fetchall()
    finally:
        connection.close()

    documents: list[Document] = []
    ids: list[str] = []

    for row in rows:
        content = (
            f"Issue: {row['issue_type']}\n"
            f"Reported: {row['description']}\n"
            f"Resolution: {row['resolution']}"
        )
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": "tickets",
                    "ticket_id": row["ticket_id"],
                    "category": row["category"],
                    "issue_type": row["issue_type"],
                    "citation": f"Ticket {row['ticket_id']} — {row['issue_type']}",
                },
            )
        )
        ids.append(f"ticket-{row['ticket_id']}")

    return documents, ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest resolved tickets into ChromaDB.")
    parser.add_argument("--reset", action="store_true", help="drop the collection first")
    args = parser.parse_args()

    if args.reset:
        reset_collection(TICKETS_COLLECTION)

    documents, ids = load_ticket_documents()
    upsert(TICKETS_COLLECTION, documents, ids)
    print(
        f"tickets: ingested {len(documents)} resolved tickets -> "
        f"{count(TICKETS_COLLECTION)} vectors in store"
    )


if __name__ == "__main__":
    main()
