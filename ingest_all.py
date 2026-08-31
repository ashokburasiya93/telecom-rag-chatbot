"""Run every ingest script in one go.

Run:  python ingest_all.py [--reset]
"""

from __future__ import annotations

import argparse

import ingest_faq
import ingest_guides
import ingest_plans
import ingest_tickets
from config import (
    FAQ_COLLECTION,
    GUIDES_COLLECTION,
    PLANS_COLLECTION,
    TICKETS_COLLECTION,
)
from vectorstore import count, reset_collection, upsert

LOADERS = (
    (FAQ_COLLECTION, ingest_faq.load_faq_documents),
    (TICKETS_COLLECTION, ingest_tickets.load_ticket_documents),
    (GUIDES_COLLECTION, ingest_guides.load_guide_documents),
    (PLANS_COLLECTION, ingest_plans.load_plan_documents),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest all knowledge sources.")
    parser.add_argument("--reset", action="store_true", help="drop collections first")
    args = parser.parse_args()

    for collection, loader in LOADERS:
        if args.reset:
            reset_collection(collection)
        documents, ids = loader()
        upsert(collection, documents, ids)
        print(f"{collection:<8} {len(documents):>4} documents -> {count(collection):>4} vectors")

    print("\nIngestion complete. Start the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
