"""Ingest the FAQ (FR-14): one CSV row becomes one vector document.

Run:  python ingest_faq.py [--reset]
"""

from __future__ import annotations

import argparse
import csv

from langchain_core.documents import Document

from config import FAQ_COLLECTION, FAQ_CSV
from vectorstore import count, reset_collection, upsert


def load_faq_documents() -> tuple[list[Document], list[str]]:
    if not FAQ_CSV.exists():
        raise FileNotFoundError(f"FAQ file not found: {FAQ_CSV}")

    documents: list[Document] = []
    ids: list[str] = []

    with FAQ_CSV.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not question or not answer:
                continue

            faq_id = (row.get("id") or str(len(ids) + 1)).strip()
            category = (row.get("category") or "general").strip()

            # Embedding both the question and the answer keeps the entry
            # retrievable whether the customer phrases it like the question or
            # describes the symptom in the answer.
            documents.append(
                Document(
                    page_content=f"Q: {question}\nA: {answer}",
                    metadata={
                        "source": "faq",
                        "faq_id": faq_id,
                        "category": category,
                        "question": question,
                        "citation": f"FAQ #{faq_id} — {question}",
                    },
                )
            )
            ids.append(f"faq-{faq_id}")

    return documents, ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest FAQ entries into ChromaDB.")
    parser.add_argument("--reset", action="store_true", help="drop the collection first")
    args = parser.parse_args()

    if args.reset:
        reset_collection(FAQ_COLLECTION)

    documents, ids = load_faq_documents()
    upsert(FAQ_COLLECTION, documents, ids)
    print(f"faq: ingested {len(documents)} entries -> {count(FAQ_COLLECTION)} vectors in store")


if __name__ == "__main__":
    main()
