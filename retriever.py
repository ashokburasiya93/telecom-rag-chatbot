"""Merged retriever across the three knowledge collections (FR-06 – FR-08).

Each query fans out to `faq`, `tickets` and `guides` in parallel and returns the
top-3 documents from each — nine source-labelled context documents per question.

Adding a fourth knowledge source (NFR-06) means writing a new `ingest_*.py` and
adding its collection name to `COLLECTIONS` below.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from langchain_core.documents import Document

from config import (
    FAQ_COLLECTION,
    GUIDES_COLLECTION,
    PLANS_COLLECTION,
    SOURCE_LABELS,
    TICKETS_COLLECTION,
    top_k_for,
)
from vectorstore import get_collection

COLLECTIONS = (FAQ_COLLECTION, TICKETS_COLLECTION, GUIDES_COLLECTION, PLANS_COLLECTION)


@dataclass(frozen=True)
class RetrievedDoc:
    """A retrieved document plus the label the prompt and UI cite it by."""

    collection: str
    document: Document

    @property
    def label(self) -> str:
        return SOURCE_LABELS.get(self.collection, self.collection.upper())

    @property
    def citation(self) -> str:
        meta = self.document.metadata
        if meta.get("citation"):
            return str(meta["citation"])
        return f"{self.label} document"

    @property
    def text(self) -> str:
        return self.document.page_content


def _search(collection: str, question: str, k: int) -> list[RetrievedDoc]:
    try:
        hits = get_collection(collection).similarity_search(question, k=k)
    except Exception as exc:  # a missing/empty collection must not kill the query
        print(f"[retriever] {collection} search failed: {exc}")
        return []
    return [RetrievedDoc(collection=collection, document=doc) for doc in hits]


def retrieve(question: str, k: int | None = None) -> list[RetrievedDoc]:
    """Fetch the top documents from every collection, in parallel.

    `k` overrides the per-collection default for every collection; left alone,
    each collection uses `top_k_for()`.
    """
    with ThreadPoolExecutor(max_workers=len(COLLECTIONS)) as pool:
        futures = [
            pool.submit(_search, name, question, k if k is not None else top_k_for(name))
            for name in COLLECTIONS
        ]
        results = [future.result() for future in futures]

    # Keep collection order (FAQ first) so the prompt reads consistently and the
    # most authoritative source leads.
    return [doc for group in results for doc in group]


def format_context(docs: list[RetrievedDoc]) -> str:
    """Render retrieved documents as the source-labelled context block."""
    if not docs:
        return "NO CONTEXT AVAILABLE"

    blocks = []
    for i, doc in enumerate(docs, start=1):
        blocks.append(f"[{i}] [{doc.label}] ({doc.citation})\n{doc.text}")
    return "\n\n".join(blocks)


def warm_up() -> None:
    """Load the embedding model and open the collections ahead of first query.

    Loading the model here, on one thread, matters: `retrieve()` fans out to
    three threads, and `lru_cache` does not hold a lock across the wrapped
    call. Left to load lazily, all three threads would miss the cache and build
    their own copy of the model — three times the load time and the memory.
    """
    from embeddings import get_embeddings

    get_embeddings()
    for name in COLLECTIONS:
        get_collection(name)
