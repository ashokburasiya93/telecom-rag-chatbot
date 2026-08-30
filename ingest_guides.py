"""Ingest the PDF technical guide (FR-16).

Chunked at 600 characters with 100 characters of overlap, so a numbered
troubleshooting procedure that straddles a chunk boundary is still retrievable
from either side of the split.

Run:  python ingest_guides.py [--reset]
"""

from __future__ import annotations

import argparse

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, GUIDE_PDF, GUIDES_COLLECTION
from vectorstore import count, reset_collection, upsert


def load_guide_documents() -> tuple[list[Document], list[str]]:
    if not GUIDE_PDF.exists():
        raise FileNotFoundError(f"Guide PDF not found: {GUIDE_PDF}")

    pages = PyPDFLoader(str(GUIDE_PDF)).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    documents: list[Document] = []
    ids: list[str] = []
    per_page_counter: dict[int, int] = {}

    for chunk in chunks:
        text = chunk.page_content.strip()
        if not text:
            continue

        page = int(chunk.metadata.get("page", 0)) + 1  # PyPDF pages are 0-indexed
        index = per_page_counter.get(page, 0)
        per_page_counter[page] = index + 1

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": "guides",
                    "document": GUIDE_PDF.name,
                    "page": page,
                    "chunk": index,
                    "citation": f"{GUIDE_PDF.name} — page {page}",
                },
            )
        )
        ids.append(f"guide-p{page}-c{index}")

    return documents, ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the PDF guide into ChromaDB.")
    parser.add_argument("--reset", action="store_true", help="drop the collection first")
    args = parser.parse_args()

    if args.reset:
        reset_collection(GUIDES_COLLECTION)

    documents, ids = load_guide_documents()
    upsert(GUIDES_COLLECTION, documents, ids)
    print(
        f"guides: ingested {len(documents)} chunks "
        f"({CHUNK_SIZE} chars / {CHUNK_OVERLAP} overlap) -> "
        f"{count(GUIDES_COLLECTION)} vectors in store"
    )


if __name__ == "__main__":
    main()
