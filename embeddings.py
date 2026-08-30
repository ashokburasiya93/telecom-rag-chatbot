"""Local embedding model (FR-09, NFR-02).

`all-MiniLM-L6-v2` runs on the machine via sentence-transformers, so no
embedding calls leave the laptop and there is no per-token embedding cost.
The model is loaded once and reused by every ingest script and the retriever.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the shared local embedding model (downloaded on first use)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
