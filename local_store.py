"""Numpy-backed vector store — the fallback when ChromaDB cannot load.

Why this exists
---------------
The PRD specifies ChromaDB (§9). On some Windows machines ChromaDB cannot be
imported at all: it pulls in `grpcio`, whose native `cygrpc` extension is
unsigned, and Smart App Control refuses to load unsigned binaries. There is no
way around that from inside Python.

This module implements the same small slice of the Chroma API that the app
actually uses — `add_documents(documents, ids)`, `similarity_search(query, k)`,
`delete_collection()` — over plain numpy. The corpus here is under a hundred
documents, so brute-force cosine similarity over a dense matrix is exact and
returns in well under a millisecond; an ANN index would be pure overhead at
this size.

Embeddings are L2-normalised by the embedding layer, so cosine similarity is
just a dot product.

`vectorstore.py` picks this backend automatically and only when ChromaDB is
unavailable. Nothing else in the app knows which backend it is talking to.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from langchain_core.documents import Document

from config import LOCAL_STORE_DIR
from embeddings import get_embeddings


class LocalVectorCollection:
    """One persisted collection: a float32 matrix plus aligned document records."""

    def __init__(self, name: str, directory: Path | None = None) -> None:
        self.name = name
        self.directory = directory or LOCAL_STORE_DIR
        self.directory.mkdir(parents=True, exist_ok=True)

        self._vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict] = []
        self._stamp: tuple[int, int] = (0, 0)
        self._load()

    # --- persistence -----------------------------------------------------

    @property
    def _vectors_path(self) -> Path:
        return self.directory / f"{self.name}.npy"

    @property
    def _records_path(self) -> Path:
        return self.directory / f"{self.name}.json"

    def _disk_stamp(self) -> tuple[int, int]:
        """Modification times of the two files backing this collection."""
        try:
            return (
                self._vectors_path.stat().st_mtime_ns,
                self._records_path.stat().st_mtime_ns,
            )
        except OSError:
            return (0, 0)

    def reload_if_stale(self) -> None:
        """Re-read the collection if another process has rewritten it.

        Support ops edit `faq.csv` and re-run `ingest_faq.py` in a separate
        process while the app is serving (US-06, US-07). Without this check the
        app would keep answering from the copy it loaded at startup and only
        pick up the new content on restart.
        """
        if self._disk_stamp() != self._stamp:
            self._load()

    def _load(self) -> None:
        if not (self._vectors_path.exists() and self._records_path.exists()):
            return
        try:
            vectors = np.load(self._vectors_path)
            records = json.loads(self._records_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[local_store] could not load '{self.name}': {exc}")
            return

        ids = records.get("ids", [])
        if len(ids) != vectors.shape[0]:
            print(f"[local_store] '{self.name}' is inconsistent; ignoring it")
            return

        self._vectors = vectors.astype(np.float32, copy=False)
        self._ids = ids
        self._texts = records.get("texts", [])
        self._metadatas = records.get("metadatas", [])
        self._stamp = self._disk_stamp()

    def _save(self) -> None:
        np.save(self._vectors_path, self._vectors)
        self._records_path.write_text(
            json.dumps(
                {"ids": self._ids, "texts": self._texts, "metadatas": self._metadatas},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self._stamp = self._disk_stamp()

    # --- Chroma-compatible surface --------------------------------------

    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        """Insert or replace documents by id (idempotent re-runs, FR-17)."""
        if not documents:
            return

        self.reload_if_stale()

        vectors = np.asarray(
            get_embeddings().embed_documents([doc.page_content for doc in documents]),
            dtype=np.float32,
        )

        position = {doc_id: i for i, doc_id in enumerate(self._ids)}
        appended: list[np.ndarray] = []

        for doc, doc_id, vector in zip(documents, ids, vectors):
            if doc_id in position:
                index = position[doc_id]
                self._vectors[index] = vector
                self._texts[index] = doc.page_content
                self._metadatas[index] = dict(doc.metadata)
            else:
                position[doc_id] = len(self._ids)
                self._ids.append(doc_id)
                self._texts.append(doc.page_content)
                self._metadatas.append(dict(doc.metadata))
                appended.append(vector)

        if appended:
            block = np.vstack(appended)
            if self._vectors.size == 0:
                self._vectors = block
            else:
                self._vectors = np.vstack([self._vectors, block])

        self._save()

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        """Return the k closest documents by cosine similarity."""
        self.reload_if_stale()
        if not self._ids:
            return []

        query_vector = np.asarray(get_embeddings().embed_query(query), dtype=np.float32)
        scores = self._vectors @ query_vector  # vectors are normalised

        k = min(k, len(self._ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        return [
            Document(
                page_content=self._texts[i],
                metadata={**self._metadatas[i], "score": round(float(scores[i]), 4)},
            )
            for i in top
        ]

    def delete_collection(self) -> None:
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        self._ids, self._texts, self._metadatas = [], [], []
        self._vectors_path.unlink(missing_ok=True)
        self._records_path.unlink(missing_ok=True)
        self._stamp = (0, 0)

    def count(self) -> int:
        self.reload_if_stale()
        return len(self._ids)
