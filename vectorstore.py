"""Vector store access (NFR-05) — ChromaDB, with a numpy fallback.

The PRD specifies ChromaDB persisted to `chroma_store/`, and that is what this
module uses whenever `chromadb` can be imported.

`requirements.txt` pins chromadb below 0.6 deliberately. From 1.0 onwards the
package imports the OpenTelemetry gRPC exporter at import time, which pulls in
grpcio's unsigned native extension; Windows Smart App Control refuses to load
it, and `import chromadb` fails before any of its own code runs. 0.5.x has no
such dependency. See the README for the full account.

If the import fails anyway, this module falls back to `local_store`, which
implements the same handful of calls over numpy, so the app still runs.
`backend()` reports which one is live, so the UI and CLI can say so plainly
rather than quietly behaving differently from the PRD.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Sequence

# Belt and braces alongside the Settings flag below. On chromadb 0.5.3 neither
# actually suppresses the ClientStartEvent — see the logger note below — but
# both are the documented switches and cost nothing.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_core.documents import Document

from config import CHROMA_DIR
from embeddings import get_embeddings

CHROMA = "chromadb"
LOCAL = "local-numpy"

try:  # pragma: no cover - depends on the machine, not the code
    from chromadb.config import Settings
    from langchain_chroma import Chroma

    # `is_persistent` and `persist_directory` must both be set here. When
    # langchain-chroma is handed a client_settings object it only copies
    # persist_directory onto it — it does NOT set is_persistent, which it would
    # have done had we passed persist_directory alone. Omit it and Chroma runs
    # in memory: ingestion reports success, writes nothing, and the app starts
    # with an empty index (NFR-05).
    #
    # anonymized_telemetry: no usage pings from a customer-support surface. The
    # 0.5.x telemetry client is also incompatible with current posthog releases,
    # so leaving it on prints a spurious failure on every call.
    _SETTINGS = Settings(
        anonymized_telemetry=False,
        is_persistent=True,
        persist_directory=str(CHROMA_DIR),
    )

    # chromadb 0.5.x emits ClientStartEvent even with telemetry disabled, and
    # its posthog call signature is stale, so every client start logs a failure.
    # Nothing is transmitted — the call raises before sending — but the noise
    # buries real output from the ingest scripts. Silence that logger only.
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
    _BACKEND = CHROMA
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # ImportError, or a blocked DLL load
    Chroma = None  # type: ignore[assignment]
    _SETTINGS = None
    _BACKEND = LOCAL
    _IMPORT_ERROR = exc


def backend() -> str:
    """Return the active backend name: "chromadb" or "local-numpy"."""
    return _BACKEND


def backend_note() -> str:
    """One line explaining the backend choice, for the UI and README."""
    if _BACKEND == CHROMA:
        return f"ChromaDB, persisted to {CHROMA_DIR.name}/"
    return f"Local numpy store — ChromaDB unavailable ({type(_IMPORT_ERROR).__name__})"


@lru_cache(maxsize=None)
def get_collection(name: str):
    """Open (or create) a persisted collection by name."""
    if _BACKEND == CHROMA:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=name,
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
            client_settings=_SETTINGS,
        )

    from local_store import LocalVectorCollection

    return LocalVectorCollection(name)


def upsert(name: str, documents: Sequence[Document], ids: Sequence[str]):
    """Write documents under stable ids.

    Stable ids are what make re-running an ingest script idempotent (FR-17):
    a second run overwrites the same rows rather than duplicating them, and an
    edited FAQ answer replaces the old one in place.
    """
    if len(documents) != len(ids):
        raise ValueError("documents and ids must be the same length")

    store = get_collection(name)
    store.add_documents(documents=list(documents), ids=list(ids))
    return store


def reset_collection(name: str) -> None:
    """Drop a collection entirely (used by `--reset`)."""
    store = get_collection(name)
    try:
        store.delete_collection()
    except Exception:  # collection may not exist yet
        pass
    get_collection.cache_clear()


def count(name: str) -> int:
    """Number of vectors currently stored in a collection."""
    store = get_collection(name)
    try:
        if _BACKEND == CHROMA:
            return store._collection.count()
        return store.count()
    except Exception:
        return 0
