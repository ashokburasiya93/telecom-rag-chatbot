"""Central configuration for the RAG telecom care chatbot.

Everything tunable lives here so the ingest scripts, the retriever, the CLI and
the Streamlit app all agree on paths, collection names and model settings.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")


def _resolve_data_dir() -> Path:
    """Return the data directory, tolerating either `data/` or `Data/`."""
    for name in ("data", "Data"):
        candidate = PROJECT_ROOT / name
        if candidate.is_dir():
            return candidate
    return PROJECT_ROOT / "data"


# --- Paths ---------------------------------------------------------------

DATA_DIR = _resolve_data_dir()
FAQ_CSV = DATA_DIR / "faq.csv"
TICKETS_DB = DATA_DIR / "tickets.db"
GUIDE_PDF = DATA_DIR / "telecom_guide.pdf"
PLANS_JSON = DATA_DIR / "plans.json"

CHROMA_DIR = PROJECT_ROOT / "chroma_store"
# Used only when ChromaDB cannot be imported on this machine (see vectorstore.py).
LOCAL_STORE_DIR = PROJECT_ROOT / "vector_store"
LOG_DIR = PROJECT_ROOT / "logs"
INTERACTION_LOG = LOG_DIR / "interactions.jsonl"

# --- Collections (PRD §8) ------------------------------------------------

FAQ_COLLECTION = "faq"
TICKETS_COLLECTION = "tickets"
GUIDES_COLLECTION = "guides"
PLANS_COLLECTION = "plans"

# Human-readable source labels injected into the prompt context (FR-08).
SOURCE_LABELS = {
    FAQ_COLLECTION: "FAQ",
    TICKETS_COLLECTION: "TICKETS",
    GUIDES_COLLECTION: "GUIDES",
    PLANS_COLLECTION: "PLANS",
}

# --- Retrieval (FR-07, FR-09) -------------------------------------------

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "3"))  # per collection

# Plans is the one collection where a correct answer usually means comparing
# several documents against each other — "the cheapest unlimited plan" is only
# cheapest relative to the rest of the lineup. Three plan documents out of
# seventeen is not enough to make that comparison safely, so it retrieves more.
COLLECTION_TOP_K = {PLANS_COLLECTION: 6}


def top_k_for(collection: str) -> int:
    """Documents to retrieve from one collection (TOP_K unless overridden)."""
    return COLLECTION_TOP_K.get(collection, TOP_K)

# --- Guide chunking (FR-16) ---------------------------------------------

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# --- Generation (FR-12, FR-13) ------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.6-27b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "none")

# --- UI ------------------------------------------------------------------

APP_TITLE = "Telecom Care Assistant"
APP_TAGLINE = (
    "Grounded answers from our plan catalog, FAQ, resolved tickets and technical guides."
)

SAMPLE_QUESTIONS = [
    "What's your cheapest unlimited plan?",
    "Do you have a family plan?",
    "I'm going to Europe for about a week. What are my roaming options?",
    "Is there a discount for students?",
    "What's the difference between the International Calling Pack and a roaming pass?",
    "Why is my mobile internet so slow?",
    "Why is my bill higher than usual?",
    "My eSIM activation keeps failing — what should I do?",
    "My calls keep dropping. What should I do?",
    "How do I set up autopay?",
]

ESCALATION_HINT = "call 611 or use the MyTelecom app"
