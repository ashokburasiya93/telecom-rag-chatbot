"""Streamlit chat UI for the telecom care assistant.

Covers FR-01 to FR-05a and FR-13a: free-text questions, one-click sample
questions, session history, clear conversation, streamed answers, a 👍/👎
control on every answer, and an expandable Sources panel.

Run:  streamlit run app.py
"""

from __future__ import annotations

import time
import uuid

import streamlit as st

from config import APP_TAGLINE, APP_TITLE, LLM_MODEL, SAMPLE_QUESTIONS, TOP_K
from interaction_log import log_answer, log_feedback

st.set_page_config(page_title=APP_TITLE, page_icon="📡", layout="centered")


# --- Lazy, cached setup --------------------------------------------------


@st.cache_resource(show_spinner="Loading the local embedding model…")
def _warm_up() -> bool:
    """Load embeddings + open collections once per server process (NFR-05)."""
    from retriever import warm_up

    warm_up()
    return True


def _collection_counts() -> dict[str, int]:
    """Read counts on every run, not once — a re-ingest while the app is
    serving should be visible in the sidebar, not stale until restart."""
    from retriever import COLLECTIONS
    from vectorstore import count

    return {name: count(name) for name in COLLECTIONS}


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending", None)


def _clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.pending = None


# --- Sidebar -------------------------------------------------------------


def _render_sidebar(counts: dict[str, int]) -> None:
    with st.sidebar:
        st.subheader("Try a question")
        st.caption("Click to ask it straight away.")
        for i, question in enumerate(SAMPLE_QUESTIONS):
            if st.button(question, key=f"sample-{i}", use_container_width=True):
                st.session_state.pending = question
                st.rerun()

        st.divider()
        st.button(
            "Clear conversation",
            key="clear",
            use_container_width=True,
            on_click=_clear_conversation,
        )

        st.divider()
        st.subheader("Knowledge base")
        total = sum(counts.values())
        if total == 0:
            st.error("No documents indexed. Run `python ingest_all.py` first.")
        else:
            for name, n in counts.items():
                st.caption(f"**{name}** — {n} documents")
            st.caption(f"Top {TOP_K} retrieved per source · {TOP_K * len(counts)} per answer")

        st.divider()
        from vectorstore import backend, backend_note

        st.caption(f"Model: `{LLM_MODEL}`")
        st.caption(f"Store: {backend_note()}")
        if backend() != "chromadb":
            st.caption(
                "ChromaDB could not load on this machine — see README, "
                "*Vector store backend*."
            )


# --- Message rendering ---------------------------------------------------


def _render_sources(message: dict) -> None:
    citations = message.get("citations") or []
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for label, citation, snippet in citations:
            st.markdown(f"**{label}** · {citation}")
            st.caption(snippet)


def _render_feedback(message: dict, index: int) -> None:
    # Error messages carry no interaction_id, so there is nothing to rate and
    # nothing to correlate feedback against.
    if not message.get("interaction_id"):
        return

    rating = message.get("rating")
    if rating:
        st.caption("Thanks — 👍 recorded." if rating == "up" else "Thanks — 👎 recorded.")
        return

    # A horizontal container keeps the two controls side by side at any width;
    # st.columns collapses them into a vertical stack in a narrow viewport.
    with st.container(horizontal=True):
        clicked = None
        if st.button("👍", key=f"up-{index}", help="This answer helped"):
            clicked = "up"
        if st.button("👎", key=f"down-{index}", help="This answer did not help"):
            clicked = "down"

    if clicked:
        message["rating"] = clicked
        log_feedback(
            interaction_id=message["interaction_id"],
            question=message["question"],
            rating=clicked,
        )
        st.rerun()


def _render_history() -> None:
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_sources(message)
                _render_feedback(message, index)


# --- Answering -----------------------------------------------------------


def _answer(question: str) -> None:
    from chain import MissingAPIKey, answer_stream

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        started = time.perf_counter()
        try:
            with st.spinner("Searching our knowledge base…"):
                docs, stream = answer_stream(question)
            text = st.write_stream(stream)
        except MissingAPIKey as exc:
            st.error(str(exc))
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ {exc}", "citations": []}
            )
            return
        except Exception as exc:  # network, rate limit, bad model name
            st.error(f"Could not reach the language model: {exc}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ {exc}", "citations": []}
            )
            return

        elapsed = time.perf_counter() - started
        citations = [
            (doc.label, doc.citation, doc.text[:280].replace("\n", " ")) for doc in docs
        ]
        interaction_id = uuid.uuid4().hex[:12]

        message = {
            "role": "assistant",
            "content": text,
            "citations": citations,
            "question": question,
            "interaction_id": interaction_id,
        }
        st.session_state.messages.append(message)

        log_answer(
            interaction_id=interaction_id,
            question=question,
            answer=text,
            citations=[c for _, c, _ in citations],
            latency_s=elapsed,
        )

        _render_sources(message)
        st.caption(f"Answered in {elapsed:.1f}s")
        _render_feedback(message, len(st.session_state.messages) - 1)


# --- Main ----------------------------------------------------------------


def main() -> None:
    _init_state()

    st.title(f"📡 {APP_TITLE}")
    st.caption(APP_TAGLINE)

    _warm_up()
    _render_sidebar(_collection_counts())

    if not st.session_state.messages:
        st.info(
            "Ask about slow data, a confusing charge, SIM or eSIM activation, roaming, "
            "call quality, or the app. I answer from our published support knowledge — "
            "I can't see your account details."
        )

    _render_history()

    typed = st.chat_input("Ask a question about your service…")
    question = typed or st.session_state.pending
    st.session_state.pending = None

    if question:
        _answer(question)


main()
