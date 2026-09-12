"""Streamlit chat UI for the telecom care assistant.

Covers FR-01 to FR-05a and FR-13a of the RAG PRD: free-text questions,
one-click sample questions, session history, clear conversation, streamed
answers, a 👍/👎 control on every answer, and an expandable Sources panel.

It is also the front end for the refund agent (agent PRD §11): the same chat,
plus a Confirm / Cancel step for an eligible refund within the approval limit.
Clearing the conversation clears the pending refund with it (FR-48).

Run:  streamlit run app.py
"""

from __future__ import annotations

import time
import uuid

import streamlit as st

from config import APP_TAGLINE, APP_TITLE, LLM_MODEL, SAMPLE_QUESTIONS, top_k_for
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
    st.session_state.setdefault("pending", None)  # a queued sample question
    st.session_state.setdefault("refund", None)  # in-flight refund, if any


def _clear_conversation() -> None:
    """FR-48 / §11: history and any half-finished refund both go."""
    st.session_state.messages = []
    st.session_state.pending = None
    st.session_state.refund = None


def _reset_demo_account() -> None:
    """Put the simulated recharges back so the demo can be run again.

    Once ₹499 has been refunded it stays refunded for the life of the server
    process, which is the honest behaviour — but it also means scenario 1 only
    works once without this.
    """
    from accounts import reset_accounts

    reset_accounts()
    _clear_conversation()


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
        st.button(
            "Reset demo account",
            key="reset-account",
            use_container_width=True,
            help="Restore the simulated recharges and start over.",
            on_click=_reset_demo_account,
        )

        st.divider()
        st.subheader("Knowledge base")
        total = sum(counts.values())
        if total == 0:
            st.error("No documents indexed. Run `python ingest_all.py` first.")
        else:
            for name, n in counts.items():
                st.caption(f"**{name}** — {n} documents (top {top_k_for(name)} retrieved)")
            per_answer = sum(top_k_for(name) for name in counts)
            st.caption(f"Up to {per_answer} documents retrieved per answer")

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


def _md(text: str) -> str:
    """Escape dollar signs before rendering markdown.

    Streamlit reads `$...$` as LaTeX, so an answer containing two prices —
    "$10 per day ... would cost $70" — renders as "10 per day ... would cost"
    with the prices swallowed into an empty math span. Escaping every `$`
    leaves the answer as written. This only started to matter once the plan
    catalog put prices in the answers.
    """
    return text.replace("$", r"\$")


def _stream_for_display(stream, sink: list):
    """Yield display-escaped chunks while collecting the raw text in `sink`.

    The raw text is what gets stored in history and written to the log; only
    what Streamlit renders is escaped. `$` is a single character, so escaping
    chunk by chunk cannot break across a chunk boundary.
    """
    for chunk in stream:
        sink.append(chunk)
        yield _md(chunk)



def _render_sources(message: dict) -> None:
    citations = message.get("citations") or []
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for label, citation, snippet in citations:
            st.markdown(f"**{label}** · {_md(citation)}")
            st.caption(_md(snippet))


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
            st.markdown(_md(message["content"]))
            if message["role"] == "assistant":
                _render_sources(message)
                _render_feedback(message, index)


# --- Answering -----------------------------------------------------------


def _fail(text: str) -> None:
    with st.chat_message("assistant"):
        st.error(text)
    st.session_state.messages.append(
        {"role": "assistant", "content": f"⚠️ {text}", "citations": []}
    )


def _record(question: str, text: str, citations: list, elapsed: float) -> dict:
    """Append an assistant message to history and log it."""
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
    return message


def _answer_refund(question: str, history: list[dict]) -> None:
    """Run one turn of the refund agent and show what it decided."""
    import refund_agent

    with st.chat_message("assistant"):
        started = time.perf_counter()
        try:
            with st.spinner("Checking your recharges…"):
                turn = refund_agent.run(
                    question, pending=st.session_state.refund, history=history
                )
        except Exception as exc:
            st.error(f"Could not complete that refund request: {exc}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ {exc}", "citations": []}
            )
            return

        elapsed = time.perf_counter() - started
        st.markdown(_md(turn.reply))

        # Whatever the agent decided is now the session's refund state; a
        # finished refund leaves None behind and the next message is routed
        # from scratch.
        st.session_state.refund = turn.pending

        _record(question, turn.reply, [], elapsed)
        st.caption(f"Refund agent · {turn.stage} · {elapsed:.1f}s")
        _render_feedback(
            st.session_state.messages[-1], len(st.session_state.messages) - 1
        )


def _answer_rag(question: str, history: list[dict]) -> None:
    """The existing grounded-answer path, unchanged apart from the history it
    now carries into the prompt (FR-02, FR-46)."""
    from chain import MissingAPIKey, answer_stream

    with st.chat_message("assistant"):
        started = time.perf_counter()
        try:
            with st.spinner("Searching our knowledge base…"):
                docs, stream = answer_stream(question, history)
            chunks: list[str] = []
            st.write_stream(_stream_for_display(stream, chunks))
            text = "".join(chunks)
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
        message = _record(question, text, citations, elapsed)

        _render_sources(message)
        st.caption(f"Answered in {elapsed:.1f}s")
        _render_feedback(message, len(st.session_state.messages) - 1)


def _handle(question: str) -> None:
    """Classify one message, then send it down its route (FR-01 – FR-03)."""
    from chain import MissingAPIKey
    from router import REFUND, route

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(_md(question))

    # The history the routes see excludes the message being answered — it is
    # the prompt, not context for itself.
    history = st.session_state.messages[:-1]

    try:
        with st.spinner("Reading your message…"):
            destination = route(question, st.session_state.refund, history)
    except MissingAPIKey as exc:
        _fail(str(exc))
        return
    except Exception as exc:
        _fail(f"Could not reach the language model: {exc}")
        return

    if destination == REFUND:
        _answer_refund(question, history)
    else:
        _answer_rag(question, history)


def _render_confirmation() -> None:
    """The Confirm / Cancel step for an eligible refund within the limit (§11).

    The buttons do not execute anything themselves — they put the confirmation
    into the conversation as the customer's own words, which then travels the
    same path a typed "yes, proceed" does. One confirmation path, so there is
    one place where consent is interpreted (GR-02).
    """
    from refund_agent import AWAITING_CONFIRMATION

    pending = st.session_state.refund
    if not pending or pending.get("stage") != AWAITING_CONFIRMATION:
        return

    with st.container(horizontal=True):
        confirm = st.button("Confirm refund", key="confirm-refund", type="primary")
        cancel = st.button("Cancel", key="cancel-refund")

    if confirm or cancel:
        st.session_state.pending = (
            "Yes, proceed." if confirm else "No, cancel the refund."
        )
        st.rerun()


# --- Main ----------------------------------------------------------------


def main() -> None:
    _init_state()

    st.title(f"📡 {APP_TITLE}")
    st.caption(APP_TAGLINE)

    _warm_up()
    _render_sidebar(_collection_counts())

    if not st.session_state.messages:
        st.info(
            "Ask about our plans and pricing, roaming passes, slow data, a confusing "
            "charge, SIM or eSIM activation, call quality, or the app — I answer from "
            "our published plan catalog and support knowledge. I can also cancel or "
            "refund a recent recharge for you."
        )

    _render_history()

    typed = st.chat_input("Ask a question, or request a refund…")
    question = typed or st.session_state.pending
    st.session_state.pending = None

    if question:
        _handle(question)

    # Last, so the buttons land under the message that asked for them on the
    # same run the agent produced it, and under the history on every run after.
    _render_confirmation()


main()
