"""Interactive CLI for the telecom care assistant (FR-18, FR-19).

Run:  python main.py
Type `quit` (or `exit`) to leave, `sources` to show what backed the last answer.
"""

from __future__ import annotations

import sys
import time
import uuid

from config import APP_TITLE, LLM_MODEL, SAMPLE_QUESTIONS
from interaction_log import log_answer

EXIT_WORDS = {"quit", "exit", ":q"}


def _make_output_crash_proof() -> None:
    """Never let an unencodable character kill a streamed answer.

    On Windows stdout defaults to the ANSI code page (cp1252 here). Answers are
    printed token by token, so a single character outside that code page — an
    emoji, a currency symbol like ₹ — raises UnicodeEncodeError mid-response and
    takes the CLI down with it. Keeping the console's own encoding preserves
    correct rendering; `errors="replace"` turns the rare unmappable character
    into "?" instead of a crash.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # not a reconfigurable stream
            pass


def _print_banner() -> None:
    from vectorstore import backend_note

    print(f"\n{APP_TITLE} — CLI")
    print(f"model: {LLM_MODEL}")
    print(f"store: {backend_note()}")
    print("type 'quit' to exit, 'sources' to see what backed the last answer\n")
    print("try one of these:")
    for question in SAMPLE_QUESTIONS[:4]:
        print(f"  · {question}")
    print()


def main() -> int:
    _make_output_crash_proof()

    from chain import MissingAPIKey, answer_stream
    from retriever import warm_up

    _print_banner()

    print("loading local embedding model…", flush=True)
    warm_up()

    last_docs: list = []

    while True:
        try:
            # Strip a BOM as well as whitespace: piping input on Windows can
            # prefix the first line with one, which would otherwise turn `quit`
            # into a question.
            question = input("you > ").lstrip("﻿").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0

        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            print("bye")
            return 0

        if question.lower() == "sources":
            if not last_docs:
                print("no answer yet\n")
                continue
            print()
            for i, doc in enumerate(last_docs, start=1):
                print(f"  [{i}] {doc.label:<8} {doc.citation}")
            print()
            continue

        started = time.perf_counter()
        try:
            docs, stream = answer_stream(question)
        except MissingAPIKey as exc:
            print(f"\n! {exc}\n")
            return 1
        except Exception as exc:
            print(f"\n! could not reach the language model: {exc}\n")
            continue

        last_docs = docs
        print("\nbot > ", end="", flush=True)
        chunks: list[str] = []
        try:
            for chunk in stream:
                chunks.append(chunk)
                print(chunk, end="", flush=True)
        except Exception as exc:
            print(f"\n! stream failed: {exc}")
            continue

        elapsed = time.perf_counter() - started
        text = "".join(chunks)
        print(f"\n\n  [{len(docs)} sources · {elapsed:.1f}s · type 'sources' to list them]\n")

        log_answer(
            interaction_id=uuid.uuid4().hex[:12],
            question=question,
            answer=text,
            citations=[doc.citation for doc in docs],
            latency_s=elapsed,
            channel="cli",
        )


if __name__ == "__main__":
    sys.exit(main())
