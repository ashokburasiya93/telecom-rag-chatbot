# RAG Telecom Customer Care Chatbot

A retrieval-augmented chatbot that answers Tier-1 telecom support questions — slow data,
confusing charges, SIM and eSIM problems, roaming, call quality, account basics — plus
plans, pricing and add-ons — grounded in four knowledge sources and nothing else.

Built to [`PRD.md`](PRD.md).

```
question ──▶ merged retriever ──▶ 15 source-labelled documents ──▶ prompt ──▶ Groq ──▶ streamed answer
              ├── faq      top-3
              ├── tickets  top-3
              ├── guides   top-3
              └── plans    top-6
```

---

## Quick start

```bash
py -3.11 -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python setup_windows_runtime.py
copy .env.example .env
python ingest_all.py
streamlit run app.py
```

**Use Python 3.11.** `chromadb` 0.5.x needs `chroma-hnswlib`, which has no prebuilt wheel
for 3.13 and won't build without a C compiler. See *Vector store backend* for why the
version is pinned there.

Put your Groq key in `.env` (free tier at <https://console.groq.com/keys>). The CLI is
`python main.py`; type `quit` to exit, `sources` to list what backed the last answer.

The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) and takes about
20 seconds. After that, retrieval is ~50 ms and a full answer lands in well under a second.

---

## Layout

| File | Role |
|---|---|
| `config.py` | Every tunable: paths, collections, model names, top-k, chunking, sample questions |
| `embeddings.py` | Local `all-MiniLM-L6-v2` embeddings — nothing leaves the machine (NFR-02) |
| `vectorstore.py` | Store access; ChromaDB, with a numpy fallback (see below) |
| `local_store.py` | The numpy fallback store |
| `ingest_faq.py` | `data/faq.csv` → one document per row |
| `ingest_tickets.py` | `data/tickets.db` → one document per **resolved** ticket |
| `ingest_guides.py` | `data/telecom_guide.pdf` → 600-char chunks, 100-char overlap |
| `ingest_plans.py` | `data/plans.json` → one document per plan or add-on, plus 3 cross-plan summaries |
| `ingest_all.py` | Runs all four |
| `retriever.py` | Fans out to all four collections in parallel |
| `chain.py` | Prompt + Groq LLM + LCEL chain |
| `app.py` | Streamlit chat UI |
| `main.py` | CLI REPL |
| `interaction_log.py` | Appends answers and 👍/👎 to `logs/interactions.jsonl` |
| `setup_windows_runtime.py` | Windows-only torch runtime repair |

## Updating the knowledge base

Support ops can change what the bot knows without a redeploy (US-06, US-07):

```bash
# edit data/faq.csv, then
python ingest_faq.py

# add resolved rows to data/tickets.db, then
python ingest_tickets.py

# change a price or launch a plan in data/plans.json, then
python ingest_plans.py
```

Re-runs are idempotent. Every document is stored under a stable id — `faq-7`,
`ticket-TK-012`, `guide-p5-c2`, `plan-PRE-UNLTD` — so a second run overwrites the same
rows instead of duplicating them, and an edited answer or a repriced plan replaces the
old one in place. Pass `--reset` to drop a collection and rebuild it from scratch — do
that when a plan is *withdrawn*, since an id that is no longer in the JSON is not
overwritten by a plain re-run.

## Adding a knowledge source

Write an `ingest_yours.py` that returns `(documents, ids)`, then add the collection name
to `COLLECTIONS` in `retriever.py` and a label in `SOURCE_LABELS` in `config.py`. Nothing
else changes (NFR-06). `ingest_plans.py` is the worked example — it was added this way.

### The plans source

`data/plans.json` is the operator's official price list, and three things about it are
worth knowing:

**It is rendered as prose, not JSON.** `all-MiniLM-L6-v2` was trained on sentences.
`"data_unlimited": true` embeds nowhere near "do you have an unlimited plan"; the line
"Unlimited 5G data" does.

**Three cross-plan summaries are indexed alongside the per-plan documents.** "What is
your *cheapest* unlimited plan?" is not answerable from any single plan document —
cheapest is a fact about the lineup. The summaries (all plans by price, unlimited plans
compared, add-ons and roaming passes) are generated from the same JSON at ingest time, so
they cannot drift from it. They are also what makes the bot volunteer the caveat that
Student Unlimited is cheaper but restricted.

**`plans` retrieves 6 documents, not 3** (`COLLECTION_TOP_K` in `config.py`). It is the
one collection where a correct answer usually means weighing several documents against
each other; three out of seventeen is not enough to compare on.

---

## Grounding

The system prompt in `chain.py` is the guardrail. It forbids answering from the model's own
knowledge, forbids stating any number that is not in the retrieved context, and requires the
bot to say plainly when it cannot answer and point to 611 or the MyTelecom app. Temperature
is 0.

Two rules exist specifically because the catalog was added:

**`[PLANS]` outranks the other sources on price.** FAQ #10 still describes a legacy "EU
Roaming Bundle at $15/day" that is not in the current catalog. Retrieval surfaces it and
the catalog side by side, and without a precedence rule the bot quoted the withdrawn
bundle. The prompt now treats `[PLANS]` as the dated price list and tells the model to
drop any plan, pass or price the other sources name that the catalog does not.

**Arithmetic on retrieved numbers is allowed, inventing numbers is not.** "Europe for a
week" needs $10/day × 7 = $70 to be compared against the $50 monthly pass. The rule
permits multiplying numbers that are in the context and still forbids producing any
number that is not.

A scope rule was added at the same time: plans, pricing, billing and technical support are
in scope, and anything else — a travel itinerary, general recommendations — gets one
sentence declining, plus the one relevant NovaCell action if there is one. Adding public
pricing widened what the bot knows about, not what it is willing to be asked.

Observed behaviour on the current corpus:

| Question | Result |
|---|---|
| "Why is my mobile internet so slow?" | Answers from FAQ #2 + guide, including the 512 kbps throttle figure — which is in the FAQ |
| "Travelling to Japan, what about roaming?" | Answers from FAQ #9 + guide §4 (Zone B). Does **not** quote the EU bundle price, which is in the corpus but not applicable |
| "What is my current account balance?" | Declines — says it cannot see account details, points to `*123#` and the app |
| "Who won the 2022 World Cup?" | Declines as out of scope |

Retrieval scores separate cleanly: in-domain top hits land at 0.65–0.86 cosine, out-of-domain
questions collapse to below 0.10.

---

## Vector store backend

ChromaDB, as the PRD specifies (§9), persisted to `chroma_store/`.

Two constraints shape the pins in `requirements.txt`, and both are load-bearing:

**1. `chromadb` must stay below 0.6.** From 1.0 onwards, `chromadb/__init__.py` reaches
`chromadb.telemetry.opentelemetry`, which imports the OpenTelemetry **gRPC** exporter at
module level, which imports `grpcio`'s native `cygrpc` extension. That `.pyd` is unsigned,
and Windows **Smart App Control** — enforced on this machine — refuses to load it:

```
ImportError: DLL load failed while importing cygrpc:
An Application Control policy has blocked this file.
```

There is no Python-side workaround: the import runs before any chromadb code does.
Versions 0.5.x have no OpenTelemetry gRPC dependency and import cleanly. Verified as
blocked on 1.0.15, 1.2.1 and 1.5.9, and on the `chromadb-client` thin client; verified
working on 0.5.3.

**2. Python 3.11, not 3.13.** `chromadb` 0.5.x depends on `chroma-hnswlib`, which has no
prebuilt wheel for 3.13 and needs a C compiler to build. On 3.11 it installs from a wheel.
The PRD requires 3.11+ (NFR-04a), so this is within spec.

`onnxruntime` is also required — not because anything uses it, but because chromadb
instantiates its default ONNX embedding function as a **default argument value** in
`CollectionCommon`, which executes at class-definition time. Without it, `import chromadb`
raises. Our embeddings come from `all-MiniLM-L6-v2` via sentence-transformers regardless.

### The numpy fallback

`local_store.py` implements the same `add_documents` / `similarity_search` /
`delete_collection` surface over a numpy matrix persisted to `vector_store/`.
`vectorstore.py` selects it automatically **only** if `chromadb` cannot be imported, so the
app still runs on a machine where the pins above cannot be satisfied. The sidebar and CLI
banner always state which backend is live. Embeddings are L2-normalised, so cosine
similarity is a dot product; at this corpus size an exact scan is well under a millisecond.

## Windows: torch fails to load

If `import torch` raises `WinError 1114` on `c10.dll`, the machine's Visual C++
redistributable is too old for torch 2.x (14.13 from VS 2017 is common, and it lacks
`msvcp140_2.dll` entirely). Run:

```bash
python setup_windows_runtime.py
```

It copies the current runtime DLLs — shipped by the `msvc-runtime` package into the venv —
next to `c10.dll`, which is the first place Windows looks when resolving that DLL's
dependencies. Nothing outside the project is touched: no system install, no registry, no
admin rights. Installing the Microsoft Visual C++ 2015-2022 Redistributable (x64)
system-wide is the alternative if you prefer.

---

## Data

| Collection | Source | Documents |
|---|---|---|
| `faq` | `data/faq.csv` | 25 |
| `tickets` | `data/tickets.db` | 19 |
| `guides` | `data/telecom_guide.pdf` | 37 chunks |
| `plans` | `data/plans.json` | 14 plans and add-ons + 3 summaries |

The ticket database holds 20 rows; `TK-020` is `escalated`, not `resolved`, so it is
excluded — an unresolved fraud case has no resolution to teach the model (FR-15).

## Logs

`logs/interactions.jsonl`, one JSON object per line:

```json
{"event":"answer","interaction_id":"a1b2c3","question":"…","citations":["FAQ #2 — …"],"latency_s":0.47}
{"event":"feedback","interaction_id":"a1b2c3","rating":"down"}
```

## Configuration

All in `.env` (see `.env.example`) — `GROQ_API_KEY` is the only required value. Optional:
`LLM_MODEL`, `LLM_TEMPERATURE`, `REASONING_EFFORT`, `EMBEDDING_MODEL`, `TOP_K`.

Never commit `.env`; `.gitignore` already excludes it.
