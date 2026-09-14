# Harry Potter RAG

A from-scratch Retrieval-Augmented Generation system over the Harry Potter books:
ChromaDB retrieval, cross-encoder reranking, Gemini generation. No RAG framework —
every stage is written and owned directly.

---

## Stack

| Layer | Choice |
|---|---|
| Vector store | ChromaDB (persistent, local `chroma/`) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Generation | Gemini `gemini-2.5-flash` (`google-genai`) |
| Chunking | `RecursiveCharacterTextSplitter`, 500 chars / 100 overlap |
| Config | pydantic-settings |
| Tooling | uv, ruff, pytest |

## Pipeline

```
text → chapter split (regex CHAPTER \w+) → 500-char chunks (100 overlap)
     → Chroma collection (1224 chunks from book 1)

query → embed → Chroma top-10 → cross-encoder rerank → top-5
      → grounded prompt → Gemini → answer
```

The prompt constrains the model to the retrieved context and requires it to reply
`I don't know based on the provided context.` rather than answer from parametric
knowledge — the retrieval is the point, so an ungrounded answer is a failure even
when it happens to be right.

## Layout

```
├── evals/questions.json     # golden eval set
├── resources/               # source texts
├── src/hpotter/
│   ├── config.py            # Settings — all tunables live here
│   ├── ingest.py            # chapter split → chunk → embed → Chroma
│   ├── query.py             # retrieve → rerank → prompt → Gemini
│   └── cli.py               # entry point
└── tests/
```

---

## Setup

```bash
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/hpotter
uv sync
```

`.env` at the repo root:

```
GEMINI_API_KEY=...
```

## Run

```bash
uv run python -m hpotter.ingest    # build the index (once)
uv run python -m hpotter.query     # interactive Q&A; 'x' to quit
uv run ruff check src/
```

---

## Evaluation

Retrieval and answering are measured **separately**. "The answer was wrong" does not
say where it broke:

- **Retrieval** — did a chunk containing the answer reach the top-k? If not, no
  prompt change can help. Fix chunking, `n_retrieve`, or the embedding model.
- **Answering** — given that the right chunk *was* present, did the model use it?
  If retrieval hit and the answer is still wrong, that is a prompt problem.

Retrieval is scored first: no API calls, deterministic, and it isolates the layer
most likely to be at fault.

### Case categories

`evals/questions.json` tags every case, because they cannot share a scoring rule:

| Category | Passes when |
|---|---|
| `extractable` | answer sits in one passage; keywords appear in retrieved chunks |
| `multi_hop` | answer requires chaining two passages; expected to fail single-shot |
| `unanswerable` | not in the corpus; correct behaviour is the refusal string |

### Scoring rules

- **All keywords must match** (not any), evaluated **across the whole retrieved set**
  rather than within a single chunk — chunk boundaries at 500 chars are arbitrary, so
  requiring co-location would measure the chunker's luck, not retrieval quality.
- **Matching is case-insensitive.** Verified necessary: `Invisibility Cloak` returns
  0 case-sensitive hits against the corpus and 13 case-insensitive.
- Every keyword is verified to exist in the corpus before entering the set. A wrong
  eval is worse than no eval — it sends you chasing a bug that does not exist.

### A note on the multi-hop case

Book one never states Harry's birth date. It is reachable in exactly two hops —
*"that Gringotts break-in happened on my birthday!"* and *"the break-in at Gringotts
on 31 July"* — and no single chunk contains the answer. Single-shot RAG cannot bridge
them by construction. The case is kept as a known failure and as the motivating
example for the planned agentic phase.

---
