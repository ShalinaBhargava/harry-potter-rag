# Harry Potter RAG

A from-scratch Retrieval-Augmented Generation system over the Harry Potter books:
ChromaDB retrieval, cross-encoder reranking, Gemini generation.

---

## Stack

| Layer | Choice |
|---|---|
| Vector store | ChromaDB (persistent, local `chroma/`) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Generation | Gemini `gemini-2.5-flash` (`google-genai`) |
| Chunking | `RecursiveCharacterTextSplitter`, 800 chars / 100 overlap |
| Config | pydantic-settings |
| Tooling | uv, ruff, pytest |

## Pipeline

```
text → chapter split (regex CHAPTER \w+) → 800-char chunks (100 overlap)
     → Chroma collection (692 chunks from book 1)

query → embed → Chroma top-25 → cross-encoder rerank → top-5
      → grounded prompt → Gemini → answer
```

The query side lives in one `RagPipeline` class. Models, the Gemini client and the
Chroma collection load once in its constructor, not at import time, and every tunable
comes from the `Settings` object passed in.

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
│   ├── query.py             # RagPipeline: retrieve → rerank → prompt → Gemini
│   ├── evals.py             # EvalCase model + load_cases
│   ├── run_evals.py         # retrieval eval runner
│   └── cli.py               # `hpotter` entry point
└── tests/                   # empty — tests not written yet
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
uv run hpotter ingest    # rebuild the index (~7 min; queries see a partial index meanwhile)
uv run hpotter query     # interactive Q&A; 'x' to quit
uv run hpotter eval      # retrieval eval — no API calls
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
most likely to be at fault. Answer-level scoring is not built yet.

### Case categories

`evals/questions.json` tags every case, because they cannot share a scoring rule:

| Category | Passes when |
|---|---|
| `extractable` | answer sits in one passage; keywords appear in retrieved chunks |
| `multi_hop` | answer requires chaining two passages; expected to fail single-shot |
| `unanswerable` | not in the corpus; correct behaviour is the refusal string |

The retrieval eval skips `unanswerable` cases — there is nothing to retrieve — and
leaves them out of the total.

### Scoring rules

- **All keywords must appear in a single chunk.** An earlier version accepted keywords
  spread across the whole retrieved set; that proved too lenient when a keyword is
  common (`McGonagall` appears ~100× in book one, so the set "passed" while no chunk
  answered the question). The cost of the stricter rule is sensitivity to chunk
  boundaries.
- **Two measurement points.** Each case is scored at retrieval (`25:`) and after
  selection (`5:`), so a failure can be pinned to the vector search or to the reranker.
- **Matching is case-insensitive.** Verified necessary: `Invisibility Cloak` returns
  0 case-sensitive hits against the corpus and 13 case-insensitive.
- Every keyword is verified to exist in the corpus before entering the set. A wrong
  eval is worse than no eval — it sends you chasing a bug that does not exist.

### Current results

`chunk_size=800`, `n_retrieve=25`, `n_rerank=5`, reranker on:

| metric | score |
|---|---|
| hit@25 (Chroma) | 13/16 |
| **hit@5 (final)** | **10/16** |

The six failures, by cause:

| cause | cases |
|---|---|
| rerank selection — answer in top 25, not chosen | Quidditch position, wand, Dumbledore's gift |
| co-location — keywords retrieved, never in one chunk | Mirror of Erised, vault 713 |
| multi-hop — no single chunk holds the answer | Harry's birth date |

The reranker is measured, not assumed: at 5 final chunks, the cross-encoder scores
9/16 against 7/16 for plain vector top-5 (measured at `chunk_size=500`). Experiment
history is in `WORKLOG.md`.

### A note on the multi-hop case

Book one never states Harry's birth date. It is reachable in exactly two hops —
*"that Gringotts break-in happened on my birthday!"* and *"the break-in at Gringotts
on 31 July"* — and no single chunk contains the answer. Single-shot RAG cannot bridge
them by construction. The case is kept as a known failure and as the motivating
example for the planned agentic phase.

---
