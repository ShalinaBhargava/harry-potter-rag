# Worklog

Running log of decisions, findings, and open threads. Not user-facing — this is the
file to reread when picking the project back up cold.

---

## 2026-09-10 — Phase 1: package refactor + typed config

Moved from two loose scripts under `src/` to a real package with typed settings.

- `src/hpotter/` package created; `data_insertion.py` → `ingest.py`,
  `query_data.py` → `query.py`
- `config.py` added: pydantic-settings `Settings` holding every tunable
- `.env` moved from `src/.env` to the repo root
- All hardcoded constants removed from `ingest.py` and `query.py`
- Verified end to end: 1224 chunks indexed, Chroma ~0.1s, rerank ~0.2s, Gemini ~3s

### Decisions

**Settings are passed as a parameter, not imported as a module singleton.**
Functions take `settings` explicitly so a test can hand in `Settings(chunk_size=50)`
without touching environment variables. `query.py` still holds a module-level
instance as an interim step; it goes away when model loading moves into a class.

**The API key is a `SecretStr`.** As a plain `str` it appears in every `repr()`, log
line and traceback — which is how keys reach CI logs and pasted stack traces.
`.get_secret_value()` is called at exactly one place, where the Gemini client is built.

**Config fails loudly at startup.** `gemini_api_key` has no default, so a missing key
raises `ValidationError` naming the field, instead of passing `None` down to surface
200 lines later as an unrelated auth error. Verified by hiding `.env` and running.

**`n_rerank <= n_retrieve` is enforced by a `@model_validator(mode="after")`.**
A field cannot reference another field inside `Field()` — at class-definition time
the neighbour is a `FieldInfo`, not a value — so cross-field rules need a validator
that runs once every field is populated.

**No bare `model` as a name.** Three "models" are in scope (Gemini, embedding,
reranker). A field called `model` caused the same line to be wrong three times in a
row, ending with `settings.reranker_model` being passed to Gemini as its model name
and returning `404 Not Found`. Renamed `gemini_model`; the call site now reads as
either obviously right or obviously wrong.

### Things that cost time

**The venv on `/mnt/c` was the performance problem.** `import sentence_transformers`
took minutes. Ctrl-C showed the stack sitting in `inspect.getmodule` → `realpath` —
torch is tens of thousands of small files and every stat crossed WSL's translation
layer to the Windows filesystem. Moving the venv to `$HOME/.venvs/hpotter` cut a cold
import to ~15s. uv's cache was already on Linux so nothing was re-downloaded, and the
hardlink warning went away as a side effect.

**A clean linter is not a working program.** Ruff went green while three real bugs
remained: a wrong attribute (`settings.reranker_model` to Gemini), a hardcoded
collection name, and an `os.getenv()` wrapped around an already-resolved secret.
Ruff checks undefined *names*, not wrong *values*.

**Interim singleton.** Silencing seven `F821 Undefined name 'settings'` errors by
adding a module-level `settings = Settings()` works, but it is the singleton the
refactor was meant to avoid. Accepted deliberately as a stepping stone.

---

## 2026-09-10 — Phase 2: evals, in progress

Wrote the golden question set. Runner not built yet.

### Corpus findings

- **The text is the US edition.** `Sorcerer` 16 hits, `Philosopher` 0. Likewise
  `Defense` 5 / `Defence` 0, and `seven hundred and thirteen` 5 / `713` 0.
- **Case sensitivity matters more than expected.** `Invisibility Cloak`: 0 hits
  case-sensitive, 13 case-insensitive. `nine and three-quarters`: 2 vs 6 — the misses
  are a chapter heading in caps and two sentence starts. Scoring must lowercase both
  sides or it reports failures that never happened.
- **Harry's birth date is never stated in book one.** `July` appears 3×: a Hogwarts
  letter deadline (1664), the Gringotts break-in date (4814), Aunt Petunia's trip to
  London (986). None says it is his birthday.
- **It is reachable in exactly two hops**: 4827 (*"that Gringotts break-in happened
  on my birthday!"*) + 4814 (*"the break-in at Gringotts on 31 July"*). Kept as the
  motivating case for the LangGraph phase — a graph that decomposes and retrieves
  twice should pass where single-shot cannot.

### The Tuesday reversal — worth remembering

Two hand-run queries suggested retrieval was broken: the broom question returned
`I don't know`, and *"when is Harry's birthday"* returned *"Harry's birthday is on a
Tuesday."* Both looked like retrieval failures.

Line 1371 reads *"then tomorrow, Tuesday, was Harry's eleventh birthday"*. The
retrieval was correct, the grounding rule worked, and the answer was accurate for the
question as asked. Retrieval was nearly "fixed" on the strength of two anecdotes.

This is the whole argument for the eval phase: two samples cannot distinguish a
broken retriever from an ambiguous question, and a confident wrong answer looks
identical to a confident right one from the outside.

### Decisions

- **All keywords must match**, not any — evaluated across the whole retrieved set
  rather than within a single chunk. Chunk boundaries at 500 chars are arbitrary, so
  requiring co-location would measure the chunker's luck rather than retrieval quality.
- **Every keyword is grep-verified against the corpus before entering the set.**
  A keyword that does not exist scores as a permanent retrieval failure no matter how
  good retrieval gets. A wrong eval is worse than no eval.
- **Cases carry a `category`** (`extractable` / `multi_hop` / `unanswerable`) because
  they cannot share a scoring rule — an unanswerable case passes when the system
  refuses, which the keyword rule cannot express.

---

## 2026-09-10 — Finding: retrieval is phrasing-sensitive

While building the retrieval scorer, tested the broom case in two phrasings against
the same index:

| query | chunk containing both `McGonagall` and `Nimbus` |
|---|---|
| "Who gave Harry his first broomstick?" | chunk 2 (of 5 reranked) |
| "who gifted harry his broom" | none |

The original hand-run failure used the casual phrasing. It was never a retrieval bug
or a chunking bug — `gifted`/`broom` simply embeds further from the passage than
`gave`/`broomstick`. Chroma's top-10 contained both keywords in both cases; only the
formal phrasing put them in the *same* chunk.

Consequences:

- **The eval set should carry paraphrases per case**, not one canonical phrasing.
  Scoring only the formal wording measures a system real users do not use.
- **Union-across-the-retrieved-set is too lenient as a scoring rule** when a keyword
  is common. `McGonagall` appears 100× in book one, so the union test passed while no
  single chunk actually answered the question. Per-chunk co-location is the stricter,
  more honest signal — at the cost of being sensitive to chunk boundaries.
- **This is the motivating case for Phase 3.** LangChain's `MultiQueryRetriever`
  generates several phrasings and unions the results, which is precisely this problem.
  Adopt it against a measured baseline, not on principle.

Also confirmed the two-measurement-point design was worth it: Chroma's top-10 held
both keywords, so measuring only the final answer would have pointed at the wrong
layer entirely.

---

## 2026-09-10 — Retrieval baseline

First measured baseline. Settings: `chunk_size=500`, `chunk_overlap=100`,
`n_retrieve=10`, `n_rerank=5`, `all-MiniLM-L6-v2`, `ms-marco-MiniLM-L6-v2`.
Scoring rule: per-chunk co-location — a case passes when a **single** chunk contains
**all** its keywords, matched case-insensitively. 16 cases scored (`unanswerable`
excluded).

| metric | score |
|---|---|
| hit@10 (Chroma) | 10/16 |
| hit@5 (after rerank) | 9/16 |
| dropped by rerank | 1 |

### Failure breakdown

| case | keywords in top-10 | diagnosis |
|---|---|---|
| On what date is Harry's birthday? | `31 July` ✗ | expected — multi-hop |
| Dumbledore's Christmas gift | `Invisibility Cloak` ✓ `father` ✓ | co-location |
| What did Hagrid name his dragon? | `Norbert` ✓ | rerank drop |
| Quidditch position | `Seeker` ✗ | retrieval miss |
| What is Harry's wand made of? | `holly` ✗ `phoenix` ✓ | retrieval miss |
| Mirror of Erised | `Erised` ✓ `desire` ✗ | retrieval miss |
| Vault seven hundred and thirteen | `seven hundred and thirteen` ✓ `grubby little package` ✗ | retrieval miss |

### Reading

Four of seven failures share one shape: Chroma returns chunks that are *topically*
correct (the Mirror scene, Ollivanders, Quidditch) but misses the sentence carrying
the fact. `Erised` is retrieved but not `desire`; `phoenix` but not `holly`. This is
the known weakness of dense retrieval with a small embedding model over short chunks —
it matches gist, and rare exact terms wash out. Hybrid retrieval (BM25 unioned with
vector search) is the standard fix and would be a second measured justification for
Phase 3, alongside `MultiQueryRetriever` for the phrasing sensitivity.

Only one case is lost by the reranker, so the cross-encoder is not the main problem —
though it is worth noting it was added in `3066ecd` ("added reranker to improve
answers") with no measurement to confirm it did.

### Next experiment

`n_retrieve` 10 → 25, nothing else changed. One value in `Settings`, no code change.

- **Prediction:** if `Seeker`, `holly` and `desire` appear at 25, retrieval is fine
  but shallow — raise `n_retrieve` and `n_rerank` and move on.
- **If they are still absent at 25**, the embedding genuinely cannot reach them and
  hybrid search is the real fix, not a deeper vector search.

---

## 2026-09-10 — Experiments: retrieval depth, and does the reranker earn its place

All at `chunk_size=500`, `chunk_overlap=100`. 16 cases scored, per-chunk co-location
rule. "Final chunks" is what would reach the LLM.

| run | `n_retrieve` | selection | final chunks | hit@retrieve | **final score** |
|---|---|---|---|---|---|
| baseline | 10 | cross-encoder → 5 | 5 | 10/16 | **9/16** |
| depth | 25 | cross-encoder → 5 | 5 | 13/16 | **9/16** |
| ceiling | 25 | none (all 25) | 25 | 13/16 | 13/16 (not comparable) |
| **A** | 25 | cross-encoder → 5 | 5 | 13/16 | **9/16** |
| **B** | 25 | plain vector top-5 | 5 | 13/16 | **7/16** |

### Depth: raised the ceiling, not the score

`n_retrieve` 10 → 25 lifted hit@retrieve from 10/16 to 13/16 — `Seeker`, `holly`,
`Norbert` and the Dumbledore gift all appear once Chroma looks deeper. The final score
did not move: every gain was discarded by the reranker, and rerank drops went 1 → 4.

One case **regressed**: "What was written on the cake" passed at `n_retrieve=10` and
failed at 25. Nothing changed but the size of the candidate pool — the cross-encoder
had more distractors and preferred them. A retrieval improvement made the system
worse, which only the two-measurement-point design makes visible.

### A vs B: the reranker wins

The `n_rerank=25` "ceiling" run is **not** a fair reranker-off test — `hit@k` rises
mechanically with `k`, so it measures context size, not selection quality. The fair
comparison holds final chunks fixed at 5:

- **A** (cross-encoder picks 5 of 25) → 9/16
- **B** (plain vector top 5) → 7/16

The cross-encoder finds two more answers than vector similarity at identical context
cost. Commit `3066ecd` ("added reranker to improve answers") was right; this is the
first measurement confirming it.

### Where the remaining wins are

The cross-encoder returns the same 9 answers whether handed 10 candidates or 25, so
the extra depth is currently wasted and **the 9 → 13 gap is entirely rerank selection
quality.**

Three cases are unreachable at any `k` and are not rerank's fault:

- `31 July` — multi-hop, no single chunk contains it (expected)
- `desire` — genuine dense-retrieval miss, absent even at 25
- vault — both keywords present at 25 but never co-located in one chunk

So 13/16 is the realistic target and the system is 4 short.

### Method note

Retrieval-only scoring cannot see whether the LLM *uses* the retrieved text. A large
`hit@25` can coexist with worse answers — long contexts cost more and models degrade
at finding facts buried mid-prompt. Answer-level evaluation is the next phase.

### Next experiment

Sweep `n_rerank` ∈ {5, 8, 10, 15} at `n_retrieve=25`, reranker on. Looking for where
the curve flattens, not the maximum — each step adds context cost. If that flattens
early, the lever is a stronger reranker (`ms-marco-MiniLM-L12`, `bge-reranker-base`),
now a one-line `Settings` change and directly comparable against 9/16.

---

## 2026-09-10 — Experiment: chunk_size 500 → 800

Re-ingested at `chunk_size=800` (1224 chunks → 692). All else held: `n_retrieve=25`,
`n_rerank=5`, reranker on.

| | 500 | 800 |
|---|---|---|
| hit@25 (ceiling) | 13/16 | 13/16 |
| **final (rerank → 5)** | **9/16** | **10/16** |

Nothing regressed. The dilution risk — larger chunks smearing the embedding so a
topically-broad chunk outranks the answer-bearing sentence — did not materialise at
800. **Adopted 800 as the default.**

### Failures changed character, which matters more than +1

- **Mirror of Erised** — `desire` was absent from the top 25 at 500, present at 800.
  Still fails, because `Erised` and `desire` are now both retrieved but not in the
  *same* chunk. Moved from retrieval miss → co-location.
- **Dumbledore's gift** — was co-location, now `25:1 5:0`: a chunk containing both
  exists in the top 25 and the reranker did not select it. Moved from co-location →
  rerank drop.
- **Dragon** now passes outright (was the sole rerank drop at 500).

Remaining six failures sort into three causes:

| cause | cases |
|---|---|
| co-location — chunks still too small | Erised, vault |
| rerank selection | Seeker, wand, Dumbledore's gift |
| structural | 31 July (multi-hop) |

### Note on method

The ceiling stayed at 13/16 across both chunk sizes, which is a useful invariant: the
same three cases are unreachable regardless of chunking, so they are not a chunking
problem and no amount of chunk tuning will move them.

### Next experiments — one variable at a time

1. `chunk_size` 800 → 1200 (requires re-ingest). Targets Erised and vault. This is
   where dilution should begin to cost something — a case flipping from pass to fail
   is the signal to stop.
2. `n_rerank` 5 → 8 or 10 (no re-ingest). Targets the three rerank drops. Cheaper, so
   arguably first.

---

## 2026-09-10 — Discarded run, and ingest has no availability story

A run scored **7/16** and looked like a severe regression: `Fluffy`, `Seeker`,
`Erised`, `desire`, `Norbert`, `Invisibility Cloak` and `father` all vanished from the
top 25, having been retrievable minutes earlier at the same `chunk_size=800`.

**Cause: the eval ran while a rebuild was in progress.** `reset_collection` deletes
the collection and then repopulates it chapter by chapter, and ingest takes ~7 minutes
on this machine. Queries during that window see a partial index.

The missing keywords sort cleanly by chapter, which is what identified it:

| keyword | chapter | found |
|---|---|---|
| `McGonagall`, `Nimbus` | early / 10 | yes |
| `Seeker` | 11 | no |
| `Erised`, `Invisibility Cloak`, `father` | 12 | no |
| `Norbert` | 14 | no |
| `Fluffy` | 16 | no |

Everything up to chapter 10 was present, everything from 11 on was absent — those
chapters had not been inserted yet. **The 7/16 run is discarded.** A clean re-run
returned 10/16, case-for-case identical to the earlier 800 measurement, which also
confirms the pipeline is reproducible across rebuilds.

Two hypotheses were wrong and worth noting as such:

- **Embedding mismatch** (Chroma's ONNX `all-MiniLM-L6-v2` at ingest vs PyTorch
  `SentenceTransformer` at query). Tested directly: both find `Fluffy`, and the two
  25-result sets are **identical**, 25/25 overlap. Not the cause — though the two
  runtimes are still an unenforced coupling worth closing.
- **HNSW nondeterminism.** Ruled out by the reproducible re-run.

### Findings

**Ingest has no availability story.** Delete-then-repopulate means ~7 minutes during
which the system returns confident partial answers with no error. In production that
is an outage presenting as a quality problem. The standard fix is to build into a new
collection and swap names at the end, so readers only ever see a complete index.

**Ingest is slow for the wrong reason.** 7m14s for 692 chunks is ~0.6s/chunk, far
above MiniLM's compute cost. `chroma/` lives on `/mnt/c`, so every upsert writes
SQLite pages and HNSW updates across the WSL filesystem boundary — the same problem as
the venv. Since `chroma/` is gitignored, the path can move to native Linux. This
matters directly: every chunk-size experiment requires a re-ingest.

**Method note.** Third time today a plausible explanation lost to a direct check —
after the Tuesday reversal and the phrasing-sensitivity finding. The embedding-mismatch
theory was well-reasoned and wrong.

---

## Open threads

- [ ] Pass an explicit embedding function to `get_or_create_collection` so ingest and
      query provably share one model read from `settings.embedding_model` (currently
      compatible by coincidence, not by construction)
- [ ] Add a summary to the eval runner: totals per category and a hit@retrieve /
      hit@selected line, so a run ends in numbers rather than 16 lines to count by hand
      (the per-case loop itself is done)
- [ ] Extract a single `retrieve(collection, question, settings)` into `query.py` and
      call it from both `main()` and `run_evals.py`. Right now the selection step is
      duplicated, and `use_reranker` affects only the eval — so the eval scores a
      pipeline the app does not run.
- [ ] Flip `use_reranker` default to `True` — arm A (9/16) beat arm B (7/16), so the
      current default ships the worse configuration
- [ ] Decide the scoring rule: union-across-set vs per-chunk co-location. The broom case
      shows union is too lenient when a keyword is common; per-chunk is chunk-boundary
      sensitive. Possibly record both, as with hit@10 / hit@5.
- [ ] Add a `paraphrases` field to `EvalCase` and score each case in several phrasings
- [ ] Move model loading out of import time into a class; drop the module-level
      `settings` singleton at the same time
- [ ] `cli.py` is empty, but `pyproject.toml` declares `hpotter = "hpotter.cli:main"`
- [ ] `tests/` is empty; pytest and ruff are configured but unused
- [ ] `ingest.py` calls `collection.add` on every run rather than upserting — re-running
      duplicates chunks. Check whether the 1224 count is already inflated.
- [ ] Book 2 is in `resources/` but never ingested
- [ ] Verify `uv run python -m hpotter.ingest` still works post-refactor (only `query`
      has been run since)
