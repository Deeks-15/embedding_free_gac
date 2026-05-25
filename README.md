# Embedding-free GAC — empirical pilot

The architecturally-authentic Generative Address Convergence (GAC) — a
retrieval system where the hot path uses **zero embeddings and zero cosine
similarity**. All routing is deterministic (term-based / regex-based);
ranking is BM25 + recency + level + keyword overlap.

This is the architecture the GAC whitepaper §3 actually claims:
> *"Retrieval itself never performs semantics again."*

For the **softened embedding-based variant** (everything else of the pilot,
including the §12 tuning attribution that did not generalize), see the sibling
repo [publish_bounded_ann](../publish_bounded_ann/).

## What's in this repo

Two implementations of GAC, one per data-shape:

| Variant | Domain | Routing primitive | Fallback |
|---|---|---|---|
| **DrainGAC** | Logs (Tier 1) | Drain3 templates (regex with wildcards) | BM25 over address summaries |
| **DocGAC** | Prose docs (Tier 2) | YAKE keyword extraction + Jaccard overlap | BM25 over chunk text |

Both share the same architectural framework — only the routing primitive
differs by data shape. No embedding model is loaded at any point on the
hot path.

## The empirical question this repo answers

> *Does GAC actually deliver on its whitepaper §3 claim — retrieval without
> per-query semantics — and does that win or lose vs RAG?*

**Headline:**
- **Logs (DrainGAC)** — wins decisively on tuning corpus (+11.9pp hit@5 vs RAG),
  holds within noise on held-out (-2.4pp). Zero hot-path embeddings.
- **Documents (DocGAC)** — trails RAG on tuning corpus (-15.9pp), within-noise
  tie on held-out (-4.8pp inside 7.1pp judge noise). Zero hot-path embeddings.

Together these validate the **whitepaper §1.1 Tier framework**: GAC wins on
Tier 1 data (templated, with discrete structure to recover), trails on Tier 2
data (free-form prose, no clean deterministic primitive).

## Layout

```
publish_embedding_free_gac/
├── README.md                  this file
├── requirements.txt           pip deps
├── .env.example               copy to .env and fill in GEMINI_API_KEY (used ONLY for judge eval)
├── src/
│   ├── drain_gac.py           the Tier 1 (logs) variant
│   ├── doc_gac.py             the Tier 2 (docs) variant
│   ├── phase3_drain.py        RAG vs DrainGAC driver
│   ├── phase4_docs.py         RAG vs DocGAC driver (needs your own corpus — see below)
│   ├── charts_phase3.py       Appendix F charts
│   ├── charts_phase4.py       Appendix G charts
│   └── ...                    shared deps (chunker, schedulers, judge harness)
├── logs/                      pre-generated synthetic log corpora (a, b) — ship runnable
├── corpus/                    EMPTY — bring your own documents (see corpus/README.md)
├── data/                      raw result JSONs (log benchmarks only; doc benchmark not bundled)
└── reports/
    └── charts/                F_*.png (DrainGAC) + G_*.png (DocGAC + Tier validation)
```

## Bring your own document corpus

The original DocGAC benchmark ran against a proprietary 31-document
corpus that cannot be republished. The `corpus/` folder ships **empty**,
and `data/phase4_results.json` + `data/chunks.jsonl` are **not bundled**.

To run the DocGAC vs RAG comparison (Phase 4), drop your own documents
into a folder and follow the steps in [corpus/README.md](corpus/README.md).
You will need to:
1. Extract + chunk your documents.
2. Replace the `EVAL_QUERIES` in `src/phase4_docs.py` with queries
   grounded in your corpus.
3. Re-run `python src/phase4_docs.py`.

The **DrainGAC (logs) benchmark in Phase 3 needs none of this** — the
synthetic log corpora in `logs/` ship runnable, and
`data/phase3_results.json` is bundled.

## The architectural invariant

The defining property of this implementation:

```python
# In drain_gac.py and doc_gac.py — verified at code level
# Deliberately NOT importing numpy/sentence_transformers/sklearn cosine
import re, time, math
from collections import Counter, defaultdict
from drain3 import TemplateMiner  # OR yake — both pure-Python, no neural net
```

The embedding model is **never loaded** during ingestion or retrieval.
Measured across 50,000+ events and 84 queries on logs: **0 embedding calls**
on the hot path.

The cartographer (LLM-based address minting) is still available for the warm
path, but DrainGAC's tests run with `skip_cartographer=True` and use
deterministic naming from template keywords — proving the architecture can
function with zero LLM dependency end-to-end.

## Reproducing

### 1. Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# GEMINI_API_KEY only needed for the LLM-judge eval (Gemini judges relevance);
# the GAC architecture itself works with NO API key.
```

### 2. Re-generate log corpora (optional; pre-generated files included)

```bash
python src/realistic_scheduler.py --truncate --rate 200 --duration 150
python src/realistic_scheduler_b.py --truncate --rate 200 --duration 150
```

### 3. Run the log head-to-head (no extra setup — ships runnable)

```bash
# Logs (DrainGAC vs RAG, both corpora)
python src/phase3_drain.py
```

### 4. Run the document head-to-head (requires your own corpus)

The bundled doc benchmark ran against a proprietary corpus that cannot
be republished. To reproduce on your own documents, see
[corpus/README.md](corpus/README.md), then:

```bash
DOC_SOURCE_DIR=/path/to/your/docs python src/extract.py
python src/chunk.py
# Edit src/phase4_docs.py — replace EVAL_QUERIES with queries for YOUR corpus
python src/phase4_docs.py
```

### 5. Regenerate reports + charts

```bash
python src/charts_phase3.py
python src/phase3_report.py
# charts_phase4.py + phase4_report.py require data/phase4_results.json
# (re-run step 4 above to generate it on your own corpus)
python src/charts_phase4.py
python src/phase4_report.py
```

## Headline results

### DrainGAC (logs, Tier 1)

| Corpus | RAG hit@5 | DrainGAC hit@5 | Delta | Verdict |
|---|---:|---:|---:|---|
| realistic (a) | 57.1% | **69.0%** | **+11.9pp** | **GAC WINS** |
| held-out (b) | 50.0% | 47.6% | -2.4pp | parity (within 0pp noise) |

### DocGAC (prose, Tier 2)

| Half | RAG hit@5 | DocGAC hit@5 | Delta | Verdict |
|---|---:|---:|---:|---|
| doc-a (train) | 77.0% | 61.1% | -15.9pp | RAG wins |
| doc-b (held-out) | 74.6% | 69.8% | -4.8pp | **within 7.1pp noise — tie** |

### Hot-path embedding calls (across all four corpora)

| System | Per-event embed calls | Per-query embed calls |
|---|---:|---:|
| RAG | 1 per event | 1 per query |
| **DrainGAC / DocGAC** | **0** | **0** (1 anomaly across 84 queries — measurement glitch) |

## Methodology disciplines

(Same as the bounded-ANN repo)

- Pinned `random_seed=42`
- Judge `temperature=0`, rubric SHA byte-identical asserted before each call
- SIGALRM 60s per-call timeout (interrupts blocking socket I/O — unlike `concurrent.futures.ThreadPoolExecutor`)
- Per-(query, pass) checkpoint to disk — idempotent restart
- Pooled-once judging: each query pool = RAG top-5 ∪ GAC top-5; judged 3 independent times for noise-floor measurement
- Held-out corpus frozen before any GAC build; not modified after
- Judge: Gemini 1.5 Flash (`gemini-2.0-flash-exp` per `.env`) — used only for relevance labeling, NOT for retrieval

## What this repo proves (defensible claims)

1. **The architectural property holds**: 0 embedding calls across 50,000+ events + 84 queries.
2. **GAC wins on Tier 1 data**: +11.9pp on logs (tuning), parity on held-out.
3. **GAC loses on Tier 2 data**: -15.9pp on docs (tuning), within-noise tie on held-out.
4. **The Tier framework predicts correctly** — both halves measured, both predictions match.
5. **Cost savings hold universally**: no vector DB needed (DrainGAC stores templates in dict; DocGAC stores chunks under deterministic addresses).

## What this repo does NOT prove

- **Tier 1 generalization at scale beyond pilot (~25k entries).** HNSW degrades log(N); template lookup is O(1) — projected to widen the gap, but not measured.
- **Real production data.** Both log corpora are synthetic-generator-based.
- **Stronger primitives for prose.** Only YAKE + BM25 tested for documents; hierarchical sections, NER, learned-TF-IDF clustering all untested.
- **Adversarial drift.** Phase B (drift handling) only tests planned drift; adversarial template changes untested.

## License

MIT (see LICENSE file in published repo).
