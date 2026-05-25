"""Phase 4 driver — RAG vs DocGAC on document corpus (split half).

NOTE FOR PUBLIC-REPO USERS:
  The original benchmark ran against a proprietary ~31-document corpus
  that is NOT included here — see corpus/README.md for bring-your-own
  instructions. The DOC_QUERIES list below is shipped EMPTY by design;
  populate it with queries grounded in YOUR own corpus before running
  this driver. See the schema comment above DOC_QUERIES for the entry
  format.


Tests the embedding-free architecture on PROSE documents (not logs).
DocGAC uses:
  - YAKE keyword extraction (deterministic, term-based)
  - Jaccard-overlap clustering for address space
  - BM25 fallback (term-based) when no keyword match
  - ZERO embeddings, ZERO cosine on the hot path

Setup (Option B — defensible held-out split):
  - 31 doc files in pilot/corpus/ are split 16 / 15 with pinned random seed
  - "Realistic (doc-a)" = chunks from train half
  - "Held-out (doc-b)" = chunks from test half
  - Same 40 queries run on both halves
  - Queries are topic-level (cross-cutting themes that appear in both halves)

Methodology lock — same as Phase 3:
  random_seed=42, judge temp=0, byte-identical rubric SHA, pooled-once,
  3 judge passes per pool, SIGALRM 60s per-call timeout, idempotent restart.

Output:
  data/phase4_results.json
  data/phase4_cache/doc_a/  data/phase4_cache/doc_b/
"""
from __future__ import annotations
import hashlib
import json
import os
import random
import signal
import time
from pathlib import Path
from typing import Dict, Any, List

import sys
sys.path.insert(0, str(Path(__file__).parent))

# load .env
ENV_PATH = Path(__file__).parent.parent / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from streaming_realistic import JUDGE_PROMPT, Judge
from streaming_replay import StreamingRAG
from doc_gac import DocGAC

DATA = Path(__file__).resolve().parent.parent / "data"
CHUNKS_JSONL = DATA / "chunks.jsonl"
RESULTS = DATA / "phase4_results.json"

RANDOM_SEED = 42
K = 5
BATCH = 5000
JUDGE_REPEATS = 3
PER_CALL_TIMEOUT_S = 60
PER_CALL_RETRIES = 2


# DOC_QUERIES — YOU MUST POPULATE THIS LIST.
#
# The original benchmark used ~40 queries grounded in a proprietary
# document corpus (NOT included in this repo — see corpus/README.md).
# The list is shipped EMPTY by design, because corpus-specific
# placeholders would be misleading (they'd suggest the queries match
# the bundled corpus, which is itself empty).
#
# Schema for each entry:
#   {
#     "id": "dNN",
#     "kind": "simple" | "paraphrase" | "abstract" | "adversarial",
#     "query": "<your question>",
#     "expect_topic": "<short topic tag — used for logging/grouping only>",
#   }
#
# Recommended composition: ~30–40 queries covering your corpus themes.
# Mix kinds: simple (direct lookup), paraphrase (same intent / different
# wording — judges co-location recall), abstract (cross-cutting concepts),
# adversarial (negation / out-of-corpus questions).
#
# Run after populating:
#   python src/phase4_docs.py
DOC_QUERIES = [
    # POPULATE ME
]


def _require_queries():
    if not DOC_QUERIES:
        raise RuntimeError(
            "src/phase4_docs.py:DOC_QUERIES is empty. The published repo "
            "ships this list empty by design — populate it with queries "
            "grounded in YOUR corpus (see the schema comment above this "
            "list, and corpus/README.md for the full BYOC workflow)."
        )


class _CallTimeout(Exception): pass
def _alarm_handler(signum, frame): raise _CallTimeout()


def judge_with_timeout(judge, query, pool, timeout_s=PER_CALL_TIMEOUT_S,
                       retries=PER_CALL_RETRIES):
    attempt = 0
    while attempt <= retries:
        attempt += 1
        old = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout_s)
        try:
            labels = judge.judge(query, pool)
            signal.alarm(0)
            return labels
        except _CallTimeout:
            print(f"  [TIMEOUT] judge call exceeded {timeout_s}s "
                  f"(attempt {attempt}/{retries+1})")
        except Exception as e:
            signal.alarm(0)
            print(f"  [judge ERROR] {e} (attempt {attempt}/{retries+1})")
        finally:
            signal.signal(signal.SIGALRM, old)
        time.sleep(2 ** attempt)
    return None


def split_corpus(chunks: List[Dict[str, Any]]):
    """Deterministic 50/50 split by doc filename. Seed pinned."""
    docs = sorted(set(c["doc"] for c in chunks))
    rng = random.Random(RANDOM_SEED)
    docs_shuf = list(docs)
    rng.shuffle(docs_shuf)
    half = len(docs_shuf) // 2 + (len(docs_shuf) % 2)   # 16 for 31 docs
    docs_a = set(docs_shuf[:half])
    docs_b = set(docs_shuf[half:])
    chunks_a = [c for c in chunks if c["doc"] in docs_a]
    chunks_b = [c for c in chunks if c["doc"] in docs_b]
    return docs_a, docs_b, chunks_a, chunks_b


def run_one_half(half_key: str, chunks: List[Dict[str, Any]], doc_set: set):
    print(f"\n{'='*70}")
    print(f"CORPUS doc-{half_key} ({len(doc_set)} docs, {len(chunks)} chunks)")
    print(f"  files: {sorted(doc_set)[:3]} ... {sorted(doc_set)[-2:]}")
    print(f"{'='*70}\n")

    cache_root = DATA / "phase4_cache" / f"doc_{half_key}"
    cache_root.mkdir(parents=True, exist_ok=True)
    judge_cache = cache_root / "judge"
    judge_cache.mkdir(exist_ok=True)
    rag_cache_f = cache_root / "rag_queries.json"
    docgac_cache_f = cache_root / "docgac.json"

    # RAG (cached)
    if rag_cache_f.exists():
        print("[cache HIT] RAG queries")
        rag_results = json.loads(rag_cache_f.read_text())
    else:
        print("building RAG...")
        rag = StreamingRAG()
        # Reuse StreamingRAG (it embeds + indexes any chunks)
        t0 = time.perf_counter()
        for off in range(0, len(chunks), BATCH):
            rag.ingest(chunks[off:off+BATCH])
        rag_secs = time.perf_counter() - t0
        print(f"  RAG built in {rag_secs:.1f}s ({rag.n_indexed} entries)")
        rag_results = {}
        for q in DOC_QUERIES:
            r = rag.query(q["query"], k=K)
            rag_results[q["id"]] = {
                "query": q["query"], "kind": q["kind"],
                "rag_hits": [(h["id"], h["text"][:1500], h.get("score"))
                              for h in r["hits"]],
                "rag_total_ms": r["total_ms"],
            }
        rag_results["_meta"] = {"rag_build_seconds": rag_secs}
        rag_cache_f.write_text(json.dumps(rag_results, default=str))

    # DocGAC (cached)
    if docgac_cache_f.exists():
        print("[cache HIT] DocGAC build")
        dg_data = json.loads(docgac_cache_f.read_text())
    else:
        print("building DocGAC...")
        random.seed(RANDOM_SEED)
        dgac = DocGAC(random_seed=RANDOM_SEED)
        boot = dgac.bootstrap(chunks)
        print(f"  DocGAC: {boot['total_addresses']} addresses, "
              f"{boot['total_mint_llm_calls']} mint calls, "
              f"0 embedding calls")

        dg_results = {}
        det_count = bm25_count = empty_count = 0
        total_embed = 0
        for q in DOC_QUERIES:
            r = dgac.query(q["query"], k=K)
            if r["fallback_used"]:
                if r["fallback_kind"].startswith("empty"):
                    empty_count += 1
                else:
                    bm25_count += 1
            else:
                det_count += 1
            total_embed += r.get("embed_calls_this_query", 0)
            dg_results[q["id"]] = {
                "docgac_hits": [(h["id"], h["text"][:1500], h.get("score"),
                                  h.get("via_address", ""))
                                for h in r["hits"]],
                "docgac_total_ms": r["total_ms"],
                "docgac_routed_address": r["routed_address"],
                "docgac_reduction": r["reduction_ratio"],
                "docgac_fallback_used": r["fallback_used"],
                "docgac_fallback_kind": r.get("fallback_kind", ""),
                "docgac_embed_calls": r.get("embed_calls_this_query", 0),
            }
        print(f"  query routing: deterministic={det_count}, BM25={bm25_count}, "
              f"empty={empty_count}, query-time embed calls={total_embed}")

        dg_data = {
            "build_stats": {
                "total_addresses": boot["total_addresses"],
                "total_mint_llm_calls": 0,
                "total_edge_llm_calls": 0,
                "total_cartographer_usd": 0.0,
                "build_secs": boot["build_secs"],
                "n_addresses": boot["total_addresses"],
            },
            "query_results": dg_results,
            "query_routing": {
                "deterministic": det_count,
                "fallback_bm25": bm25_count,
                "empty_no_match": empty_count,
                "total_query_time_embeds": total_embed,
            },
            "address_snapshot": [{
                "address": a["address"], "summary": a["summary"],
                "top_keywords": a["top_keywords"],
                "n_chunks": a["n_chunks"],
                "dominant_doc": a["dominant_doc"],
                "doc_purity": a["doc_purity"],
            } for a in dgac.addresses],
        }
        docgac_cache_f.write_text(json.dumps(dg_data, default=str))

    # Judge
    rubric_hash = hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()
    judge = Judge()

    pools = {}
    for q in DOC_QUERIES:
        qid = q["id"]
        pool = {}
        for cid, txt, *_ in rag_results[qid]["rag_hits"]:
            pool[cid] = {"id": cid, "text": txt, "from": ["rag"]}
        for cid, txt, *_ in dg_data["query_results"][qid]["docgac_hits"]:
            if cid in pool:
                pool[cid]["from"].append("docgac")
            else:
                pool[cid] = {"id": cid, "text": txt, "from": ["docgac"]}
        pools[qid] = list(pool.values())
    sizes = [len(p) for p in pools.values()]
    print(f"  pool sizes: min={min(sizes)} max={max(sizes)} "
          f"avg={sum(sizes)/len(sizes):.1f}")

    total = len(DOC_QUERIES) * JUDGE_REPEATS
    already = sum(1 for q in DOC_QUERIES for p in range(JUDGE_REPEATS)
                  if (judge_cache / f"{q['id']}_p{p}.json").exists())
    print(f"  judge cache: {already}/{total} already done")

    done = already
    for q in DOC_QUERIES:
        for pi in range(JUDGE_REPEATS):
            cf = judge_cache / f"{q['id']}_p{pi}.json"
            if cf.exists():
                continue
            actual = hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()
            assert actual == rubric_hash, "RUBRIC DRIFT!"
            labels = judge_with_timeout(judge, q["query"], pools[q["id"]])
            if labels is None:
                cf.write_text(json.dumps({
                    "qid": q["id"], "pass": pi, "timed_out": True,
                    "labels": None}))
            else:
                cf.write_text(json.dumps({
                    "qid": q["id"], "pass": pi, "timed_out": False,
                    "labels": labels}))
                done += 1
                if done % 5 == 0:
                    print(f"  [{done}/{total}] judge calls "
                          f"(~${judge.total_usd:.4f})")

    # Score (3 passes → noise floor)
    all_labels = {}
    for q in DOC_QUERIES:
        all_labels[q["id"]] = []
        for pi in range(JUDGE_REPEATS):
            d = json.loads((judge_cache / f"{q['id']}_p{pi}.json").read_text())
            all_labels[q["id"]].append({} if d.get("timed_out") else d["labels"])

    def _score(system_name: str, hits_field: str, hits_data: Dict):
        pass_scores = []
        for pi in range(JUDGE_REPEATS):
            per_q = []
            for q in DOC_QUERIES:
                qid = q["id"]
                labels = all_labels[qid][pi]
                top5 = [c for c, _, *_ in hits_data[qid][hits_field]]
                rel = [labels.get(c, False) for c in top5]
                per_q.append({
                    "qid": qid, "kind": q["kind"],
                    "hit_at_k": any(rel),
                    "precision_at_k": sum(rel) / max(1, len(rel)),
                })
            n = len(per_q)
            def avg(xs): return sum(xs)/max(1, len(xs))
            agg = {
                "hit_at_k_rate": sum(q["hit_at_k"] for q in per_q) / n,
                "precision_at_k_avg": avg([q["precision_at_k"] for q in per_q]),
            }
            by_kind = {}
            for kind in sorted({q["kind"] for q in per_q}):
                sub = [q for q in per_q if q["kind"] == kind]
                by_kind[kind] = {
                    "n": len(sub),
                    "hit_at_k": sum(q["hit_at_k"] for q in sub) / len(sub),
                    "precision_at_k": avg([q["precision_at_k"] for q in sub]),
                }
            pass_scores.append({"aggregate": agg, "by_kind": by_kind})

        def stats(vs): return {
            "min": min(vs), "mean": sum(vs)/len(vs), "max": max(vs),
            "range": max(vs)-min(vs),
        }
        agg_s = {}
        for m in ("hit_at_k_rate", "precision_at_k_avg"):
            agg_s[m] = stats([p["aggregate"][m] for p in pass_scores])
        bk_s = {}
        for kind in sorted(pass_scores[0]["by_kind"]):
            n = pass_scores[0]["by_kind"][kind]["n"]
            bk_s[kind] = {"n": n}
            for m in ("hit_at_k", "precision_at_k"):
                bk_s[kind][m] = stats([p["by_kind"][kind][m] for p in pass_scores])
        return {"aggregate_stats": agg_s, "by_kind_stats": bk_s}

    rag_score = _score("rag", "rag_hits", rag_results)
    dg_score = _score("docgac", "docgac_hits", dg_data["query_results"])

    rag_h = rag_score["aggregate_stats"]["hit_at_k_rate"]
    dg_h = dg_score["aggregate_stats"]["hit_at_k_rate"]
    delta = (dg_h["mean"] - rag_h["mean"]) * 100
    nf = rag_h["range"] * 100

    print(f"\n  RAG hit@5:    {rag_h['mean']*100:.1f}% (noise floor {nf:.1f}pp)")
    print(f"  DocGAC hit@5: {dg_h['mean']*100:.1f}% "
          f"(range {dg_h['range']*100:.1f}pp)")
    print(f"  delta:        {delta:+.1f}pp")
    if abs(delta) <= nf:
        print(f"  → within noise band — TIE")
    elif delta > 0:
        print(f"  → DocGAC AHEAD by {delta:.1f}pp")
    else:
        print(f"  → DocGAC BEHIND by {-delta:.1f}pp")

    rag_lats = [rag_results[q["id"]]["rag_total_ms"] for q in DOC_QUERIES]
    dg_lats = [dg_data["query_results"][q["id"]]["docgac_total_ms"]
                for q in DOC_QUERIES]
    print(f"  latency (avg): RAG {sum(rag_lats)/len(rag_lats):.2f}ms  "
          f"DocGAC {sum(dg_lats)/len(dg_lats):.2f}ms")

    return {
        "half_key": half_key,
        "n_docs": len(doc_set),
        "n_chunks": len(chunks),
        "doc_files": sorted(doc_set),
        "docgac_build_stats": dg_data["build_stats"],
        "docgac_query_routing": dg_data["query_routing"],
        "docgac_address_snapshot": dg_data["address_snapshot"],
        "rag_score": rag_score,
        "docgac_score": dg_score,
        "delta_hit_at_k_mean": (dg_h["mean"] - rag_h["mean"]),
        "rag_noise_floor_pp": nf,
        "rag_avg_latency_ms": sum(rag_lats) / len(rag_lats),
        "docgac_avg_latency_ms": sum(dg_lats) / len(dg_lats),
        "judge_calls_this_corpus": judge.calls,
        "judge_usd_this_corpus": judge.total_usd,
    }


def main():
    _require_queries()
    print("=" * 70)
    print("PHASE 4 — RAG vs DocGAC (zero-embedding architecture for prose)")
    print(f"  source: your document corpus, deterministic 50/50 split")
    print(f"  random_seed={RANDOM_SEED}, judge_repeats={JUDGE_REPEATS}, "
          f"timeout={PER_CALL_TIMEOUT_S}s")
    print("=" * 70)

    rubric_hash = hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()
    print(f"rubric SHA256: {rubric_hash}")

    # Load corpus chunks
    chunks = []
    with open(CHUNKS_JSONL) as f:
        for line in f:
            chunks.append(json.loads(line))
    docs_a, docs_b, chunks_a, chunks_b = split_corpus(chunks)
    print(f"\nSplit: {len(docs_a)} docs ({len(chunks_a)} chunks) train / "
          f"{len(docs_b)} docs ({len(chunks_b)} chunks) test")

    results_a = run_one_half("a", chunks_a, docs_a)
    results_b = run_one_half("b", chunks_b, docs_b)

    out = {
        "phase": 4,
        "meta": {
            "random_seed": RANDOM_SEED,
            "judge_repeats": JUDGE_REPEATS,
            "per_call_timeout_s": PER_CALL_TIMEOUT_S,
            "rubric_sha256": rubric_hash,
            "n_total_docs": len(docs_a) + len(docs_b),
            "n_queries": len(DOC_QUERIES),
        },
        "halves": [results_a, results_b],
    }
    RESULTS.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n=== PHASE 4 COMPLETE ===")
    print(f"  doc-a (train half): DocGAC vs RAG = "
          f"{results_a['delta_hit_at_k_mean']*100:+.1f}pp")
    print(f"  doc-b (test half):  DocGAC vs RAG = "
          f"{results_b['delta_hit_at_k_mean']*100:+.1f}pp")
    print(f"  results → {RESULTS}")


if __name__ == "__main__":
    main()
