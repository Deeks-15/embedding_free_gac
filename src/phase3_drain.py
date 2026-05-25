"""Phase 3 driver — RAG vs DrainGAC head-to-head, embedding-free hot path.

Tests the authentic GAC architecture (deterministic template routing via
Drain3, BM25 fallback, ZERO embeddings + ZERO cosine on the hot path)
against the RAG baseline on both corpora.

Same locked methodology as Phase 1/2:
  - random_seed=42 pinned
  - judge temp=0
  - rubric SHA256 byte-identical (asserted each call)
  - SIGALRM-based 60s timeout per Gemini call (interrupts blocking I/O)
  - pooled-once judging (RAG ∪ DrainGAC top-5 per query)
  - 3 judge passes per pool → noise floor
  - per-(query, pass) checkpoint to disk → idempotent restart
  - GAC build cached per config

Two corpora in one driver pass:
  Realistic (a)  — logs/realistic.log
  Held-out (b)   — logs/realistic_b.log

For each corpus runs ONLY:
  - RAG baseline (same StreamingRAG as before)
  - DrainGAC

Output:
  data/phase3_results.json
  data/phase3_cache/realistic_a/  (built artefacts + judge labels for corpus a)
  data/phase3_cache/realistic_b/  (built artefacts + judge labels for corpus b)
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

import numpy as np

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

from realistic_chunker import chunk_file
from streaming_realistic import EVAL_QUERIES, JUDGE_PROMPT, Judge
from streaming_replay import StreamingRAG
from drain_gac import DrainGAC

DATA = Path(__file__).resolve().parent.parent / "data"
RESULTS = DATA / "phase3_results.json"

RANDOM_SEED = 42
K = 5
BATCH = 5000
JUDGE_REPEATS = 3
PER_CALL_TIMEOUT_S = 60
PER_CALL_RETRIES = 2
BOOTSTRAP_LINES = 1500   # same convention as TunedGAC pilots

CORPORA = [
    {"key": "a", "log": Path(__file__).resolve().parent.parent / "logs/realistic.log"},
    {"key": "b", "log": Path(__file__).resolve().parent.parent / "logs/realistic_b.log"},
]


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


def cartographer_with_timeout(callable_fn, *args, timeout_s=120, retries=1):
    """Wrap any Cartographer call with SIGALRM timeout. Returns None on failure."""
    attempt = 0
    while attempt <= retries:
        attempt += 1
        old = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout_s)
        try:
            result = callable_fn(*args)
            signal.alarm(0)
            return result
        except _CallTimeout:
            print(f"  [CARTOGRAPHER TIMEOUT] >{timeout_s}s, attempt {attempt}")
        except Exception as e:
            signal.alarm(0)
            print(f"  [CARTOGRAPHER ERROR] {e}, attempt {attempt}")
        finally:
            signal.signal(signal.SIGALRM, old)
        time.sleep(2 ** attempt)
    return None


def patch_cartographer_with_timeout():
    """Monkey-patch the Cartographer's batched methods so they have SIGALRM
    protection too — DrainGAC's bootstrap calls them and they'd otherwise
    hang the entire run on a stuck Gemini call (as the smoke test demonstrated)."""
    from gac import Cartographer
    if hasattr(Cartographer, "_patched"):
        return
    orig_mint = Cartographer.mint_addresses_batch
    orig_edge = Cartographer.mint_edges_batch

    def safe_mint(self, clusters):
        result = cartographer_with_timeout(orig_mint, self, clusters)
        if result is None:
            print("  [WARN] mint timed out — using fallback addresses")
            # deterministic fallback: name each cluster generically
            return [{"address": f"/fallback/cluster-{i}",
                     "summary": "fallback (cartographer timeout)"}
                    for i in range(len(clusters))]
        return result

    def safe_edge(self, new_addrs, existing):
        if not existing:
            return [[] for _ in new_addrs]
        result = cartographer_with_timeout(orig_edge, self, new_addrs, existing)
        if result is None:
            print("  [WARN] edges timed out — empty edges for this batch")
            return [[] for _ in new_addrs]
        return result

    Cartographer.mint_addresses_batch = safe_mint
    Cartographer.mint_edges_batch = safe_edge
    Cartographer._patched = True


# ---------- per-corpus pipeline -----------------------------------------

def run_one_corpus(corpus_key: str, log_path: Path):
    print(f"\n{'='*70}")
    print(f"CORPUS ({corpus_key}): {log_path.name}")
    print(f"{'='*70}\n")

    cache_root = DATA / "phase3_cache" / f"realistic_{corpus_key}"
    cache_root.mkdir(parents=True, exist_ok=True)
    judge_cache = cache_root / "judge"
    judge_cache.mkdir(exist_ok=True)
    rag_cache_f = cache_root / "rag_queries.json"
    drain_cache_f = cache_root / "drain_config.json"

    chunks = chunk_file(log_path)
    print(f"chunked {len(chunks):,} entries from {log_path.name}")

    # ---- RAG (built once per corpus, cached) ----
    if rag_cache_f.exists():
        print("[cache HIT] RAG queries")
        rag_results = json.loads(rag_cache_f.read_text())
        rag_build_time = 0
    else:
        print("building RAG + querying...")
        rag = StreamingRAG()
        rag_t0 = time.perf_counter()
        for off in range(0, len(chunks), BATCH):
            rag.ingest(chunks[off:off+BATCH])
        rag_build_time = time.perf_counter() - rag_t0
        print(f"  RAG built in {rag_build_time:.1f}s ({rag.n_indexed:,} entries)")
        rag_results = {}
        for q in EVAL_QUERIES:
            r = rag.query(q["query"], k=K)
            rag_results[q["id"]] = {
                "query": q["query"], "kind": q["kind"],
                "rag_hits": [(h["id"], h["text"][:1500], h.get("score"))
                              for h in r["hits"]],
                "rag_total_ms": r["total_ms"],
            }
        rag_results["_meta"] = {"rag_build_seconds": rag_build_time}
        rag_cache_f.write_text(json.dumps(rag_results, default=str))

    # ---- DrainGAC (built once per corpus, cached) ----
    if drain_cache_f.exists():
        print("[cache HIT] DrainGAC build")
        drain_data = json.loads(drain_cache_f.read_text())
    else:
        print("building DrainGAC...")
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        # skip_cartographer=True: name addresses deterministically from template
        # tokens (no LLM). For this experiment the LLM-minted names are decorative;
        # retrieval mechanics (template-id routing, BM25 fallback, recency rerank)
        # are unchanged. Trades pretty names for reliability when Gemini is
        # rate-limiting / hanging — which has happened repeatedly in this pilot.
        dgac = DrainGAC(random_seed=RANDOM_SEED, skip_cartographer=True)
        boot = chunks[:BOOTSTRAP_LINES]
        t0 = time.perf_counter()
        boot_stats = dgac.bootstrap(boot)
        remaining = chunks[BOOTSTRAP_LINES:]
        for off in range(0, len(remaining), BATCH):
            dgac.stream_ingest(remaining[off:off+BATCH])
        build_secs = time.perf_counter() - t0
        # drain pending novel one last time (in case some are leftover)
        dgac._drain_pending_novel()
        print(f"  DrainGAC built in {build_secs:.1f}s: "
              f"{len(dgac.addresses)} addresses, "
              f"{dgac.events_with_known_template} known-template events "
              f"(0 embeddings), "
              f"{dgac.events_with_novel_template} novel-template events "
              f"(triggered warm path)")

        drain_results = {}
        deterministic_count = 0
        fallback_count = 0
        empty_count = 0
        total_embed_calls = 0
        for q in EVAL_QUERIES:
            r = dgac.query(q["query"], k=K)
            if r["fallback_used"]:
                fallback_count += 1
                if r.get("fallback_kind") == "empty":
                    empty_count += 1
            else:
                deterministic_count += 1
            total_embed_calls += r.get("embed_calls_this_query", 0)
            drain_results[q["id"]] = {
                "drain_hits": [(h["id"], h["text"][:1500], h.get("score"),
                                 h.get("via_address", ""))
                               for h in r["hits"]],
                "drain_total_ms": r["total_ms"],
                "drain_routed_address": r["routed_address"],
                "drain_reduction": r["reduction_ratio"],
                "drain_fallback_used": r["fallback_used"],
                "drain_fallback_kind": r.get("fallback_kind", ""),
                "drain_embed_calls": r.get("embed_calls_this_query", 0),
            }
        print(f"  query routing: deterministic={deterministic_count}, "
              f"BM25-fallback={fallback_count} (of which empty={empty_count}), "
              f"total query-time embed calls = {total_embed_calls}")

        drain_data = {
            "build_stats": {
                "total_addresses": len(dgac.addresses),
                "total_mint_llm_calls": dgac.total_mint_calls,
                "total_edge_llm_calls": dgac.total_edge_calls,
                "total_cartographer_usd": dgac.total_cartographer_usd,
                "events_with_known_template": dgac.events_with_known_template,
                "events_with_novel_template": dgac.events_with_novel_template,
                "build_secs": build_secs,
                "n_addresses": len(dgac.addresses),
            },
            "query_results": drain_results,
            "query_routing": {
                "deterministic": deterministic_count,
                "fallback_bm25": fallback_count,
                "empty_no_match": empty_count,
                "total_query_time_embeds": total_embed_calls,
            },
            "address_snapshot": [{
                "address": a["address"], "summary": a["summary"],
                "template": a["template"],
                "n_chunks": len(a["chunk_ids"]),
                "dominant_svc": a["dominant_svc"],
                "dominant_lvl": a["dominant_lvl"],
                "svc_purity": a["svc_purity"],
                "lvl_purity": a["lvl_purity"],
            } for a in dgac.addresses],
        }
        drain_cache_f.write_text(json.dumps(drain_data, default=str))

    # ---- Judge ----
    rubric_hash = hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()
    judge = Judge()

    # Build pools per query: RAG top-5 ∪ DrainGAC top-5
    pools = {}
    for q in EVAL_QUERIES:
        qid = q["id"]
        pool = {}
        for cid, txt, *_ in rag_results[qid]["rag_hits"]:
            pool[cid] = {"id": cid, "text": txt, "from": ["rag"]}
        for cid, txt, *_ in drain_data["query_results"][qid]["drain_hits"]:
            if cid in pool:
                pool[cid]["from"].append("draingac")
            else:
                pool[cid] = {"id": cid, "text": txt, "from": ["draingac"]}
        pools[qid] = list(pool.values())
    sizes = [len(p) for p in pools.values()]
    print(f"  pool sizes: min={min(sizes)} max={max(sizes)} "
          f"avg={sum(sizes)/len(sizes):.1f}")

    total = len(EVAL_QUERIES) * JUDGE_REPEATS
    already = sum(1 for q in EVAL_QUERIES for p in range(JUDGE_REPEATS)
                  if (judge_cache / f"{q['id']}_p{p}.json").exists())
    print(f"  judge cache: {already}/{total} already done")

    done = already
    for q in EVAL_QUERIES:
        for pi in range(JUDGE_REPEATS):
            cf = judge_cache / f"{q['id']}_p{pi}.json"
            if cf.exists():
                continue
            actual_hash = hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()
            assert actual_hash == rubric_hash, "RUBRIC DRIFT!"
            labels = judge_with_timeout(judge, q["query"], pools[q["id"]])
            if labels is None:
                cf.write_text(json.dumps({
                    "qid": q["id"], "pass": pi, "timed_out": True,
                    "labels": None,
                }))
            else:
                cf.write_text(json.dumps({
                    "qid": q["id"], "pass": pi, "timed_out": False,
                    "labels": labels,
                }))
                done += 1
                if done % 5 == 0:
                    print(f"  [{done}/{total}] judge calls "
                          f"(~${judge.total_usd:.4f})")

    # ---- Score ----
    all_labels = {}
    for q in EVAL_QUERIES:
        all_labels[q["id"]] = []
        for pi in range(JUDGE_REPEATS):
            f = judge_cache / f"{q['id']}_p{pi}.json"
            data = json.loads(f.read_text())
            all_labels[q["id"]].append({} if data.get("timed_out") else data["labels"])

    def score_system(system: str, hits_field: str, hits_data: Dict):
        """system: 'rag' or 'draingac'. hits_data: rag_results or drain_data['query_results']."""
        pass_scores = []
        for pi in range(JUDGE_REPEATS):
            per_q = []
            for q in EVAL_QUERIES:
                qid = q["id"]
                labels = all_labels[qid][pi]
                top5 = [c for c, _, *_ in hits_data[qid][hits_field]]
                rel = [labels.get(c, False) for c in top5]
                per_q.append({
                    "qid": qid, "kind": q["kind"],
                    "hit_at_k": any(rel),
                    "precision_at_k": sum(rel) / max(1, len(rel)),
                    "top1_relevant": rel[0] if rel else False,
                })
            def avg(xs): return sum(xs)/max(1, len(xs))
            n = len(per_q)
            pass_agg = {
                "hit_at_k_rate": sum(q["hit_at_k"] for q in per_q) / n,
                "precision_at_k_avg": avg([q["precision_at_k"] for q in per_q]),
                "top1_relevant_rate": sum(q["top1_relevant"] for q in per_q) / n,
            }
            pass_by_kind = {}
            for kind in sorted({q["kind"] for q in per_q}):
                sub = [q for q in per_q if q["kind"] == kind]
                pass_by_kind[kind] = {
                    "n": len(sub),
                    "hit_at_k": sum(q["hit_at_k"] for q in sub) / len(sub),
                    "precision_at_k": avg([q["precision_at_k"] for q in sub]),
                }
            pass_scores.append({"aggregate": pass_agg, "by_kind": pass_by_kind})

        def stats(vs): return {
            "min": min(vs), "mean": sum(vs)/len(vs), "max": max(vs),
            "range": max(vs)-min(vs),
        }
        agg_s = {}
        for m in ("hit_at_k_rate", "precision_at_k_avg", "top1_relevant_rate"):
            agg_s[m] = stats([p["aggregate"][m] for p in pass_scores])
        bk_s = {}
        for kind in sorted(pass_scores[0]["by_kind"]):
            n = pass_scores[0]["by_kind"][kind]["n"]
            bk_s[kind] = {"n": n}
            for m in ("hit_at_k", "precision_at_k"):
                bk_s[kind][m] = stats([p["by_kind"][kind][m] for p in pass_scores])
        return {"aggregate_stats": agg_s, "by_kind_stats": bk_s}

    rag_score = score_system("rag", "rag_hits", rag_results)
    drain_score = score_system("draingac", "drain_hits", drain_data["query_results"])

    rag_hit_mean = rag_score["aggregate_stats"]["hit_at_k_rate"]["mean"]
    drain_hit_mean = drain_score["aggregate_stats"]["hit_at_k_rate"]["mean"]
    delta = drain_hit_mean - rag_hit_mean
    nf = rag_score["aggregate_stats"]["hit_at_k_rate"]["range"]

    print(f"\n  RAG hit@5:      {rag_hit_mean*100:.1f}% "
          f"(range {nf*100:.1f}pp = noise floor)")
    print(f"  DrainGAC hit@5: {drain_hit_mean*100:.1f}% "
          f"(range {drain_score['aggregate_stats']['hit_at_k_rate']['range']*100:.1f}pp)")
    print(f"  delta:          {delta*100:+.1f}pp")
    if abs(delta) <= nf:
        print(f"  → within RAG noise band ({nf*100:.1f}pp) — TIE")
    elif delta > 0:
        print(f"  → DrainGAC AHEAD by {delta*100:.1f}pp")
    else:
        print(f"  → DrainGAC BEHIND by {-delta*100:.1f}pp")

    # latency summary
    rag_lats = [rag_results[q["id"]]["rag_total_ms"] for q in EVAL_QUERIES]
    drain_lats = [drain_data["query_results"][q["id"]]["drain_total_ms"]
                   for q in EVAL_QUERIES]
    print(f"  latency (avg): RAG {sum(rag_lats)/len(rag_lats):.2f}ms  "
          f"DrainGAC {sum(drain_lats)/len(drain_lats):.2f}ms")

    return {
        "corpus_key": corpus_key,
        "n_corpus_entries": len(chunks),
        "drain_build_stats": drain_data["build_stats"],
        "drain_query_routing": drain_data["query_routing"],
        "drain_address_snapshot": drain_data["address_snapshot"],
        "rag_score": rag_score,
        "drain_score": drain_score,
        "delta_hit_at_k_mean": delta,
        "rag_noise_floor_pp": nf * 100,
        "rag_avg_latency_ms": sum(rag_lats) / len(rag_lats),
        "drain_avg_latency_ms": sum(drain_lats) / len(drain_lats),
        "judge_calls_this_corpus": judge.calls,
        "judge_usd_this_corpus": judge.total_usd,
    }


def main():
    print("=" * 70)
    print("PHASE 3 — RAG vs DrainGAC (authentic GAC, zero embeddings on hot path)")
    print(f"  corpora: realistic (a) and held-out (b)")
    print(f"  judge_repeats={JUDGE_REPEATS}, timeout={PER_CALL_TIMEOUT_S}s")
    print("=" * 70)

    patch_cartographer_with_timeout()
    print("Cartographer mint+edge calls now SIGALRM-protected.")

    rubric_hash = hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()
    print(f"rubric SHA256: {rubric_hash}")

    all_results = []
    for corpus in CORPORA:
        r = run_one_corpus(corpus["key"], corpus["log"])
        all_results.append(r)

    out = {
        "phase": 3,
        "meta": {
            "random_seed": RANDOM_SEED,
            "judge_repeats": JUDGE_REPEATS,
            "per_call_timeout_s": PER_CALL_TIMEOUT_S,
            "rubric_sha256": rubric_hash,
        },
        "corpora": all_results,
    }
    RESULTS.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n=== PHASE 3 COMPLETE ===")
    print(f"  results → {RESULTS}")
    print("\nGeneralization read:")
    a = all_results[0]
    b = all_results[1]
    print(f"  realistic (a): DrainGAC vs RAG = {a['delta_hit_at_k_mean']*100:+.1f}pp")
    print(f"  held-out (b):  DrainGAC vs RAG = {b['delta_hit_at_k_mean']*100:+.1f}pp")


if __name__ == "__main__":
    main()
