"""Realistic-corpus streaming RAG vs GAC + LLM-judge eval.

Uses the new mixed-format multi-line corpus from realistic_scheduler.py and
the multi-line-aware chunker from realistic_chunker.py. Otherwise the
architecture is the same as streaming_tuned.py: HDBSCAN + batched mint +
hybrid routing.

Then runs a 40+ query eval against both systems with LLM-judge ground truth
(chunk-level relevance, not bucket substring) — fixes the methodology
critique that bucket-level substring matching was entangled with GAC's
optimization.

Output:
  data/realistic_results.json — streaming metrics
  data/realistic_judged.json  — LLM-judge eval results
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from realistic_chunker import chunk_file, summary as chunker_summary
from streaming_replay import (
    EMBED_USD_PER_TOKEN, PINECONE_USD_PER_WRITE, PINECONE_USD_PER_READ,
    GEMINI_IN, GEMINI_OUT, TOKENS_PER_QUERY, StreamingRAG,
)
from streaming_tuned import TunedGAC

# load .env for Gemini
ENV_PATH = Path(__file__).parent.parent / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from google import genai
from google.genai import types as genai_types

LOG_FILE = Path(__file__).resolve().parent.parent / "logs/realistic.log"
DATA = Path(__file__).resolve().parent.parent / "data"
STREAM_RESULTS = DATA / "realistic_results.json"
JUDGE_RESULTS = DATA / "realistic_judged.json"

N_BOOTSTRAP_ENTRIES = 1500    # smaller bootstrap — bigger entries (multi-line)
N_SNAPSHOTS = 5
K = 5

GEMINI_MODEL = os.environ.get("GCP_MODEL", "gemini-2.0-flash-exp")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# ---------- Eval queries ----------------------------------------------
# 40+ queries across the realistic corpus's format mix, services, and intents.
# Includes simple lookups, paraphrases, adversarial negations, cross-format,
# and multi-line-targeting queries.

EVAL_QUERIES = [
    # === Java exceptions (multi-line ground truth) ===
    {"id": "q01", "kind": "multi_line_exc",
     "query": "null pointer exception in order processing",
     "expect_topic": "java NPE order"},
    {"id": "q02", "kind": "multi_line_exc",
     "query": "database query timeout in Java",
     "expect_topic": "java db query timeout"},

    # === Postgres (multi-line slow query / deadlock) ===
    {"id": "q03", "kind": "multi_line_db",
     "query": "slow postgres query with query plan",
     "expect_topic": "postgres slow query"},
    {"id": "q04", "kind": "multi_line_db",
     "query": "database deadlock detected",
     "expect_topic": "postgres deadlock"},

    # === Mobile crash (multi-line) ===
    {"id": "q05", "kind": "multi_line_mobile",
     "query": "mobile app crashes",
     "expect_topic": "mobile crash"},
    {"id": "q06", "kind": "multi_line_mobile",
     "query": "iOS or Android client errors",
     "expect_topic": "mobile crash"},

    # === Payment events (JSON) ===
    {"id": "q07", "kind": "simple",
     "query": "successful payment processed",
     "expect_topic": "checkout success"},
    {"id": "q08", "kind": "simple",
     "query": "payment declined",
     "expect_topic": "payment declined"},
    {"id": "q09", "kind": "paraphrase",
     "query": "credit card transactions rejected",
     "expect_topic": "payment declined"},

    # === Auth (JSON + audit + Java) ===
    {"id": "q10", "kind": "simple",
     "query": "user successfully authenticated",
     "expect_topic": "auth login success"},
    {"id": "q11", "kind": "simple",
     "query": "failed authentication attempts",
     "expect_topic": "auth failed"},
    {"id": "q12", "kind": "paraphrase",
     "query": "users who could not sign in",
     "expect_topic": "auth failed"},
    {"id": "q13", "kind": "adversarial",
     "query": "users that did NOT successfully log in",
     "expect_topic": "auth failed"},

    # === Inventory ===
    {"id": "q14", "kind": "simple",
     "query": "low stock warnings",
     "expect_topic": "inventory low"},
    {"id": "q15", "kind": "paraphrase",
     "query": "items running out in warehouse",
     "expect_topic": "inventory low"},

    # === Kubernetes ===
    {"id": "q16", "kind": "simple",
     "query": "kubernetes pod scheduled successfully",
     "expect_topic": "k8s pod scheduled"},
    {"id": "q17", "kind": "simple",
     "query": "container killed for exceeding memory",
     "expect_topic": "k8s OOM"},
    {"id": "q18", "kind": "paraphrase",
     "query": "OOM kills in production",
     "expect_topic": "k8s OOM"},

    # === Nginx / HTTP ===
    {"id": "q19", "kind": "simple",
     "query": "HTTP 5xx errors at the gateway",
     "expect_topic": "nginx 5xx"},
    {"id": "q20", "kind": "simple",
     "query": "search queries from product page",
     "expect_topic": "nginx search"},
    {"id": "q21", "kind": "simple",
     "query": "checkout API requests",
     "expect_topic": "nginx checkout"},

    # === CDN ===
    {"id": "q22", "kind": "simple",
     "query": "CDN cache misses to origin",
     "expect_topic": "cdn cache miss"},
    {"id": "q23", "kind": "simple",
     "query": "DDoS attack blocked at edge",
     "expect_topic": "cdn ddos"},

    # === Recommender ===
    {"id": "q24", "kind": "simple",
     "query": "product recommendations generated",
     "expect_topic": "recommender inference"},
    {"id": "q25", "kind": "simple",
     "query": "cold start fallback in recommendations",
     "expect_topic": "recommender cold start"},

    # === Worker ===
    {"id": "q26", "kind": "simple",
     "query": "transactional emails sent",
     "expect_topic": "worker email"},
    {"id": "q27", "kind": "simple",
     "query": "image resize workers",
     "expect_topic": "worker image resize"},

    # === Elasticsearch ===
    {"id": "q28", "kind": "simple",
     "query": "elasticsearch shard allocation problems",
     "expect_topic": "es unassigned"},

    # === Audit ===
    {"id": "q29", "kind": "simple",
     "query": "privilege escalation attempts",
     "expect_topic": "audit privilege"},
    {"id": "q30", "kind": "simple",
     "query": "data export events for compliance",
     "expect_topic": "audit data export"},

    # === Redis ===
    {"id": "q31", "kind": "simple",
     "query": "redis cache operations",
     "expect_topic": "redis"},

    # === Cross-format / cross-service abstract ===
    {"id": "q32", "kind": "abstract",
     "query": "anything anomalous in production traffic",
     "expect_topic": "any anomaly"},
    {"id": "q33", "kind": "abstract",
     "query": "errors that affect customer purchases",
     "expect_topic": "payment or checkout error"},
    {"id": "q34", "kind": "abstract",
     "query": "operations that took unusually long",
     "expect_topic": "slow query or timeout"},

    # === Long-tail / rare events ===
    {"id": "q35", "kind": "rare",
     "query": "TLS certificate expiring soon",
     "expect_topic": "rare certificate"},
    {"id": "q36", "kind": "rare",
     "query": "DNS resolution failures",
     "expect_topic": "rare dns"},
    {"id": "q37", "kind": "rare",
     "query": "GDPR data deletion completed",
     "expect_topic": "rare gdpr"},
    {"id": "q38", "kind": "rare",
     "query": "webhook delivery retries",
     "expect_topic": "rare webhook"},
    {"id": "q39", "kind": "rare",
     "query": "promo codes redeemed",
     "expect_topic": "rare promo"},

    # === Paraphrase pairs (low overlap, hard) ===
    {"id": "q40", "kind": "paraphrase_hard",
     "query": "backend service threw an exception while processing an order",
     "expect_topic": "java NPE order"},
    {"id": "q41", "kind": "paraphrase_hard",
     "query": "queries hitting the orders table that timed out",
     "expect_topic": "java db query timeout OR postgres slow query"},
    {"id": "q42", "kind": "paraphrase_hard",
     "query": "mobile users experiencing app instability",
     "expect_topic": "mobile crash"},
]


# ---------- LLM judge --------------------------------------------------

JUDGE_PROMPT = """You are evaluating relevance for a log-retrieval system.

QUERY: "{query}"

CANDIDATE LOG ENTRIES (each is one logical log entry, possibly multi-line, identified by id):
{candidates}

For each candidate, label whether it is RELEVANT to the query's intent. A log
entry is RELEVANT if it describes the kind of event the query is asking about.

Be strict:
- A log line about a successful login is NOT relevant to a query about failed logins,
  even if both mention "login".
- A query about "queries that timed out" is RELEVANT to entries about timeouts
  or query-cancel events, but NOT to entries about queries completing normally.
- A query about "anything anomalous" is RELEVANT to any ERROR or WARN entry,
  but NOT to routine INFO entries.

Return ONLY valid JSON:
{{"judgements": [{{"id": "<id>", "relevant": true|false}}, ...]}}
in the same order as the candidates."""


class Judge:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = GEMINI_MODEL
        self.calls = 0
        self.total_usd = 0.0

    def judge(self, query: str, candidates: List[Dict[str, Any]]) -> Dict[str, bool]:
        if not candidates:
            return {}
        cand_str = "\n".join(
            f"  [{c['id']}] {c['text'][:300].replace(chr(10), ' | ')}"
            for c in candidates
        )
        prompt = JUDGE_PROMPT.format(query=query, candidates=cand_str)
        try:
            resp = self.client.models.generate_content(
                model=self.model, contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0, response_mime_type="application/json",
                ),
            )
            self.calls += 1
            self.total_usd += (
                (len(prompt) / 4) * GEMINI_IN + (len(resp.text) / 4) * GEMINI_OUT
            )
            data = json.loads(resp.text)
            out = {}
            for j in data.get("judgements", []):
                if "id" in j and "relevant" in j:
                    out[j["id"]] = bool(j["relevant"])
            for c in candidates:
                out.setdefault(c["id"], False)
            return out
        except Exception as e:
            print(f"  [judge WARN] {e}")
            return {c["id"]: False for c in candidates}


# ---------- main ---------------------------------------------------------

def main():
    print("=" * 70)
    print("REALISTIC CORPUS STREAMING + LLM-JUDGE EVAL")
    print("=" * 70)

    # 1. Chunk the realistic log with multi-line awareness
    print(f"\n[1/4] Chunking {LOG_FILE.name} with multi-line aware chunker…")
    entries = chunk_file(LOG_FILE)
    stats = chunker_summary(entries)
    print(json.dumps(stats, indent=2))

    # The downstream systems expect chunks with `text` field. Add `svc` and
    # `level` so hybrid routing can use them.
    chunks = entries  # already in the right shape

    # 2. Build both systems
    print(f"\n[2/4] Building RAG (Chroma) and Tuned GAC (HDBSCAN + batched mint)…")
    rag = StreamingRAG()
    BATCH = 5000
    t0 = time.perf_counter()
    for off in range(0, len(chunks), BATCH):
        rag.ingest(chunks[off:off+BATCH])
    print(f"  RAG indexed {rag.n_indexed} entries in {time.perf_counter()-t0:.1f}s")

    gac = TunedGAC()
    t1 = time.perf_counter()
    print(f"  bootstrapping GAC on first {N_BOOTSTRAP_ENTRIES} entries...")
    boot = chunks[:N_BOOTSTRAP_ENTRIES]
    gac.bootstrap(boot)
    print(f"  streaming the remaining {len(chunks) - N_BOOTSTRAP_ENTRIES} entries...")
    remaining = chunks[N_BOOTSTRAP_ENTRIES:]
    for off in range(0, len(remaining), BATCH):
        gac.stream_ingest(remaining[off:off+BATCH])
    print(f"  GAC indexed {gac.n_indexed} entries across {len(gac.addresses)} addresses "
          f"in {time.perf_counter()-t1:.1f}s")
    print(f"  GAC LLM calls: {gac.total_mint_calls} mint + {gac.total_edge_calls} edge")

    # 3. Run all queries through both systems, pool candidates, judge
    print(f"\n[3/4] Running {len(EVAL_QUERIES)} eval queries through both systems...")
    judge = Judge()
    per_query = []
    for qi, q in enumerate(EVAL_QUERIES):
        r_rag = rag.query(q["query"], k=K)
        r_gac = gac.query(q["query"], k=K)

        # pool unique candidates
        pool = {}
        for h in r_rag["hits"]:
            pool[h["id"]] = {"id": h["id"], "text": h["text"], "from": ["rag"]}
        for h in r_gac["hits"]:
            if h["id"] in pool:
                pool[h["id"]]["from"].append("gac")
            else:
                pool[h["id"]] = {"id": h["id"], "text": h["text"], "from": ["gac"]}
        pool_list = list(pool.values())

        judgements = judge.judge(q["query"], pool_list)

        rag_ids = [h["id"] for h in r_rag["hits"]]
        gac_ids = [h["id"] for h in r_gac["hits"]]
        rag_rel = [judgements.get(i, False) for i in rag_ids]
        gac_rel = [judgements.get(i, False) for i in gac_ids]
        n_pool_relevant = sum(1 for v in judgements.values() if v)

        per_query.append({
            "qid": q["id"], "kind": q["kind"], "query": q["query"],
            "expect_topic": q["expect_topic"],
            "rag_hits": rag_ids,
            "gac_hits": gac_ids,
            "rag_top1_relevant": rag_rel[0] if rag_rel else False,
            "gac_top1_relevant": gac_rel[0] if gac_rel else False,
            "rag_precision_at_k": sum(rag_rel) / max(1, len(rag_rel)),
            "gac_precision_at_k": sum(gac_rel) / max(1, len(gac_rel)),
            "rag_hit_at_k": any(rag_rel),
            "gac_hit_at_k": any(gac_rel),
            "rag_recall_in_pool": sum(rag_rel) / max(1, n_pool_relevant),
            "gac_recall_in_pool": sum(gac_rel) / max(1, n_pool_relevant),
            "n_pool_candidates": len(pool_list),
            "n_pool_relevant": n_pool_relevant,
            "rag_total_ms": r_rag["total_ms"],
            "gac_total_ms": r_gac["total_ms"],
            "gac_routed_address": r_gac["routed_address"],
            "gac_reduction": r_gac["reduction_ratio"],
        })
        if (qi + 1) % 5 == 0:
            print(f"  judged {qi+1}/{len(EVAL_QUERIES)} (judge calls: {judge.calls}, "
                  f"~${judge.total_usd:.4f})")

    # 4. Aggregate
    def avg(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / max(1, len(xs))

    n_q = len(per_query)
    agg = {
        "n_queries": n_q,
        "n_judge_calls": judge.calls,
        "judge_total_usd_estimate": judge.total_usd,
        "rag": {
            "hit_at_k_rate": sum(q["rag_hit_at_k"] for q in per_query) / n_q,
            "precision_at_k_avg": avg([q["rag_precision_at_k"] for q in per_query]),
            "top1_relevant_rate": sum(q["rag_top1_relevant"] for q in per_query) / n_q,
            "recall_in_pool_avg": avg([q["rag_recall_in_pool"] for q in per_query]),
            "avg_latency_ms": avg([q["rag_total_ms"] for q in per_query]),
        },
        "gac": {
            "hit_at_k_rate": sum(q["gac_hit_at_k"] for q in per_query) / n_q,
            "precision_at_k_avg": avg([q["gac_precision_at_k"] for q in per_query]),
            "top1_relevant_rate": sum(q["gac_top1_relevant"] for q in per_query) / n_q,
            "recall_in_pool_avg": avg([q["gac_recall_in_pool"] for q in per_query]),
            "avg_latency_ms": avg([q["gac_total_ms"] for q in per_query]),
            "n_addresses": len(gac.addresses),
            "total_mint_calls": gac.total_mint_calls,
            "total_edge_calls": gac.total_edge_calls,
            "avg_reduction": avg([q["gac_reduction"] for q in per_query]),
        },
        "by_kind": {},
    }
    kinds = sorted({q["kind"] for q in per_query})
    for kind in kinds:
        sub = [q for q in per_query if q["kind"] == kind]
        agg["by_kind"][kind] = {
            "n": len(sub),
            "rag_hit_at_k": sum(q["rag_hit_at_k"] for q in sub) / len(sub),
            "gac_hit_at_k": sum(q["gac_hit_at_k"] for q in sub) / len(sub),
            "rag_precision_at_k": avg([q["rag_precision_at_k"] for q in sub]),
            "gac_precision_at_k": avg([q["gac_precision_at_k"] for q in sub]),
        }

    JUDGE_RESULTS.write_text(json.dumps({
        "aggregate": agg, "per_query": per_query,
        "corpus_stats": stats,
        "addresses": [{
            "address": a["address"], "summary": a["summary"],
            "n_chunks": len(a["chunk_ids"]),
            "dominant_svc": a["dominant_svc"],
            "dominant_lvl": a["dominant_lvl"],
            "svc_purity": a["svc_purity"], "lvl_purity": a["lvl_purity"],
        } for a in gac.addresses],
    }, indent=2))

    print("\n" + "=" * 70)
    print("REALISTIC + LLM-JUDGE SUMMARY")
    print("=" * 70)
    print(f"  corpus:     {stats['n_entries']:,} logical entries "
          f"({stats['n_raw_lines']:,} raw lines, "
          f"{stats['n_multiline_entries']:,} multi-line, "
          f"{stats['n_malformed_entries']:,} malformed)")
    print(f"  GAC:        {len(gac.addresses)} addresses, "
          f"{gac.total_mint_calls} mint + {gac.total_edge_calls} edge LLM calls")
    print(f"  judged:     {judge.calls} judge calls "
          f"(~${judge.total_usd:.4f} for Gemini Flash)")
    print()
    print(f"  RAG  hit@5 {agg['rag']['hit_at_k_rate']*100:5.1f}%  "
          f"prec@5 {agg['rag']['precision_at_k_avg']*100:5.1f}%  "
          f"top1-rel {agg['rag']['top1_relevant_rate']*100:5.1f}%  "
          f"recall-in-pool {agg['rag']['recall_in_pool_avg']*100:5.1f}%")
    print(f"  GAC  hit@5 {agg['gac']['hit_at_k_rate']*100:5.1f}%  "
          f"prec@5 {agg['gac']['precision_at_k_avg']*100:5.1f}%  "
          f"top1-rel {agg['gac']['top1_relevant_rate']*100:5.1f}%  "
          f"recall-in-pool {agg['gac']['recall_in_pool_avg']*100:5.1f}%")
    print()
    print(f"  by kind:")
    for k, m in agg["by_kind"].items():
        print(f"    {k:22s}  n={m['n']:2d}  "
              f"RAG hit={m['rag_hit_at_k']*100:5.1f}% prec={m['rag_precision_at_k']*100:5.1f}%  "
              f"GAC hit={m['gac_hit_at_k']*100:5.1f}% prec={m['gac_precision_at_k']*100:5.1f}%")
    print(f"\n  results → {JUDGE_RESULTS}")


if __name__ == "__main__":
    main()
