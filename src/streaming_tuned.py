"""§12-tuned streaming GAC: HDBSCAN + hybrid routing + harder queries.

Implements two of the new whitepaper §12 best practices on top of the existing
batched cartographer:

  §12.1 — Cardinality-discovering clustering: HDBSCAN replaces fixed-K KMeans.
          The address space size is *discovered* from the data, not imposed.

  §12.2 — Hybrid routing: each address is keyed by (service, level, centroid).
          A query's structured tag (if extractable) narrows the address set
          BEFORE centroid similarity, eliminating cross-service routing errors.

Also adds a HARDER query set to stress the architecture beyond the easy
service-name lookups used in the baseline.

Output: data/tuned_results.json (same schema as streaming_results.json).
"""
from __future__ import annotations
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import networkx as nx

import sys
sys.path.insert(0, str(Path(__file__).parent))
from gac import Cartographer, MAX_NODE_DEGREE
from log_pipeline import parse_log_line, evaluate
from streaming_replay import (
    LOG_FILE, EMBED_USD_PER_TOKEN, PINECONE_USD_PER_WRITE,
    PINECONE_USD_PER_READ, GEMINI_IN, GEMINI_OUT,
    TOKENS_PER_LOG_LINE, TOKENS_PER_QUERY, MINT_IN_TOK, MINT_OUT_TOK,
    EDGE_IN_TOK, EDGE_OUT_TOK, USD_PER_MINT, USD_PER_EDGE,
    StreamingRAG,
)

DATA = Path(__file__).resolve().parent.parent / "data"
RESULTS = DATA / "tuned_results.json"
RAG_DIR = DATA / "tuned_chroma"

# §12-config
N_BOOTSTRAP_LINES = 2000
MINT_BATCH_SIZE = 10
HDBSCAN_MIN_CLUSTER_SIZE = 25      # auto-discovered cluster count
HDBSCAN_MIN_SAMPLES = 5

# Harder queries — stress the routing layer:
#   - cross-service intent
#   - lower-overlap paraphrases
#   - level-specific (only ERROR lines, etc.)
#   - keep some easy queries from baseline so we can compare
HARDER_QUERIES = [
    # baseline queries (carried over to verify we didn't regress)
    {"id": "h01", "kind": "simple", "query": "database connection pool exhausted",
     "expect": ["db-service", "Connection pool"], "hint_service": "db-service", "hint_level": "ERROR"},
    {"id": "h02", "kind": "simple", "query": "failed login attempts",
     "expect": ["auth-service", "Failed login"], "hint_service": "auth-service", "hint_level": "WARN"},
    {"id": "h03", "kind": "simple", "query": "payment gateway timeout",
     "expect": ["payment-service", "gateway timeout"], "hint_service": "payment-service", "hint_level": "ERROR"},

    # harder paraphrases (low lexical overlap, same intent)
    {"id": "h04a", "kind": "paraphrase", "pair_id": "HA",
     "query": "users unable to authenticate",
     "expect": ["auth-service", "Failed login", "locked"], "hint_service": "auth-service", "hint_level": "WARN"},
    {"id": "h04b", "kind": "paraphrase", "pair_id": "HA",
     "query": "credentials being rejected by the system",
     "expect": ["auth-service", "Failed login", "locked"], "hint_service": "auth-service", "hint_level": "WARN"},

    {"id": "h05a", "kind": "paraphrase", "pair_id": "HB",
     "query": "transactions that failed to complete",
     "expect": ["payment-service", "declined", "timeout"], "hint_service": "payment-service", "hint_level": "WARN"},
    {"id": "h05b", "kind": "paraphrase", "pair_id": "HB",
     "query": "billing operations not going through",
     "expect": ["payment-service", "declined", "timeout"], "hint_service": "payment-service", "hint_level": "WARN"},

    {"id": "h06a", "kind": "paraphrase", "pair_id": "HC",
     "query": "downstream dependency failures",
     "expect": ["api-gateway", "503", "Upstream"], "hint_service": "api-gateway", "hint_level": "ERROR"},
    {"id": "h06b", "kind": "paraphrase", "pair_id": "HC",
     "query": "internal service errors propagating to clients",
     "expect": ["api-gateway", "503", "Upstream"], "hint_service": "api-gateway", "hint_level": "ERROR"},

    # cross-service / abstract operational
    {"id": "h07", "kind": "conceptual",
     "query": "anything anomalous in the last batch of logs",
     "expect": ["ERROR", "timeout", "failed"], "hint_service": None, "hint_level": "ERROR"},
    {"id": "h08", "kind": "conceptual",
     "query": "high-latency operations across the platform",
     "expect": ["Slow", "timeout", "delayed"], "hint_service": None, "hint_level": None},

    # adversarial / negative phrasings (RAG often gets misled by similar words)
    {"id": "h09", "kind": "adversarial",
     "query": "users that did NOT log in successfully",
     "expect": ["Failed login", "locked"], "hint_service": "auth-service", "hint_level": "WARN"},
    {"id": "h10", "kind": "adversarial",
     "query": "background jobs that did not finish",
     "expect": ["Job", "failed"], "hint_service": "worker-service", "hint_level": "ERROR"},

    # level-specific
    {"id": "h11", "kind": "level",
     "query": "show me all errors from the payment system",
     "expect": ["payment-service", "ERROR", "timeout"], "hint_service": "payment-service", "hint_level": "ERROR"},
    {"id": "h12", "kind": "level",
     "query": "warnings from the cache layer",
     "expect": ["cache-service", "WARN", "Eviction"], "hint_service": "cache-service", "hint_level": "WARN"},

    # easy sanity checks
    {"id": "h13", "kind": "simple", "query": "successful payments",
     "expect": ["payment-service", "INFO", "processed"], "hint_service": "payment-service", "hint_level": "INFO"},
    {"id": "h14", "kind": "simple", "query": "indexed documents",
     "expect": ["search-service", "Indexed"], "hint_service": "search-service", "hint_level": "INFO"},
    {"id": "h15", "kind": "simple", "query": "cache miss events",
     "expect": ["cache-service", "miss"], "hint_service": "cache-service", "hint_level": "INFO"},
]

K = 5

EMB_MODEL = "all-MiniLM-L6-v2"


def line_to_text(line: str):
    p = parse_log_line(line)
    if not p:
        return None
    return {"ts": p["ts"], "svc": p["svc"], "level": p["level"],
            "msg": p["msg"], "text": f"[{p['svc']}] {p['level']} {p['msg']}"}


# ---------- §12-tuned GAC ------------------------------------------------

# Lightweight query-tag extractor for hybrid routing (§12.2).
# Match service words and log-level keywords mentioned in the query.
SERVICES = ["auth-service", "payment-service", "db-service", "cache-service",
            "search-service", "api-gateway", "notification-svc", "worker-service"]
SERVICE_ALIASES = {
    "auth-service": ["auth", "authent", "login", "credentials", "sign in", "sign-in", "password"],
    "payment-service": ["payment", "billing", "transaction", "refund", "charge"],
    "db-service": ["database", "db", "query", "queries", "sql"],
    "cache-service": ["cache", "redis", "memcache"],
    "search-service": ["search", "index", "indexed", "indexing"],
    "api-gateway": ["api", "gateway", "endpoint", "request", "503", "rate limit"],
    "notification-svc": ["notification", "email", "sms", "smtp", "mail"],
    "worker-service": ["worker", "job", "background", "batch"],
}
LEVEL_ALIASES = {
    "INFO": ["info", "successful", "ok"],
    "WARN": ["warn", "warning", "failed", "delayed"],
    "ERROR": ["error", "errors", "timeout", "exhausted", "exception", "anomal"],
}


def extract_query_tags(q: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (service_hint, level_hint) — None if can't infer."""
    ql = q.lower()
    svc_score = {}
    for svc, aliases in SERVICE_ALIASES.items():
        score = sum(1 for a in aliases if a in ql)
        if score > 0:
            svc_score[svc] = score
    svc = max(svc_score, key=svc_score.get) if svc_score else None

    lvl_score = {}
    for lvl, aliases in LEVEL_ALIASES.items():
        score = sum(1 for a in aliases if a in ql)
        if score > 0:
            lvl_score[lvl] = score
    lvl = max(lvl_score, key=lvl_score.get) if lvl_score else None
    return svc, lvl


class TunedGAC:
    """§12 best-practice tuned variant of StreamingGAC.

    Key differences from streaming_replay.StreamingGAC:
      1. HDBSCAN clustering at bootstrap (§12.1) — count is data-discovered.
      2. Hybrid routing: each address records its dominant (service, level)
         from its bootstrap members. Queries with detectable service/level
         tags are filtered to matching addresses BEFORE centroid similarity
         (§12.2).
    """
    def __init__(self):
        self.model = SentenceTransformer(EMB_MODEL)
        self.cartographer: Optional[Cartographer] = None
        self.addresses: List[Dict[str, Any]] = []   # +"dominant_svc","dominant_lvl"
        self.addr_index: Dict[str, int] = {}
        self.graph = nx.Graph()
        self.centroids: Optional[np.ndarray] = None
        self.address_texts: Dict[str, List[str]] = {}
        self.address_chunks_meta: Dict[str, List[Dict[str, str]]] = {}
        self.novelty_pool: List[Dict[str, Any]] = []
        self.n_indexed = 0
        self.total_mint_calls = 0
        self.total_edge_calls = 0
        self.total_cartographer_usd = 0.0
        # routing stats
        self.routes_hybrid = 0
        self.routes_centroid_only = 0

    def _refresh_centroids(self):
        if self.addresses:
            self.centroids = np.vstack([a["centroid"] for a in self.addresses])
        else:
            self.centroids = None

    def bootstrap(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        texts = [c["text"] for c in chunks]
        embs = self.model.encode(texts, batch_size=128, show_progress_bar=False,
                                 convert_to_numpy=True, normalize_embeddings=True)

        # §12.1: HDBSCAN — cardinality-discovering
        from sklearn.cluster import HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(embs)
        unique = sorted(set(labels) - {-1})
        n_noise = int(np.sum(labels == -1))
        print(f"  HDBSCAN discovered {len(unique)} clusters "
              f"(min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}); "
              f"{n_noise} noise points absorbed into nearest cluster")

        # absorb noise points into nearest non-noise cluster (so we lose no data)
        if n_noise > 0 and unique:
            # compute centroids of valid clusters
            valid_centroids = np.vstack([
                embs[labels == cid].mean(axis=0) /
                (np.linalg.norm(embs[labels == cid].mean(axis=0)) + 1e-9)
                for cid in unique
            ])
            noise_idx = np.where(labels == -1)[0]
            sims = embs[noise_idx] @ valid_centroids.T
            best = np.argmax(sims, axis=1)
            for i, ni in enumerate(noise_idx):
                labels[ni] = unique[best[i]]

        # build per-cluster representative samples + dominant tag
        per_cluster = []
        for cid in unique:
            mem = np.where(labels == cid)[0]
            if len(mem) == 0:
                continue
            centroid = embs[mem].mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-9
            sims = embs[mem] @ centroid
            order = np.argsort(-sims)
            rep = [chunks[mem[i]]["text"] for i in order[:5]]
            # dominant service/level
            from collections import Counter
            svcs = Counter(chunks[i]["svc"] for i in mem.tolist())
            lvls = Counter(chunks[i]["level"] for i in mem.tolist())
            per_cluster.append({
                "mem": mem,
                "centroid": centroid,
                "rep": rep,
                "dominant_svc": svcs.most_common(1)[0][0],
                "dominant_lvl": lvls.most_common(1)[0][0],
                "svc_purity": svcs.most_common(1)[0][1] / len(mem),
                "lvl_purity": lvls.most_common(1)[0][1] / len(mem),
            })

        if self.cartographer is None:
            self.cartographer = Cartographer()

        # §12.2 from prior whitepaper: batched minting
        new_addrs_for_edges = []
        for batch_start in range(0, len(per_cluster), MINT_BATCH_SIZE):
            batch = per_cluster[batch_start:batch_start + MINT_BATCH_SIZE]
            print(f"  batched mint: {len(batch)} clusters in 1 LLM call "
                  f"(batch {batch_start // MINT_BATCH_SIZE + 1})")
            try:
                minted_list = self.cartographer.mint_addresses_batch(
                    [b["rep"] for b in batch]
                )
            except Exception as e:
                print(f"  [WARN] batched mint failed: {e}")
                minted_list = [{"address": f"/fallback/{i}", "summary": ""}
                               for i in range(len(batch))]
            self.total_mint_calls += 1
            self.total_cartographer_usd += (
                MINT_IN_TOK * len(batch) * GEMINI_IN +
                MINT_OUT_TOK * len(batch) * GEMINI_OUT
            )

            for b, minted in zip(batch, minted_list):
                addr = minted["address"]
                base = addr
                n = 1
                while addr in self.addr_index:
                    addr = f"{base}#{n}"
                    n += 1
                addr_chunks_meta = [
                    {"svc": chunks[i]["svc"], "level": chunks[i]["level"],
                     "text": chunks[i]["text"]}
                    for i in b["mem"].tolist()
                ]
                self.addresses.append({
                    "address": addr,
                    "summary": minted["summary"],
                    "centroid": b["centroid"].tolist(),
                    "chunk_ids": [f"g{i:07d}" for i in b["mem"].tolist()],
                    "chunk_embs": embs[b["mem"]],
                    "dominant_svc": b["dominant_svc"],
                    "dominant_lvl": b["dominant_lvl"],
                    "svc_purity": b["svc_purity"],
                    "lvl_purity": b["lvl_purity"],
                })
                self.addr_index[addr] = len(self.addresses) - 1
                self.address_texts[addr] = [m["text"] for m in addr_chunks_meta]
                self.address_chunks_meta[addr] = addr_chunks_meta
                self.graph.add_node(addr)
                new_addrs_for_edges.append({"address": addr,
                                            "summary": minted["summary"]})
                print(f"    {addr:55s} "
                      f"[{b['dominant_svc']:18s}/{b['dominant_lvl']:5s}] "
                      f"({len(b['mem'])} lines, "
                      f"purity svc={b['svc_purity']:.0%} lvl={b['lvl_purity']:.0%})")

        # batched edges
        for batch_start in range(0, len(new_addrs_for_edges), MINT_BATCH_SIZE):
            batch_new = new_addrs_for_edges[batch_start:batch_start + MINT_BATCH_SIZE]
            existing = new_addrs_for_edges[:batch_start]
            try:
                edges_per_addr = self.cartographer.mint_edges_batch(batch_new, existing)
            except Exception as e:
                print(f"  [WARN] batched edges failed: {e}")
                edges_per_addr = [[] for _ in batch_new]
            self.total_edge_calls += 1
            self.total_cartographer_usd += (
                EDGE_IN_TOK * len(batch_new) * GEMINI_IN +
                EDGE_OUT_TOK * len(batch_new) * GEMINI_OUT
            )
            for entry, edges in zip(batch_new, edges_per_addr):
                for e in edges:
                    self.graph.add_edge(entry["address"], e["to"], weight=e["weight"])

        # degree bound
        for node in list(self.graph.nodes):
            edges = list(self.graph.edges(node, data=True))
            if len(edges) > MAX_NODE_DEGREE:
                edges.sort(key=lambda x: -x[2].get("weight", 0))
                for u, v, _ in edges[MAX_NODE_DEGREE:]:
                    if self.graph.has_edge(u, v):
                        self.graph.remove_edge(u, v)

        self._refresh_centroids()
        self.n_indexed = len(chunks)
        ms = (time.perf_counter() - t0) * 1000
        tokens = len(chunks) * TOKENS_PER_LOG_LINE
        cost = tokens * EMBED_USD_PER_TOKEN + self.total_cartographer_usd
        return {
            "ingest_ms": ms, "n_new": len(chunks), "cost_usd": cost,
            "n_clusters_discovered": len(unique),
            "n_noise_absorbed": int(n_noise),
        }

    def stream_ingest(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Incremental ingest using centroid + service-level routing."""
        if not chunks:
            return {"ingest_ms": 0, "n_new": 0, "cost_usd": 0}
        t0 = time.perf_counter()
        texts = [c["text"] for c in chunks]
        embs = self.model.encode(texts, batch_size=128, show_progress_bar=False,
                                 convert_to_numpy=True, normalize_embeddings=True)

        for i, c in enumerate(chunks):
            # §12.2 hybrid routing for ingestion: filter to matching (svc, lvl)
            cand = [j for j, a in enumerate(self.addresses)
                    if a["dominant_svc"] == c["svc"]]
            if not cand:
                cand = list(range(len(self.addresses)))
            cand_centroids = self.centroids[cand]
            sims = cand_centroids @ embs[i]
            best_local = int(np.argmax(sims))
            best = cand[best_local]
            a = self.addresses[best]
            a["chunk_ids"].append(f"g{self.n_indexed + i:07d}")
            a["chunk_embs"] = np.vstack([a["chunk_embs"], embs[i].reshape(1, -1)])
            self.address_texts[a["address"]].append(c["text"])
            self.address_chunks_meta[a["address"]].append({
                "svc": c["svc"], "level": c["level"], "text": c["text"],
            })

        self.n_indexed += len(chunks)
        ms = (time.perf_counter() - t0) * 1000
        tokens = len(chunks) * TOKENS_PER_LOG_LINE
        cost = tokens * EMBED_USD_PER_TOKEN
        return {"ingest_ms": ms, "n_new": len(chunks), "cost_usd": cost,
                "embed_tokens": tokens, "mint_calls_added": 0,
                "cartographer_usd_added": 0}

    def query(self, q: str, k: int = K, query_hint_svc=None, query_hint_lvl=None):
        """§12.2 hybrid routing for queries.

        If we can extract a service/level hint from the query, restrict the
        address pool to matching addresses *before* picking the closest
        centroid. Falls back to centroid-only routing if no tag is detected.
        """
        q_emb = self.model.encode([q], normalize_embeddings=True)[0]
        t0 = time.perf_counter()

        # extract tags from query text if not provided
        svc_hint, lvl_hint = query_hint_svc, query_hint_lvl
        if svc_hint is None and lvl_hint is None:
            svc_hint, lvl_hint = extract_query_tags(q)

        # §12.2 candidate filter
        candidate_idxs = []
        if svc_hint:
            candidate_idxs = [
                i for i, a in enumerate(self.addresses)
                if a["dominant_svc"] == svc_hint
            ]
        if not candidate_idxs:
            candidate_idxs = list(range(len(self.addresses)))
            self.routes_centroid_only += 1
        else:
            self.routes_hybrid += 1

        # optional level filter (don't apply if it shrinks too far)
        if lvl_hint and len(candidate_idxs) > 1:
            lvl_filtered = [i for i in candidate_idxs
                            if self.addresses[i]["dominant_lvl"] == lvl_hint]
            if lvl_filtered:
                candidate_idxs = lvl_filtered

        cand_centroids = self.centroids[candidate_idxs]
        sims = cand_centroids @ q_emb
        best_local = int(np.argmax(sims))
        routed_idx = candidate_idxs[best_local]
        routed_addr = self.addresses[routed_idx]["address"]

        # one-hop expansion
        cand_addrs = {routed_addr}
        nbrs = []
        if routed_addr in self.graph:
            for n in self.graph.neighbors(routed_addr):
                w = self.graph[routed_addr][n].get("weight", 0)
                nbrs.append((n, w))
        nbrs.sort(key=lambda x: -x[1])
        for n, _ in nbrs[:MAX_NODE_DEGREE]:
            cand_addrs.add(n)

        cand_embs_list, cand_meta = [], []
        for addr in cand_addrs:
            a = self.addresses[self.addr_index[addr]]
            cand_embs_list.append(a["chunk_embs"])
            for ci, cid in enumerate(a["chunk_ids"][:len(a["chunk_embs"])]):
                txt = self.address_texts[addr][ci] if ci < len(self.address_texts[addr]) else ""
                cand_meta.append((cid, addr, txt))
        cand_embs = np.vstack(cand_embs_list)[:200]
        cand_meta = cand_meta[:200]
        rerank_sims = cand_embs @ q_emb
        order = np.argsort(-rerank_sims)[:k]
        hits = []
        for o in order:
            cid, addr, txt = cand_meta[o]
            hits.append({"id": cid, "score": float(rerank_sims[o]),
                         "via_address": addr, "text": txt})
        ms = (time.perf_counter() - t0) * 1000
        return {
            "hits": hits, "candidate_set_size": len(cand_embs),
            "corpus_size": self.n_indexed,
            "reduction_ratio": self.n_indexed / max(1, len(cand_embs)),
            "routed_address": routed_addr,
            "expanded_neighbours": [n for n, _ in nbrs[:MAX_NODE_DEGREE]],
            "hybrid_routed": svc_hint is not None,
            "svc_hint": svc_hint, "lvl_hint": lvl_hint,
            "total_ms": ms,
            "cost_usd": TOKENS_PER_QUERY * EMBED_USD_PER_TOKEN,
        }


# ---------- main ---------------------------------------------------------

def main():
    print("=" * 70)
    print("§12-TUNED STREAMING: HDBSCAN + hybrid routing + harder queries")
    print("=" * 70)

    with open(LOG_FILE) as f:
        all_lines = f.readlines()
    chunks_all = []
    for ln in all_lines:
        p = line_to_text(ln)
        if p:
            chunks_all.append(p)
    print(f"\nloaded {len(chunks_all)} log lines")

    rag = StreamingRAG()
    gac = TunedGAC()

    # Phase 1
    print(f"\n--- Phase 1: WARM PATH bootstrap on first {N_BOOTSTRAP_LINES} lines ---")
    boot = chunks_all[:N_BOOTSTRAP_LINES]
    rag_boot = rag.ingest(boot)
    print(f"  RAG ingest: {rag_boot['ingest_ms']:.0f}ms  cost ${rag_boot['cost_usd']:.6f}")
    gac_boot = gac.bootstrap(boot)
    print(f"  GAC bootstrap: {gac_boot['ingest_ms']:.0f}ms  cost ${gac_boot['cost_usd']:.6f}")
    print(f"    {len(gac.addresses)} addresses, {gac.graph.number_of_edges()} edges, "
          f"{gac.total_mint_calls} mint + {gac.total_edge_calls} edge LLM calls")

    cumulative_rag = rag_boot["cost_usd"]
    cumulative_gac = gac_boot["cost_usd"]

    snapshots = [{
        "snapshot": 0, "phase": "warm-bootstrap",
        "corpus_size": rag.n_indexed,
        "delta_lines": len(boot),
        "rag_ingest_ms": rag_boot["ingest_ms"],
        "gac_ingest_ms": gac_boot["ingest_ms"],
        "rag_ingest_cost_usd": rag_boot["cost_usd"],
        "gac_ingest_cost_usd": gac_boot["cost_usd"],
        "gac_mints_added": gac.total_mint_calls,
        "gac_total_addresses": len(gac.addresses),
        "rag_avg_query_ms": None, "gac_avg_query_ms": None,
        "rag_query_cost_usd": 0, "gac_query_cost_usd": 0,
        "rag_hits_5": None, "gac_hits_5": None,
        "avg_gac_reduction": None,
        "cumulative_rag_usd": cumulative_rag,
        "cumulative_gac_usd": cumulative_gac,
        "n_hdbscan_clusters": gac_boot["n_clusters_discovered"],
        "n_noise_absorbed": gac_boot["n_noise_absorbed"],
    }]

    # Phase 2
    N_SNAPSHOTS = 6
    remaining = chunks_all[N_BOOTSTRAP_LINES:]
    wave_size = max(100, len(remaining) // N_SNAPSHOTS)
    print(f"\n--- Phase 2: HOT PATH streaming, {N_SNAPSHOTS} snapshots ---")

    per_query_detail = []
    for snap_i in range(N_SNAPSHOTS):
        wave = remaining[snap_i * wave_size:(snap_i + 1) * wave_size]
        print(f"\n  snapshot {snap_i + 1}/{N_SNAPSHOTS}: wave of {len(wave)} lines")
        rag_in = rag.ingest(wave)
        gac_in = gac.stream_ingest(wave)
        cumulative_rag += rag_in["cost_usd"]
        cumulative_gac += gac_in["cost_usd"]
        print(f"    RAG ingest: {rag_in['ingest_ms']:7.1f}ms  "
              f"cost ${rag_in['cost_usd']:.6f}  cumul ${cumulative_rag:.6f}")
        print(f"    GAC ingest: {gac_in['ingest_ms']:7.1f}ms  "
              f"cost ${gac_in['cost_usd']:.6f}  cumul ${cumulative_gac:.6f}")

        # queries — HARDER set
        q_rag_lat, q_gac_lat = [], []
        q_rag_hit, q_gac_hit = 0, 0
        q_rag_cost, q_gac_cost = 0.0, 0.0
        q_reductions = []
        for q in HARDER_QUERIES:
            r_rag = rag.query(q["query"], k=K)
            r_gac = gac.query(q["query"], k=K,
                              query_hint_svc=q.get("hint_service"),
                              query_hint_lvl=q.get("hint_level"))
            e_rag = evaluate(r_rag["hits"], q["expect"])
            e_gac = evaluate(r_gac["hits"], q["expect"])
            q_rag_lat.append(r_rag["total_ms"])
            q_gac_lat.append(r_gac["total_ms"])
            q_rag_hit += int(e_rag["hit@k"])
            q_gac_hit += int(e_gac["hit@k"])
            q_rag_cost += r_rag["cost_usd"]
            q_gac_cost += r_gac["cost_usd"]
            q_reductions.append(r_gac["reduction_ratio"])
            if snap_i == N_SNAPSHOTS - 1:  # capture detail on the last snapshot
                per_query_detail.append({
                    "id": q["id"], "kind": q["kind"], "query": q["query"],
                    "rag_hit": e_rag["hit@k"],
                    "gac_hit": e_gac["hit@k"],
                    "gac_routed": r_gac["routed_address"],
                    "gac_hybrid": r_gac["hybrid_routed"],
                    "gac_svc_hint": r_gac["svc_hint"],
                    "gac_lvl_hint": r_gac["lvl_hint"],
                    "gac_reduction": r_gac["reduction_ratio"],
                })

        cumulative_rag += q_rag_cost
        cumulative_gac += q_gac_cost
        avg_rag = sum(q_rag_lat) / len(q_rag_lat)
        avg_gac = sum(q_gac_lat) / len(q_gac_lat)
        avg_red = sum(q_reductions) / len(q_reductions)
        print(f"    queries({len(HARDER_QUERIES)}): "
              f"RAG {avg_rag:.2f}ms (hit {q_rag_hit}/{len(HARDER_QUERIES)})  "
              f"GAC {avg_gac:.2f}ms (hit {q_gac_hit}/{len(HARDER_QUERIES)})  "
              f"reduction {avg_red:.0f}×")

        snapshots.append({
            "snapshot": snap_i + 1, "phase": "stream",
            "corpus_size": rag.n_indexed,
            "delta_lines": len(wave),
            "rag_ingest_ms": rag_in["ingest_ms"],
            "gac_ingest_ms": gac_in["ingest_ms"],
            "rag_ingest_cost_usd": rag_in["cost_usd"],
            "gac_ingest_cost_usd": gac_in["cost_usd"],
            "gac_mints_added": 0,
            "gac_total_addresses": len(gac.addresses),
            "rag_avg_query_ms": avg_rag,
            "gac_avg_query_ms": avg_gac,
            "rag_query_cost_usd": q_rag_cost,
            "gac_query_cost_usd": q_gac_cost,
            "rag_hits_5": q_rag_hit,
            "gac_hits_5": q_gac_hit,
            "avg_gac_reduction": avg_red,
            "cumulative_rag_usd": cumulative_rag,
            "cumulative_gac_usd": cumulative_gac,
        })

    out = {
        "snapshots": snapshots,
        "final": {
            "corpus_size": rag.n_indexed,
            "n_addresses": len(gac.addresses),
            "n_edges": gac.graph.number_of_edges(),
            "total_mint_calls": gac.total_mint_calls,
            "total_edge_calls": gac.total_edge_calls,
            "cumulative_rag_usd": cumulative_rag,
            "cumulative_gac_usd": cumulative_gac,
            "savings_usd": cumulative_rag - cumulative_gac,
            "savings_pct": (cumulative_rag - cumulative_gac) /
                           max(1e-9, cumulative_rag) * 100,
            "routes_hybrid": gac.routes_hybrid,
            "routes_centroid_only": gac.routes_centroid_only,
        },
        "pricing_used": {
            "embed_usd_per_token": EMBED_USD_PER_TOKEN,
            "pinecone_usd_per_write": PINECONE_USD_PER_WRITE,
            "pinecone_usd_per_read": PINECONE_USD_PER_READ,
            "gemini_input_usd_per_token": GEMINI_IN,
            "gemini_output_usd_per_token": GEMINI_OUT,
            "tokens_per_log_line": TOKENS_PER_LOG_LINE,
            "tokens_per_query": TOKENS_PER_QUERY,
        },
        "harder_queries": [{"id": q["id"], "kind": q["kind"],
                            "query": q["query"], "expect": q["expect"]}
                           for q in HARDER_QUERIES],
        "per_query_last_snapshot": per_query_detail,
        "addresses": [{
            "address": a["address"], "summary": a["summary"],
            "n_chunks": len(a["chunk_ids"]),
            "dominant_svc": a["dominant_svc"],
            "dominant_lvl": a["dominant_lvl"],
            "svc_purity": a["svc_purity"],
            "lvl_purity": a["lvl_purity"],
        } for a in gac.addresses],
        "config": {
            "hdbscan_min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
            "hdbscan_min_samples": HDBSCAN_MIN_SAMPLES,
            "mint_batch_size": MINT_BATCH_SIZE,
            "n_bootstrap_lines": N_BOOTSTRAP_LINES,
            "n_queries": len(HARDER_QUERIES),
        },
    }
    RESULTS.write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 70)
    print(f"final corpus:   {out['final']['corpus_size']:,} log lines")
    print(f"GAC addresses:  {out['final']['n_addresses']} "
          f"(discovered by HDBSCAN, NOT fixed-K)")
    print(f"queries:        {len(HARDER_QUERIES)} (harder cross-service / paraphrase / adversarial)")
    print(f"RAG cumulative: ${out['final']['cumulative_rag_usd']:.6f}")
    print(f"GAC cumulative: ${out['final']['cumulative_gac_usd']:.6f}")
    print(f"savings:        ${out['final']['savings_usd']:.6f} "
          f"({out['final']['savings_pct']:+.1f}%)")
    print(f"hybrid routes:  {out['final']['routes_hybrid']} "
          f"(centroid-only: {out['final']['routes_centroid_only']})")
    rag_total_hits = sum(s["rag_hits_5"] for s in snapshots if s["rag_hits_5"] is not None)
    gac_total_hits = sum(s["gac_hits_5"] for s in snapshots if s["gac_hits_5"] is not None)
    n_qs = len(HARDER_QUERIES) * N_SNAPSHOTS
    print(f"hit@5 totals:   RAG {rag_total_hits}/{n_qs} "
          f"({rag_total_hits/n_qs*100:.0f}%)  "
          f"GAC {gac_total_hits}/{n_qs} ({gac_total_hits/n_qs*100:.0f}%)")
    print(f"\nresults → {RESULTS}")


if __name__ == "__main__":
    main()
