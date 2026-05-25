"""Deterministic streaming RAG vs GAC replay.

Replaces the live-scheduler streaming test with a deterministic replay over
the already-written enterprise.log. The cost model and measurements are
identical to what a live test would produce, but the test is reproducible,
fast, and can't hang on flaky API responses.

Design:
  Phase 1 (WARM PATH — amortized one-time cost):
    Bootstrap GAC's address space on the first N_BOOTSTRAP lines.
    All cartographer LLM calls happen here; ~12 mints + ~12 edges.
    RAG also indexes the same lines (embed + write).

  Phase 2 (HOT PATH — streaming snapshots):
    Replay the rest of the log file in S equal-sized waves.
    Each wave: incremental ingest into both systems, then run query set,
    record cost & latency for both, accumulate cumulative cost.
    GAC should produce 0 new mints in this phase (templates saturated).

The test is faster (~3 min total) and resilient — if a Gemini call fails
mid-bootstrap, we fall back to a deterministic addressing function so the
test always completes.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import networkx as nx

import sys
sys.path.insert(0, str(Path(__file__).parent))
from gac import Cartographer, MAX_NODE_DEGREE
from log_pipeline import parse_log_line, LOG_QUERIES, evaluate

LOG_FILE = Path(__file__).resolve().parent.parent / "logs/enterprise.log"
DATA = Path(__file__).resolve().parent.parent / "data"
RESULTS = DATA / "streaming_results.json"
RAG_DIR = DATA / "streaming_chroma"

# ---------- config -------------------------------------------------------
EMB_MODEL = "all-MiniLM-L6-v2"
N_BOOTSTRAP_LINES = 2000          # warm-path: build address space on first 2k lines
N_BOOTSTRAP_CLUSTERS = 30         # 30 clusters; §12.2 batching collapses LLM calls
N_SNAPSHOTS = 6                   # streaming waves after bootstrap
MINT_BATCH_SIZE = 10              # §12.2: mint this many clusters per LLM call
NOVELTY_THRESHOLD = 0.55           # below this similarity → "novel"
NOVELTY_BATCH = 80                 # mint when novelty pool reaches this
GAC_MAX_ADDRESSES = 60
WRITE_BATCH = 5000

# ---------- pricing (matches cost_model.py) -----------------------------
EMBED_USD_PER_TOKEN = 0.02 / 1_000_000
PINECONE_USD_PER_WRITE = 4.00 / 1_000_000
PINECONE_USD_PER_READ = 16.00 / 1_000_000
GEMINI_IN = 0.075 / 1_000_000
GEMINI_OUT = 0.30 / 1_000_000
MINT_IN_TOK, MINT_OUT_TOK = 600, 80
EDGE_IN_TOK, EDGE_OUT_TOK = 1500, 120
TOKENS_PER_LOG_LINE = 35
TOKENS_PER_QUERY = 12

USD_PER_MINT = MINT_IN_TOK * GEMINI_IN + MINT_OUT_TOK * GEMINI_OUT
USD_PER_EDGE = EDGE_IN_TOK * GEMINI_IN + EDGE_OUT_TOK * GEMINI_OUT


def line_to_text(line: str) -> Dict[str, Any] | None:
    p = parse_log_line(line)
    if not p:
        return None
    return {"ts": p["ts"], "svc": p["svc"], "level": p["level"],
            "msg": p["msg"], "text": f"[{p['svc']}] {p['level']} {p['msg']}"}


# ---------- streaming systems -------------------------------------------

class StreamingRAG:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(RAG_DIR))
        try:
            self.client.delete_collection("stream")
        except Exception:
            pass
        self.coll = self.client.create_collection(
            "stream", metadata={"hnsw:space": "cosine"}
        )
        self.model = SentenceTransformer(EMB_MODEL)
        self.n_indexed = 0

    def ingest(self, chunks: List[Dict[str, Any]]) -> Dict[str, float]:
        if not chunks:
            return {"ingest_ms": 0, "n_new": 0, "cost_usd": 0,
                    "embed_tokens": 0, "writes": 0}
        texts = [c["text"] for c in chunks]
        t0 = time.perf_counter()
        embs = self.model.encode(texts, batch_size=128, show_progress_bar=False,
                                 convert_to_numpy=True, normalize_embeddings=True)
        ids = [f"r{self.n_indexed + i:07d}" for i in range(len(chunks))]
        for off in range(0, len(chunks), WRITE_BATCH):
            end = min(off + WRITE_BATCH, len(chunks))
            self.coll.add(
                ids=ids[off:end],
                embeddings=embs[off:end].tolist(),
                documents=texts[off:end],
                metadatas=[{
                    "svc": c.get("svc") or "unknown",
                    "level": c.get("level") or "UNKNOWN",
                    "ts": c.get("ts", ""),
                } for c in chunks[off:end]],
            )
        ms = (time.perf_counter() - t0) * 1000
        tokens = len(chunks) * TOKENS_PER_LOG_LINE
        self.n_indexed += len(chunks)
        cost = tokens * EMBED_USD_PER_TOKEN + len(chunks) * PINECONE_USD_PER_WRITE
        return {"ingest_ms": ms, "n_new": len(chunks), "cost_usd": cost,
                "embed_tokens": tokens, "writes": len(chunks)}

    def query(self, q: str, k: int = 5):
        q_emb = self.model.encode([q], normalize_embeddings=True)[0]
        t0 = time.perf_counter()
        res = self.coll.query(query_embeddings=[q_emb.tolist()], n_results=k,
                              include=["documents", "metadatas", "distances"])
        ms = (time.perf_counter() - t0) * 1000
        hits = []
        for i in range(len(res["ids"][0])):
            hits.append({
                "id": res["ids"][0][i],
                "score": 1 - res["distances"][0][i],
                "text": res["documents"][0][i],
            })
        return {"hits": hits, "candidate_set_size": self.n_indexed,
                "total_ms": ms,
                "cost_usd": TOKENS_PER_QUERY * EMBED_USD_PER_TOKEN +
                            PINECONE_USD_PER_READ}


class StreamingGAC:
    def __init__(self):
        self.model = SentenceTransformer(EMB_MODEL)
        self.cartographer: Cartographer | None = None
        self.addresses: List[Dict[str, Any]] = []
        self.addr_index: Dict[str, int] = {}
        self.graph = nx.Graph()
        self.centroids: np.ndarray | None = None
        self.address_texts: Dict[str, List[str]] = {}  # for hit-detection
        self.novelty_pool: List[Dict[str, Any]] = []
        self.n_indexed = 0
        self.total_mint_calls = 0
        self.total_edge_calls = 0
        self.total_cartographer_usd = 0.0

    def _refresh_centroids(self):
        if self.addresses:
            self.centroids = np.vstack([a["centroid"] for a in self.addresses])
        else:
            self.centroids = None

    def _add_address(self, address: str, summary: str, centroid: np.ndarray,
                     chunks_for_addr: List[Dict[str, Any]], embs_for_addr: np.ndarray):
        # disambiguate
        base = address
        n = 1
        while address in self.addr_index:
            address = f"{base}#{n}"; n += 1
        self.addresses.append({
            "address": address,
            "summary": summary,
            "centroid": centroid.tolist(),
            "chunk_ids": [c.get("id", f"x{i}") for i, c in enumerate(chunks_for_addr)],
            "chunk_embs": embs_for_addr,
        })
        self.address_texts[address] = [c["text"] for c in chunks_for_addr]
        self.addr_index[address] = len(self.addresses) - 1
        self.graph.add_node(address)
        return address

    def _route(self, embs: np.ndarray):
        if self.centroids is None:
            return np.full(len(embs), -1), np.zeros(len(embs))
        sims = embs @ self.centroids.T
        best = np.argmax(sims, axis=1)
        return best, sims[np.arange(len(embs)), best]

    def _mint_with_cartographer(self, rep_texts: List[str]) -> Dict[str, str]:
        """Mint via Gemini, with safe fallback to deterministic addressing if it fails."""
        if self.cartographer is None:
            self.cartographer = Cartographer()
        try:
            return self.cartographer.mint_address(rep_texts)
        except Exception as e:
            print(f"  [WARN] cartographer mint failed ({e}); using fallback")
            # deterministic fallback: extract service/level hints from text
            from collections import Counter
            words = []
            for t in rep_texts[:3]:
                for w in t.replace("[", "").replace("]", "").lower().split():
                    if w.isalpha() and len(w) > 3:
                        words.append(w)
            top = [w for w, _ in Counter(words).most_common(3)]
            return {"address": "/log/" + "/".join(top), "summary": " ".join(rep_texts[0].split()[:15])}

    def _mint_edges(self, new_addr: str, summary: str) -> int:
        if self.cartographer is None or not self.addresses[:-1]:
            return 0
        existing = [{"address": a["address"], "summary": a["summary"]}
                    for a in self.addresses[:-1]]
        try:
            edges = self.cartographer.mint_edges(new_addr, summary, existing)
            for e in edges:
                self.graph.add_edge(new_addr, e["to"], weight=e["weight"])
            self.total_edge_calls += 1
            self.total_cartographer_usd += USD_PER_EDGE
            return len(edges)
        except Exception as e:
            print(f"  [WARN] cartographer edge failed ({e}); zero edges for {new_addr}")
            return 0

    def bootstrap(self, chunks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Phase 1: build the address space (warm path).

        §12.2 optimization: BATCH minting — send up to MINT_BATCH_SIZE clusters
        per LLM call rather than one at a time. Collapses ~20 round-trips into
        ~2-3 calls (~10× wall-clock speedup, ~same $ cost).
        """
        t0 = time.perf_counter()
        texts = [c["text"] for c in chunks]
        embs = self.model.encode(texts, batch_size=128, show_progress_bar=False,
                                 convert_to_numpy=True, normalize_embeddings=True)

        from sklearn.cluster import KMeans
        n_clusters = min(N_BOOTSTRAP_CLUSTERS, max(5, len(chunks) // 100))
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(embs)

        # Pre-compute per-cluster centroids + representative samples
        per_cluster = []
        for cid in range(n_clusters):
            mem = np.where(labels == cid)[0]
            if len(mem) == 0:
                continue
            centroid = embs[mem].mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-9
            sims = embs[mem] @ centroid
            order = np.argsort(-sims)
            rep = [chunks[mem[i]]["text"] for i in order[:5]]
            per_cluster.append({"mem": mem, "centroid": centroid, "rep": rep})

        if self.cartographer is None:
            self.cartographer = Cartographer()

        # §12.2: BATCHED minting
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
                print(f"  [WARN] batched mint failed ({e}); falling back to per-cluster")
                minted_list = [self._mint_with_cartographer(b["rep"]) for b in batch]
                self.total_mint_calls += len(batch)
                self.total_cartographer_usd += USD_PER_MINT * len(batch)
            else:
                self.total_mint_calls += 1   # one call covered the whole batch
                # token cost: input scales with batch size, output ~same per cluster
                self.total_cartographer_usd += (
                    600 * len(batch) * GEMINI_IN + 80 * len(batch) * GEMINI_OUT
                )

            for b, minted in zip(batch, minted_list):
                addr_chunks = [{**chunks[i], "id": f"g{i:07d}"}
                               for i in b["mem"].tolist()]
                addr = self._add_address(
                    minted["address"], minted["summary"], b["centroid"],
                    addr_chunks, embs[b["mem"]],
                )
                new_addrs_for_edges.append({
                    "address": addr, "summary": minted["summary"],
                })

        # §12.2: BATCHED edge minting — emit edges for all new addresses in one call
        if new_addrs_for_edges:
            # split into edge batches (similar logic — at most MINT_BATCH_SIZE per call)
            for batch_start in range(0, len(new_addrs_for_edges), MINT_BATCH_SIZE):
                batch_new = new_addrs_for_edges[batch_start:batch_start + MINT_BATCH_SIZE]
                # "existing" = everything minted BEFORE this batch
                existing_for_edges = new_addrs_for_edges[:batch_start]
                print(f"  batched edges: {len(batch_new)} new addresses in 1 LLM call "
                      f"(batch {batch_start // MINT_BATCH_SIZE + 1})")
                try:
                    edges_per_addr = self.cartographer.mint_edges_batch(
                        batch_new, existing_for_edges,
                    )
                except Exception as e:
                    print(f"  [WARN] batched edges failed ({e}); skipping edges for this batch")
                    edges_per_addr = [[] for _ in batch_new]
                else:
                    self.total_edge_calls += 1
                    self.total_cartographer_usd += (
                        1500 * len(batch_new) * GEMINI_IN + 120 * len(batch_new) * GEMINI_OUT
                    )
                for entry, edges in zip(batch_new, edges_per_addr):
                    for e in edges:
                        self.graph.add_edge(entry["address"], e["to"], weight=e["weight"])
                    print(f"    {entry['address']:60s} ({len(edges)} edges)")

        # enforce degree bound (§8)
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
        return {"ingest_ms": ms, "n_new": len(chunks), "cost_usd": cost,
                "embed_tokens": tokens, "mint_calls_added": self.total_mint_calls,
                "edge_calls_added": self.total_edge_calls,
                "cartographer_usd": self.total_cartographer_usd}

    def stream_ingest(self, chunks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Phase 2: route each new line to existing address; queue novel ones."""
        if not chunks:
            return {"ingest_ms": 0, "n_new": 0, "cost_usd": 0,
                    "embed_tokens": 0, "mint_calls_added": 0}
        t0 = time.perf_counter()
        cart_before = self.total_mint_calls
        cart_usd_before = self.total_cartographer_usd

        texts = [c["text"] for c in chunks]
        embs = self.model.encode(texts, batch_size=128, show_progress_bar=False,
                                 convert_to_numpy=True, normalize_embeddings=True)

        best, sims = self._route(embs)
        for i, c in enumerate(chunks):
            if sims[i] >= NOVELTY_THRESHOLD:
                ai = int(best[i])
                cid = f"g{self.n_indexed + i:07d}"
                self.addresses[ai]["chunk_ids"].append(cid)
                self.addresses[ai]["chunk_embs"] = np.vstack(
                    [self.addresses[ai]["chunk_embs"], embs[i].reshape(1, -1)]
                )
                self.address_texts[self.addresses[ai]["address"]].append(c["text"])
            else:
                self.novelty_pool.append({
                    "id": f"g{self.n_indexed + i:07d}",
                    "text": c["text"], "embedding": embs[i],
                })

        # mint new addresses from novelty pool if it grew enough
        while (len(self.novelty_pool) >= NOVELTY_BATCH and
               len(self.addresses) < GAC_MAX_ADDRESSES):
            pool_embs = np.array([p["embedding"] for p in self.novelty_pool])
            # cluster the pool into 1 — anything more cohesive than the threshold becomes an address
            centroid = pool_embs.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-9
            csim = pool_embs @ centroid
            order = np.argsort(-csim)
            rep = [self.novelty_pool[i]["text"] for i in order[:5]]
            minted = self._mint_with_cartographer(rep)
            self.total_mint_calls += 1
            self.total_cartographer_usd += USD_PER_MINT
            pool_chunks = [{"text": p["text"], "id": p["id"]}
                           for p in self.novelty_pool]
            addr = self._add_address(
                minted["address"], minted["summary"], centroid,
                pool_chunks, pool_embs,
            )
            n_edges = self._mint_edges(addr, minted["summary"])
            print(f"  stream: minted new address {addr} "
                  f"({len(pool_chunks)} lines, {n_edges} edges)")
            self.novelty_pool = []
            self._refresh_centroids()
            break  # one mint per snapshot keeps things bounded

        self.n_indexed += len(chunks)
        ms = (time.perf_counter() - t0) * 1000
        tokens = len(chunks) * TOKENS_PER_LOG_LINE
        mints_added = self.total_mint_calls - cart_before
        cart_added = self.total_cartographer_usd - cart_usd_before
        cost = tokens * EMBED_USD_PER_TOKEN + cart_added
        return {"ingest_ms": ms, "n_new": len(chunks), "cost_usd": cost,
                "embed_tokens": tokens, "mint_calls_added": mints_added,
                "cartographer_usd_added": cart_added}

    def query(self, q: str, k: int = 5):
        q_emb = self.model.encode([q], normalize_embeddings=True)[0]
        t0 = time.perf_counter()
        sims = self.centroids @ q_emb
        routed_idx = int(np.argmax(sims))
        routed_addr = self.addresses[routed_idx]["address"]
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
        return {"hits": hits, "candidate_set_size": len(cand_embs),
                "corpus_size": self.n_indexed,
                "reduction_ratio": self.n_indexed / max(1, len(cand_embs)),
                "routed_address": routed_addr,
                "expanded_neighbours": [n for n, _ in nbrs[:MAX_NODE_DEGREE]],
                "total_ms": ms,
                "cost_usd": TOKENS_PER_QUERY * EMBED_USD_PER_TOKEN}


# ---------- main ---------------------------------------------------------

def main():
    print("=" * 70)
    print("STREAMING REPLAY: RAG vs GAC over enterprise.log")
    print("=" * 70)

    with open(LOG_FILE) as f:
        all_lines = f.readlines()
    chunks_all = []
    for ln in all_lines:
        p = line_to_text(ln)
        if p:
            chunks_all.append(p)
    print(f"\nloaded {len(chunks_all)} log lines from {LOG_FILE.name}")

    rag = StreamingRAG()
    gac = StreamingGAC()

    # ---------- Phase 1: warm-path bootstrap ----------------------------
    print(f"\n--- Phase 1: WARM PATH bootstrap on first {N_BOOTSTRAP_LINES} lines ---")
    boot = chunks_all[:N_BOOTSTRAP_LINES]
    rag_boot = rag.ingest(boot)
    print(f"  RAG ingest: {rag_boot['ingest_ms']:.0f}ms  cost ${rag_boot['cost_usd']:.6f}")
    gac_boot = gac.bootstrap(boot)
    print(f"  GAC bootstrap: {gac_boot['ingest_ms']:.0f}ms  cost ${gac_boot['cost_usd']:.6f}")
    print(f"    {len(gac.addresses)} addresses, {gac.graph.number_of_edges()} edges, "
          f"{gac.total_mint_calls} mints + {gac.total_edge_calls} edge calls")

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
        "gac_mints_added": gac_boot["mint_calls_added"],
        "gac_total_addresses": len(gac.addresses),
        "rag_avg_query_ms": None, "gac_avg_query_ms": None,
        "rag_query_cost_usd": 0, "gac_query_cost_usd": 0,
        "rag_hits_5": None, "gac_hits_5": None,
        "avg_gac_reduction": None,
        "cumulative_rag_usd": cumulative_rag,
        "cumulative_gac_usd": cumulative_gac,
    }]

    # ---------- Phase 2: streaming snapshots ----------------------------
    print(f"\n--- Phase 2: HOT PATH streaming, {N_SNAPSHOTS} snapshots ---")
    remaining = chunks_all[N_BOOTSTRAP_LINES:]
    wave_size = max(100, len(remaining) // N_SNAPSHOTS)
    print(f"  wave size: {wave_size} lines × {N_SNAPSHOTS} snapshots = "
          f"{wave_size * N_SNAPSHOTS} lines total")

    for snap_i in range(N_SNAPSHOTS):
        wave = remaining[snap_i * wave_size:(snap_i + 1) * wave_size]
        print(f"\n  snapshot {snap_i + 1}/{N_SNAPSHOTS}: wave of {len(wave)} lines")
        rag_in = rag.ingest(wave)
        gac_in = gac.stream_ingest(wave)
        cumulative_rag += rag_in["cost_usd"]
        cumulative_gac += gac_in["cost_usd"]
        print(f"    RAG  ingest {rag_in['ingest_ms']:7.1f}ms  "
              f"cost ${rag_in['cost_usd']:.6f}  cumul ${cumulative_rag:.6f}")
        print(f"    GAC  ingest {gac_in['ingest_ms']:7.1f}ms  "
              f"cost ${gac_in['cost_usd']:.6f}  cumul ${cumulative_gac:.6f}  "
              f"(+{gac_in['mint_calls_added']} mints)")

        # queries
        q_rag_lat, q_gac_lat = [], []
        q_rag_hit, q_gac_hit = 0, 0
        q_rag_cost, q_gac_cost = 0.0, 0.0
        q_reductions = []
        for q in LOG_QUERIES:
            r_rag = rag.query(q["query"])
            r_gac = gac.query(q["query"])
            e_rag = evaluate(r_rag["hits"], q["expect"])
            e_gac = evaluate(r_gac["hits"], q["expect"])
            q_rag_lat.append(r_rag["total_ms"])
            q_gac_lat.append(r_gac["total_ms"])
            q_rag_hit += int(e_rag["hit@k"])
            q_gac_hit += int(e_gac["hit@k"])
            q_rag_cost += r_rag["cost_usd"]
            q_gac_cost += r_gac["cost_usd"]
            q_reductions.append(r_gac["reduction_ratio"])
        cumulative_rag += q_rag_cost
        cumulative_gac += q_gac_cost
        avg_rag = sum(q_rag_lat) / len(q_rag_lat)
        avg_gac = sum(q_gac_lat) / len(q_gac_lat)
        avg_red = sum(q_reductions) / len(q_reductions)
        print(f"    queries({len(LOG_QUERIES)}): RAG {avg_rag:.2f}ms (hit {q_rag_hit}/{len(LOG_QUERIES)})  "
              f"GAC {avg_gac:.2f}ms (hit {q_gac_hit}/{len(LOG_QUERIES)})  "
              f"reduction {avg_red:.0f}×")

        snapshots.append({
            "snapshot": snap_i + 1, "phase": "stream",
            "corpus_size": rag.n_indexed,
            "delta_lines": len(wave),
            "rag_ingest_ms": rag_in["ingest_ms"],
            "gac_ingest_ms": gac_in["ingest_ms"],
            "rag_ingest_cost_usd": rag_in["cost_usd"],
            "gac_ingest_cost_usd": gac_in["cost_usd"],
            "gac_mints_added": gac_in["mint_calls_added"],
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

    # ---------- summary -------------------------------------------------
    summary = {
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
        "addresses": [{
            "address": a["address"], "summary": a["summary"],
            "n_chunks": len(a["chunk_ids"]),
        } for a in gac.addresses],
    }
    RESULTS.write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 70)
    print(f"final corpus:   {summary['final']['corpus_size']:,} log lines")
    print(f"GAC addresses:  {summary['final']['n_addresses']} ({summary['final']['total_mint_calls']} mints)")
    print(f"RAG cumulative: ${summary['final']['cumulative_rag_usd']:.6f}")
    print(f"GAC cumulative: ${summary['final']['cumulative_gac_usd']:.6f}")
    print(f"savings:        ${summary['final']['savings_usd']:.6f} "
          f"({summary['final']['savings_pct']:+.1f}%)")
    print(f"\nresults → {RESULTS}")


if __name__ == "__main__":
    main()
