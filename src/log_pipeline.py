"""Log-corpus pipeline: chunkify enterprise.log → build RAG + GAC → compare.

This is the §6 Case A target workload for GAC: low concept cardinality
(~30 log templates), high recurrence (each template fires many times).
We expect GAC to dominate on:
  - cost-curve flattening (LLM mints saturate quickly)
  - bounded scope (each query touches only its template + neighbours)
  - explainability (the route IS the template classification)
"""
from __future__ import annotations
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any

from rag import RAGSystem
from gac import GACSystem

LOG_FILE = Path(__file__).resolve().parent.parent / "logs/enterprise.log"
LOG_CHUNKS = Path(__file__).resolve().parent.parent / "data/log_chunks.jsonl"
LOG_RESULTS = Path(__file__).resolve().parent.parent / "data/log_results.json"
LOG_RAG_DIR = Path(__file__).resolve().parent.parent / "data/log_chroma"
LOG_GAC_INDEX = Path(__file__).resolve().parent.parent / "data/log_gac_index.json"
LOG_GAC_EMBS = Path(__file__).resolve().parent.parent / "data/log_chunk_embs.npy"

K = 5

# Operational queries — what an SRE / on-call engineer actually asks.
LOG_QUERIES = [
    # Simple lookups (template-targeted)
    {"id": "lq01", "kind": "simple", "query": "database connection pool exhausted",
     "expect": ["db-service", "Connection pool"]},
    {"id": "lq02", "kind": "simple", "query": "failed login attempts",
     "expect": ["auth-service", "Failed login"]},
    {"id": "lq03", "kind": "simple", "query": "payment gateway timeout",
     "expect": ["payment-service", "gateway timeout"]},
    {"id": "lq04", "kind": "simple", "query": "cache eviction",
     "expect": ["cache-service", "Eviction"]},
    {"id": "lq05", "kind": "simple", "query": "API gateway rate limit",
     "expect": ["api-gateway", "Rate limit"]},

    # Paraphrase pairs (low lexical overlap, same operational intent)
    {"id": "lq06a", "kind": "paraphrase", "pair_id": "LA",
     "query": "authentication failures",
     "expect": ["auth-service", "Failed login", "locked"]},
    {"id": "lq06b", "kind": "paraphrase", "pair_id": "LA",
     "query": "users who could not sign in",
     "expect": ["auth-service", "Failed login", "locked"]},

    {"id": "lq07a", "kind": "paraphrase", "pair_id": "LB",
     "query": "slow database queries",
     "expect": ["db-service", "Slow query", "timeout"]},
    {"id": "lq07b", "kind": "paraphrase", "pair_id": "LB",
     "query": "queries taking too long to complete",
     "expect": ["db-service", "Slow query", "timeout"]},

    {"id": "lq08a", "kind": "paraphrase", "pair_id": "LC",
     "query": "upstream service unavailable",
     "expect": ["api-gateway", "503", "Upstream"]},
    {"id": "lq08b", "kind": "paraphrase", "pair_id": "LC",
     "query": "503 errors from downstream services",
     "expect": ["api-gateway", "503", "Upstream"]},

    {"id": "lq09a", "kind": "paraphrase", "pair_id": "LD",
     "query": "email delivery problems",
     "expect": ["notification-svc", "SMTP", "Email"]},
    {"id": "lq09b", "kind": "paraphrase", "pair_id": "LD",
     "query": "mail sending failures",
     "expect": ["notification-svc", "SMTP"]},

    # Conceptual / cross-service
    {"id": "lq10", "kind": "conceptual",
     "query": "background job failures",
     "expect": ["worker-service", "Job", "failed"]},
    {"id": "lq11", "kind": "conceptual",
     "query": "what is happening with payments",
     "expect": ["payment-service"]},
]


# ---------- chunkify -----------------------------------------------------

def parse_log_line(line: str) -> Dict[str, str] | None:
    # 2026-05-24T07:23:11.479Z [auth-service      ] INFO  User u_kwaufg logged in...
    m = re.match(
        r"^(\S+)\s+\[([^\]]+?)\s*\]\s+(\S+)\s+(.+)$",
        line.strip(),
    )
    if not m:
        return None
    return {"ts": m.group(1), "svc": m.group(2).strip(),
            "level": m.group(3), "msg": m.group(4)}


def chunkify_logs(snapshot_lines: int | None = None) -> List[Dict[str, Any]]:
    """Each log line becomes one retrievable chunk."""
    chunks = []
    with open(LOG_FILE) as f:
        lines = f.readlines()
    if snapshot_lines:
        lines = lines[:snapshot_lines]
    for i, line in enumerate(lines):
        parsed = parse_log_line(line)
        if not parsed:
            continue
        # text we embed = service + level + message (timestamp omitted for semantic match)
        text = f"[{parsed['svc']}] {parsed['level']} {parsed['msg']}"
        chunks.append({
            "id": f"l{i:06d}",
            "doc": "enterprise.log",
            "section": f"line {i} @ {parsed['ts']}",
            "text": text,
            "meta": parsed,
        })
    return chunks


# ---------- pipeline -----------------------------------------------------

def build_systems(chunks):
    print(f"\n[pipeline] building RAG over {len(chunks)} log lines…")
    # use a separate Chroma dir so we don't clobber the doc-corpus index
    rag = RAGSystem(persist_dir=LOG_RAG_DIR)
    # nuke the doc-corpus collection ref by giving Chroma a fresh dir
    rag.build_index(chunks, rebuild=True)

    print(f"\n[pipeline] building GAC over {len(chunks)} log lines…")
    # GAC writes its cache to a fixed path; redirect via module globals
    import gac as gac_mod
    gac_mod.GAC_DATA = LOG_GAC_INDEX.parent
    # use a higher cluster count for ~30 templates
    gac_mod.N_ADDRESSES = 32
    gac = GACSystem()
    # Override cache paths to log-specific files
    orig_save = gac._save_cache
    orig_load = gac._load_cache

    def _save(cp, ep):
        orig_save(LOG_GAC_INDEX, LOG_GAC_EMBS)
    def _load(chunks, cp, ep):
        orig_load(chunks, LOG_GAC_INDEX, LOG_GAC_EMBS)
    gac._save_cache = _save
    gac._load_cache = _load
    # decide if we have a cached index
    rebuild = not (LOG_GAC_INDEX.exists() and LOG_GAC_EMBS.exists())
    if not rebuild:
        gac._load_cache(chunks, LOG_GAC_INDEX, LOG_GAC_EMBS)
        print(f"[pipeline] loaded cached GAC log index: {len(gac.addresses)} addresses")
    else:
        # mimic build_index but pin the cache path
        import numpy as np
        from sklearn.cluster import KMeans
        from gac import Cartographer, MAX_NODE_DEGREE
        gac.chunks = chunks
        gac.chunk_id_to_idx = {c["id"]: i for i, c in enumerate(chunks)}
        t0 = time.perf_counter()
        gac.chunk_embs = gac.emb_model.encode(
            [c["text"] for c in chunks], batch_size=64, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        gac.embed_time_s = time.perf_counter() - t0
        print(f"[pipeline] embedded {len(chunks)} lines in {gac.embed_time_s:.2f}s")

        t1 = time.perf_counter()
        n_clusters = min(gac_mod.N_ADDRESSES, max(5, len(chunks) // 80))
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(gac.chunk_embs)
        gac.cluster_time_s = time.perf_counter() - t1
        print(f"[pipeline] discovered {n_clusters} log-template clusters "
              f"in {gac.cluster_time_s:.2f}s")

        gac.cartographer = Cartographer()
        for cid in range(n_clusters):
            mem = np.where(labels == cid)[0]
            if len(mem) == 0:
                continue
            cembs = gac.chunk_embs[mem]
            centroid = cembs.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-9
            sims = cembs @ centroid
            order = np.argsort(-sims)
            rep = [chunks[mem[i]]["text"] for i in order[:5]]
            minted = gac.cartographer.mint_address(rep)
            addr = minted["address"]
            base = addr
            n = 1
            while addr in gac.addr_index:
                addr = f"{base}#{n}"; n += 1
            existing = [{"address": a["address"], "summary": a["summary"]}
                        for a in gac.addresses]
            idx = len(gac.addresses)
            gac.addresses.append({
                "address": addr, "summary": minted["summary"],
                "centroid": centroid.tolist(),
                "chunk_ids": [chunks[i]["id"] for i in mem.tolist()],
            })
            gac.addr_index[addr] = idx
            gac.graph.add_node(addr)
            edges = gac.cartographer.mint_edges(addr, minted["summary"], existing)
            for e in edges:
                gac.graph.add_edge(addr, e["to"], weight=e["weight"])
            print(f"  [{cid+1}/{n_clusters}] {addr}  ({len(mem)} lines, {len(edges)} edges)")

        gac._enforce_degree_bound()
        gac.centroids = np.vstack([a["centroid"] for a in gac.addresses])
        gac.warm_time_s = gac.cartographer.total_warm_time_s
        gac.index_build_time_s = (
            gac.embed_time_s + gac.cluster_time_s + gac.warm_time_s
        )
        gac.n_chunks_indexed = len(chunks)
        gac._save_cache(LOG_GAC_INDEX, LOG_GAC_EMBS)
        print(f"[pipeline] GAC: {len(gac.addresses)} addresses, "
              f"{gac.graph.number_of_edges()} edges, warm {gac.warm_time_s:.1f}s")

    return rag, gac


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb)) if (sa or sb) else 1.0


def hit_match(text: str, expect: List[str]) -> bool:
    """A hit matches if the chunk text contains ALL expected substrings (any subset)
    — more lenient: at least one substring must appear."""
    tl = text.lower()
    return any(e.lower() in tl for e in expect)


def evaluate(hits, expect):
    matches = [hit_match(h["text"], expect) for h in hits]
    first = next((i + 1 for i, m in enumerate(matches) if m), None)
    return {"hit@k": any(matches), "first_match_rank": first,
            "n_matches": sum(matches), "precision@k": sum(matches) / len(matches)}


def run():
    if not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0:
        print(f"ERROR: log file empty or missing: {LOG_FILE}")
        return

    chunks = chunkify_logs()
    print(f"[pipeline] {len(chunks)} log lines loaded")
    with open(LOG_CHUNKS, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    rag, gac = build_systems(chunks)

    print(f"\n[pipeline] running {len(LOG_QUERIES)} queries…")
    per_query = []
    for q in LOG_QUERIES:
        r_rag = rag.query(q["query"], k=K)
        r_gac = gac.query(q["query"], k=K)
        e_rag = evaluate(r_rag["hits"], q["expect"])
        e_gac = evaluate(r_gac["hits"], q["expect"])
        overlap = jaccard([h["id"] for h in r_rag["hits"]],
                          [h["id"] for h in r_gac["hits"]])
        per_query.append({
            "id": q["id"], "kind": q["kind"], "pair_id": q.get("pair_id"),
            "query": q["query"], "expect": q["expect"],
            "rag": {
                "hits": [{"id": h["id"], "score": h["score"],
                          "text": h["text"][:200]} for h in r_rag["hits"]],
                "candidate_set_size": r_rag["candidate_set_size"],
                "total_ms": r_rag["total_ms"],
                "eval": e_rag,
            },
            "gac": {
                "hits": [{"id": h["id"], "score": h["score"],
                          "text": h["text"][:200],
                          "via_address": h["via_address"]} for h in r_gac["hits"]],
                "routed_address": r_gac["routed_address"],
                "routing_confidence": r_gac["routing_confidence"],
                "expanded_neighbours": r_gac["expanded_neighbours"],
                "candidate_set_size": r_gac["candidate_set_size"],
                "corpus_size": r_gac["corpus_size"],
                "reduction_ratio": r_gac["reduction_ratio"],
                "total_ms": r_gac["total_ms"],
                "eval": e_gac,
            },
            "overlap_jaccard_at_k": overlap,
        })
        rh = "✓" if e_rag["hit@k"] else "✗"
        gh = "✓" if e_gac["hit@k"] else "✗"
        print(f"  [{q['id']}] RAG {rh} {r_rag['total_ms']:5.1f}ms  "
              f"GAC {gh} {r_gac['total_ms']:5.1f}ms  "
              f"reduction={r_gac['reduction_ratio']:5.1f}×  "
              f"prec@k RAG={e_rag['precision@k']:.2f} GAC={e_gac['precision@k']:.2f}  "
              f"→ {r_gac['routed_address']}")

    # paraphrase pair analysis
    pairs = []
    qmap = {p["id"]: p for p in per_query}
    pair_groups = {}
    for p in per_query:
        if p["pair_id"]:
            pair_groups.setdefault(p["pair_id"], []).append(p)
    for pid, group in pair_groups.items():
        if len(group) != 2:
            continue
        a, b = group
        pairs.append({
            "pair_id": pid,
            "queries": [a["query"], b["query"]],
            "rag_jaccard": jaccard([h["id"] for h in a["rag"]["hits"]],
                                    [h["id"] for h in b["rag"]["hits"]]),
            "gac_jaccard": jaccard([h["id"] for h in a["gac"]["hits"]],
                                    [h["id"] for h in b["gac"]["hits"]]),
            "gac_same_address": a["gac"]["routed_address"] == b["gac"]["routed_address"],
            "gac_one_hop": (
                b["gac"]["routed_address"] in a["gac"]["expanded_neighbours"] or
                a["gac"]["routed_address"] in b["gac"]["expanded_neighbours"]
            ),
            "gac_routed_addresses": [a["gac"]["routed_address"], b["gac"]["routed_address"]],
        })

    # aggregate
    def avg(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0

    rag_lat = [q["rag"]["total_ms"] for q in per_query]
    gac_lat = [q["gac"]["total_ms"] for q in per_query]

    agg = {
        "n_queries": len(per_query),
        "corpus_size": len(chunks),
        "k": K,
        "rag": {
            "hit_at_k_rate": avg([q["rag"]["eval"]["hit@k"] for q in per_query]),
            "avg_precision_at_k": avg([q["rag"]["eval"]["precision@k"] for q in per_query]),
            "avg_latency_ms": avg(rag_lat),
            "p95_latency_ms": sorted(rag_lat)[int(0.95 * len(rag_lat))],
            "avg_candidate_set_size": avg([q["rag"]["candidate_set_size"] for q in per_query]),
            "index_build_time_s": (rag.embed_time_s + rag.index_build_time_s),
        },
        "gac": {
            "hit_at_k_rate": avg([q["gac"]["eval"]["hit@k"] for q in per_query]),
            "avg_precision_at_k": avg([q["gac"]["eval"]["precision@k"] for q in per_query]),
            "avg_latency_ms": avg(gac_lat),
            "p95_latency_ms": sorted(gac_lat)[int(0.95 * len(gac_lat))],
            "avg_candidate_set_size": avg([q["gac"]["candidate_set_size"] for q in per_query]),
            "avg_reduction_ratio": avg([q["gac"]["reduction_ratio"] for q in per_query]),
            "index_build_time_s": gac.index_build_time_s,
            "warm_path_time_s": gac.warm_time_s,
            "n_addresses": len(gac.addresses),
            "n_edges": gac.graph.number_of_edges(),
            "mint_calls": gac.cartographer.mint_calls if gac.cartographer else 0,
            "edge_calls": gac.cartographer.edge_calls if gac.cartographer else 0,
        },
        "paraphrase_pairs": {
            "n_pairs": len(pairs),
            "rag_avg_jaccard": avg([p["rag_jaccard"] for p in pairs]),
            "gac_avg_jaccard": avg([p["gac_jaccard"] for p in pairs]),
            "gac_same_address_rate": avg([p["gac_same_address"] for p in pairs]),
            "gac_one_hop_recovery_rate": avg(
                [p["gac_one_hop"] and not p["gac_same_address"] for p in pairs]
            ),
            "gac_co_location_recall": avg(
                [p["gac_same_address"] or p["gac_one_hop"] for p in pairs]
            ),
        },
    }

    out = {"aggregate": agg, "per_query": per_query, "pairs": pairs}
    LOG_RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\n[pipeline] results → {LOG_RESULTS}")
    print(f"\n  RAG  hit@{K}={agg['rag']['hit_at_k_rate']*100:.0f}%  "
          f"avg lat={agg['rag']['avg_latency_ms']:.1f}ms")
    print(f"  GAC  hit@{K}={agg['gac']['hit_at_k_rate']*100:.0f}%  "
          f"avg lat={agg['gac']['avg_latency_ms']:.1f}ms  "
          f"({agg['gac']['n_addresses']} addrs, "
          f"avg reduction {agg['gac']['avg_reduction_ratio']:.0f}×)")
    print(f"  Paraphrase co-location recall (GAC): "
          f"{agg['paraphrase_pairs']['gac_co_location_recall']*100:.0f}%")


if __name__ == "__main__":
    run()
