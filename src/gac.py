"""GAC: Generative Address Convergence.

Implements the three-plane architecture from the whitepaper:

  Warm path (rare, amortized): LLM (Gemini) mints a canonical address for each
    discovered concept cluster AND emits weighted adjacency edges to nearby
    addresses. This is the only place the LLM runs.

  Hot path (per event, CPU-only): incoming items route to their nearest address
    centroid by closed-set similarity. No LLM call. No whole-corpus comparison.

  Cold path (periodic batch): not exercised in this single-shot pilot, but the
    code paths are stubbed for future fragmentation merging.

Retrieval:
  query -> hot-path route to single address -> candidate set = address chunks
        + chunks at one-hop graph neighbours (bounded by degree/depth/ceiling)
        -> optional scoped refinement (cosine similarity rerank inside the set).
"""
from __future__ import annotations
import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx

# load .env
ENV_PATH = Path(__file__).parent.parent / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from google import genai  # noqa: E402
from google.genai import types as genai_types  # noqa: E402

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data/chunks.jsonl"
GAC_DATA = Path(__file__).resolve().parent.parent / "data"
EMB_MODEL = "all-MiniLM-L6-v2"

# Bounded-graph invariants from whitepaper §8.
MAX_NODE_DEGREE = 5
TRAVERSAL_DEPTH = 1
CANDIDATE_CEILING = 200

# Address-space sizing: target ~30-50 concepts for a 575-chunk corpus
# (~12-20 chunks per address — well within the candidate ceiling).
N_ADDRESSES = 40

GEMINI_MODEL = os.environ.get("GCP_MODEL", "gemini-3-flash-preview")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# ---------- Cartographer (Gemini, warm path) -----------------------------

class Cartographer:
    """LLM that mints canonical addresses and adjacency edges. Warm path only.

    Stats are recorded so we can demonstrate the §5 economic claim:
    the LLM does not see per-event traffic.
    """

    MINT_PROMPT = """You are a CARTOGRAPHER for a discrete retrieval system.

Given a cluster of related text passages from a corpus of GenAI / consulting
material, your job is ONCE to mint a canonical, hierarchical, slash-separated
"address" string that names the concept this cluster is about. Addresses are
the routing primitive — every future query about this concept must collapse
to this exact symbol.

Rules for the address:
- Format: /domain/subdomain/concept (2-4 segments, lowercase, hyphen-separated)
- Specific enough to be unambiguous, general enough to absorb paraphrases
- Examples:
    /offering/velocity-ai/sdlc
    /case-study/finance/fraud-detection
    /platform/accelerator/prompt-advisor

Also produce a 1-sentence concept summary (what queries should route here).

Return ONLY valid JSON: {"address": "/...", "summary": "..."}"""

    EDGE_PROMPT = """You are a CARTOGRAPHER for a discrete retrieval system.

You just minted the address {new_addr} with summary:
  "{new_summary}"

Below are existing addresses in the address space (address + summary).

Emit weighted edges to addresses that are SEMANTICALLY ADJACENT — i.e. a query
intended for {new_addr} could plausibly land on the neighbour by paraphrase
("payment failure" vs "billing rejection"). Weight 0.0-1.0.

CRITICAL:
- Return AT MOST {max_edges} edges, only the strongest ones.
- Do NOT emit edges to addresses that are merely topically related but
  semantically distinct (e.g. /offering/* vs /case-study/*).
- If nothing qualifies, return empty list.

Existing addresses:
{existing}

Return ONLY valid JSON: {{"edges": [{{"to": "/...", "weight": 0.0-1.0}}, ...]}}"""

    def __init__(self, model: str = GEMINI_MODEL):
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = model
        self.mint_calls = 0
        self.edge_calls = 0
        self.total_warm_time_s = 0.0

    def _call(self, prompt: str) -> str:
        t0 = time.perf_counter()
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        self.total_warm_time_s += time.perf_counter() - t0
        return resp.text

    def mint_address(self, cluster_samples: List[str]) -> Dict[str, str]:
        joined = "\n\n---\n\n".join(s[:600] for s in cluster_samples[:5])
        prompt = self.MINT_PROMPT + f"\n\nCLUSTER PASSAGES:\n{joined}\n\nReturn JSON now:"
        raw = self._call(prompt)
        self.mint_calls += 1
        try:
            data = json.loads(raw)
            addr = data.get("address", "").strip()
            if not addr.startswith("/"):
                addr = "/" + addr
            return {"address": addr, "summary": data.get("summary", "")}
        except Exception:
            # fallback: derive from TF-IDF top terms (rare; keeps the system robust)
            tfidf = TfidfVectorizer(stop_words="english", max_features=5)
            tfidf.fit(cluster_samples)
            terms = tfidf.get_feature_names_out()
            return {"address": "/unknown/" + "-".join(terms[:3]), "summary": "fallback"}

    # ------- §12.2 batched variants (mint many clusters per LLM call) -----

    MINT_BATCH_PROMPT = """You are a CARTOGRAPHER for a discrete retrieval system.

You will be given N CLUSTERS of related passages. For EACH cluster, mint a
canonical hierarchical "address" string and a 1-sentence summary that names
the concept. Each address routes all paraphrases of that concept to itself.

Rules for each address:
- Format: /domain/subdomain/concept (2-4 segments, lowercase, hyphen-separated)
- Specific enough to be unambiguous, general enough to absorb paraphrases
- Examples: /offering/velocity-ai/sdlc, /security/auth/failed-login

CRITICAL: Return EXACTLY one JSON object per cluster in a JSON array, in the
same order as the input clusters.

CLUSTERS:
{clusters}

Return ONLY a valid JSON array of objects, one per cluster, like:
[{{"address": "/.../...", "summary": "..."}}, {{"address": "/.../...", "summary": "..."}}]"""

    EDGE_BATCH_PROMPT = """You are a CARTOGRAPHER for a discrete retrieval system.

You just minted these NEW addresses (each with its summary). For EACH new
address, emit weighted edges to addresses in the EXISTING address space that
are SEMANTICALLY ADJACENT — i.e. a query intended for the new address could
plausibly land on the neighbour by paraphrase.

Weight 0.0-1.0. At MOST {max_edges} edges per new address. Do NOT emit edges
to merely topically-related addresses; only true paraphrase neighbours.

NEW ADDRESSES (in order):
{new_addrs}

EXISTING ADDRESSES:
{existing}

Return ONLY a valid JSON array, in the same order as the new addresses, like:
[
  {{"address": "/new/addr/1", "edges": [{{"to": "/existing/...", "weight": 0.0-1.0}}, ...]}},
  {{"address": "/new/addr/2", "edges": []}}
]"""

    def mint_addresses_batch(self, clusters: List[List[str]]) -> List[Dict[str, str]]:
        """Mint addresses for N clusters in a SINGLE LLM call. Implements §12.2."""
        if not clusters:
            return []
        # build prompt: each cluster gets a numbered block with up to 5 samples
        cluster_blocks = []
        for i, samples in enumerate(clusters, 1):
            block = f"\n=== Cluster {i} ===\n"
            for s in samples[:4]:
                block += s[:400].replace("\n", " ") + "\n---\n"
            cluster_blocks.append(block)
        prompt = self.MINT_BATCH_PROMPT.format(clusters="\n".join(cluster_blocks))
        raw = self._call(prompt)
        self.mint_calls += 1
        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("expected JSON array")
            out = []
            for i, item in enumerate(data[:len(clusters)]):
                addr = item.get("address", "").strip()
                if not addr.startswith("/"):
                    addr = "/" + addr
                out.append({"address": addr, "summary": item.get("summary", "")})
            # pad if model returned fewer
            while len(out) < len(clusters):
                out.append({"address": f"/fallback/cluster-{len(out)}",
                            "summary": "fallback (model returned fewer items)"})
            return out
        except Exception:
            # fallback: per-cluster TF-IDF naming
            from sklearn.feature_extraction.text import TfidfVectorizer
            out = []
            for cs in clusters:
                try:
                    tfidf = TfidfVectorizer(stop_words="english", max_features=5)
                    tfidf.fit(cs)
                    terms = tfidf.get_feature_names_out()
                    out.append({"address": "/unknown/" + "-".join(terms[:3]),
                                "summary": "fallback (batch parse failed)"})
                except Exception:
                    out.append({"address": "/unknown/empty", "summary": "fallback"})
            return out

    def mint_edges_batch(self, new_addrs_with_summary: List[Dict[str, str]],
                         existing: List[Dict[str, str]]
                         ) -> List[List[Dict[str, Any]]]:
        """Emit adjacency edges for N new addresses in a single LLM call."""
        if not new_addrs_with_summary:
            return []
        if not existing:
            return [[] for _ in new_addrs_with_summary]
        new_block = "\n".join(
            f"  {n['address']}  —  {n['summary']}" for n in new_addrs_with_summary
        )
        existing_block = "\n".join(
            f"  {a['address']}  —  {a['summary']}" for a in existing[:80]
        )
        prompt = self.EDGE_BATCH_PROMPT.format(
            new_addrs=new_block, existing=existing_block,
            max_edges=MAX_NODE_DEGREE,
        )
        raw = self._call(prompt)
        self.edge_calls += 1
        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("expected JSON array")
            valid_addrs = {a["address"] for a in existing}
            out = []
            for item in data[:len(new_addrs_with_summary)]:
                edges = item.get("edges", [])
                cleaned = []
                for e in edges:
                    if e.get("to") in valid_addrs:
                        cleaned.append({"to": e["to"],
                                        "weight": float(e.get("weight", 0.5))})
                out.append(cleaned[:MAX_NODE_DEGREE])
            while len(out) < len(new_addrs_with_summary):
                out.append([])
            return out
        except Exception:
            return [[] for _ in new_addrs_with_summary]

    def mint_edges(self, new_addr: str, new_summary: str,
                   existing: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        if not existing:
            return []
        # bound prompt size: include up to 50 existing addresses
        sample = existing[:50]
        existing_str = "\n".join(f"  {a['address']}  —  {a['summary']}" for a in sample)
        prompt = self.EDGE_PROMPT.format(
            new_addr=new_addr, new_summary=new_summary,
            existing=existing_str, max_edges=MAX_NODE_DEGREE,
        )
        raw = self._call(prompt)
        self.edge_calls += 1
        try:
            data = json.loads(raw)
            edges = data.get("edges", [])
            valid_addrs = {a["address"] for a in existing}
            return [
                {"to": e["to"], "weight": float(e["weight"])}
                for e in edges if e.get("to") in valid_addrs
            ][:MAX_NODE_DEGREE]
        except Exception:
            return []


# ---------- GAC system ---------------------------------------------------

class GACSystem:
    def __init__(self, model_name: str = EMB_MODEL):
        self.emb_model = SentenceTransformer(model_name)
        self.cartographer: Optional[Cartographer] = None
        # address registry
        self.addresses: List[Dict[str, Any]] = []  # {address, summary, centroid, chunk_ids}
        self.addr_index: Dict[str, int] = {}        # address -> idx
        self.graph = nx.Graph()
        self.chunks: List[Dict[str, Any]] = []
        self.chunk_id_to_idx: Dict[str, int] = {}
        self.chunk_embs: Optional[np.ndarray] = None
        # CPU classifier (closed-set, hot path)
        self.centroids: Optional[np.ndarray] = None
        # stats
        self.index_build_time_s = 0.0
        self.embed_time_s = 0.0
        self.cluster_time_s = 0.0
        self.warm_time_s = 0.0
        self.n_chunks_indexed = 0

    # --- index build (one-time warm path work) ---------------------------

    def build_index(self, chunks: List[Dict[str, Any]], rebuild: bool = False):
        cache_path = GAC_DATA / "gac_index.json"
        emb_cache = GAC_DATA / "chunk_embs.npy"
        if cache_path.exists() and emb_cache.exists() and not rebuild:
            self._load_cache(chunks, cache_path, emb_cache)
            print(
                f"[GAC] loaded cached index: {len(self.addresses)} addresses, "
                f"{self.graph.number_of_edges()} edges, {len(self.chunks)} chunks"
            )
            return

        self.chunks = chunks
        self.chunk_id_to_idx = {c["id"]: i for i, c in enumerate(chunks)}

        # embed all chunks (one-time addressing cost, equivalent to RAG embed)
        t0 = time.perf_counter()
        texts = [c["text"] for c in chunks]
        self.chunk_embs = self.emb_model.encode(
            texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.embed_time_s = time.perf_counter() - t0
        print(f"[GAC] embedded {len(chunks)} chunks in {self.embed_time_s:.2f}s")

        # discover concept clusters (proxy for what an LLM would do iteratively)
        t1 = time.perf_counter()
        n_clusters = min(N_ADDRESSES, max(5, len(chunks) // 12))
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(self.chunk_embs)
        self.cluster_time_s = time.perf_counter() - t1
        print(f"[GAC] discovered {n_clusters} concept clusters in {self.cluster_time_s:.2f}s")

        # warm path: mint addresses + edges for each cluster
        self.cartographer = Cartographer()
        t2 = time.perf_counter()
        for cid in range(n_clusters):
            member_idxs = np.where(labels == cid)[0]
            if len(member_idxs) == 0:
                continue
            # pick representative samples nearest centroid
            cluster_embs = self.chunk_embs[member_idxs]
            centroid = cluster_embs.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-9
            sims = cluster_embs @ centroid
            order = np.argsort(-sims)
            rep_samples = [chunks[member_idxs[i]]["text"] for i in order[:5]]

            minted = self.cartographer.mint_address(rep_samples)
            address = minted["address"]
            # disambiguate duplicates
            base = address
            n = 1
            while address in self.addr_index:
                address = f"{base}#{n}"
                n += 1

            existing_for_edges = [
                {"address": a["address"], "summary": a["summary"]}
                for a in self.addresses
            ]
            new_idx = len(self.addresses)
            self.addresses.append({
                "address": address,
                "summary": minted["summary"],
                "centroid": centroid.tolist(),
                "chunk_ids": [chunks[i]["id"] for i in member_idxs.tolist()],
            })
            self.addr_index[address] = new_idx
            self.graph.add_node(address)

            edges = self.cartographer.mint_edges(
                address, minted["summary"], existing_for_edges
            )
            for e in edges:
                self.graph.add_edge(address, e["to"], weight=e["weight"])

            print(
                f"  [{cid+1}/{n_clusters}] minted  {address}  "
                f"({len(member_idxs)} chunks, {len(edges)} edges)"
            )

        # enforce bounded-degree invariant (§8) — keep top-K strongest per node
        self._enforce_degree_bound()

        # build centroid matrix for hot-path classifier
        self.centroids = np.vstack([a["centroid"] for a in self.addresses])
        self.warm_time_s = self.cartographer.total_warm_time_s
        self.index_build_time_s = (
            self.embed_time_s + self.cluster_time_s + self.warm_time_s
        )
        self.n_chunks_indexed = len(chunks)

        self._save_cache(cache_path, emb_cache)
        print(
            f"[GAC] built: {len(self.addresses)} addresses, "
            f"{self.graph.number_of_edges()} edges, "
            f"warm-path LLM calls = mint:{self.cartographer.mint_calls} "
            f"edge:{self.cartographer.edge_calls} "
            f"({self.warm_time_s:.1f}s total)"
        )

    def _enforce_degree_bound(self):
        for node in list(self.graph.nodes):
            edges = list(self.graph.edges(node, data=True))
            if len(edges) <= MAX_NODE_DEGREE:
                continue
            edges.sort(key=lambda x: -x[2].get("weight", 0))
            keep = set(tuple(sorted([u, v])) for u, v, _ in edges[:MAX_NODE_DEGREE])
            for u, v, _ in edges[MAX_NODE_DEGREE:]:
                if tuple(sorted([u, v])) not in keep:
                    if self.graph.has_edge(u, v):
                        self.graph.remove_edge(u, v)

    def _save_cache(self, cache_path: Path, emb_cache: Path):
        # graph as list of edges
        edges = [
            {"u": u, "v": v, "weight": d.get("weight", 0)}
            for u, v, d in self.graph.edges(data=True)
        ]
        data = {
            "addresses": self.addresses,
            "edges": edges,
            "stats": {
                "embed_time_s": self.embed_time_s,
                "cluster_time_s": self.cluster_time_s,
                "warm_time_s": self.warm_time_s,
                "mint_calls": self.cartographer.mint_calls if self.cartographer else 0,
                "edge_calls": self.cartographer.edge_calls if self.cartographer else 0,
            },
        }
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)
        np.save(emb_cache, self.chunk_embs)

    def _load_cache(self, chunks: List[Dict[str, Any]], cache_path: Path, emb_cache: Path):
        with open(cache_path) as f:
            data = json.load(f)
        self.addresses = data["addresses"]
        self.addr_index = {a["address"]: i for i, a in enumerate(self.addresses)}
        self.graph = nx.Graph()
        for a in self.addresses:
            self.graph.add_node(a["address"])
        for e in data["edges"]:
            self.graph.add_edge(e["u"], e["v"], weight=e["weight"])
        self.centroids = np.vstack([a["centroid"] for a in self.addresses])
        self.chunks = chunks
        self.chunk_id_to_idx = {c["id"]: i for i, c in enumerate(chunks)}
        self.chunk_embs = np.load(emb_cache)
        self.n_chunks_indexed = len(chunks)
        stats = data.get("stats", {})
        self.embed_time_s = stats.get("embed_time_s", 0)
        self.cluster_time_s = stats.get("cluster_time_s", 0)
        self.warm_time_s = stats.get("warm_time_s", 0)
        self.index_build_time_s = (
            self.embed_time_s + self.cluster_time_s + self.warm_time_s
        )

    # --- retrieval (hot path, CPU-only) ----------------------------------

    def query(self, q: str, k: int = 5) -> Dict[str, Any]:
        # 1) HOT PATH: route to single address (CPU, no per-event LLM)
        # In a production GAC the router would be TF-IDF + LogReg over the closed
        # address set. Here we use cosine to the addr centroid — equivalent
        # closed-set classification, and keeps the comparison apples-to-apples.
        t0 = time.perf_counter()
        q_emb = self.emb_model.encode([q], normalize_embeddings=True)[0]
        embed_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        sims = self.centroids @ q_emb
        routed_idx = int(np.argmax(sims))
        routed_address = self.addresses[routed_idx]["address"]
        routing_confidence = float(sims[routed_idx])
        route_ms = (time.perf_counter() - t1) * 1000

        # 2) BOUNDED TOPOLOGICAL RECOVERY (§8): one-hop graph expansion
        t2 = time.perf_counter()
        candidate_addrs = {routed_address}
        neighbours = []
        if routed_address in self.graph:
            for n in self.graph.neighbors(routed_address):
                w = self.graph[routed_address][n].get("weight", 0)
                neighbours.append((n, w))
        neighbours.sort(key=lambda x: -x[1])
        for n, _ in neighbours[:MAX_NODE_DEGREE]:
            candidate_addrs.add(n)

        candidate_chunk_ids = []
        for addr in candidate_addrs:
            if addr in self.addr_index:
                candidate_chunk_ids.extend(
                    self.addresses[self.addr_index[addr]]["chunk_ids"]
                )
        # apply expansion ceiling
        candidate_chunk_ids = candidate_chunk_ids[:CANDIDATE_CEILING]
        expand_ms = (time.perf_counter() - t2) * 1000

        # 3) SCOPED REFINEMENT: cosine rerank inside the bounded candidate set
        # (the whitepaper allows the refinement to be semantic; the contribution
        # is that it runs over ~hundreds, not the whole corpus.)
        t3 = time.perf_counter()
        cand_idxs = [self.chunk_id_to_idx[cid] for cid in candidate_chunk_ids
                     if cid in self.chunk_id_to_idx]
        cand_embs = self.chunk_embs[cand_idxs]
        rerank_sims = cand_embs @ q_emb
        order = np.argsort(-rerank_sims)[:k]
        refine_ms = (time.perf_counter() - t3) * 1000

        hits = []
        for o in order:
            ci = cand_idxs[o]
            c = self.chunks[ci]
            hits.append({
                "id": c["id"],
                "doc": c["doc"],
                "section": c["section"],
                "score": float(rerank_sims[o]),
                "text": c["text"],
                "via_address": next(
                    (a["address"] for a in self.addresses if c["id"] in a["chunk_ids"]),
                    "?"
                ),
            })

        total_ms = embed_ms + route_ms + expand_ms + refine_ms
        return {
            "hits": hits,
            "routed_address": routed_address,
            "routing_confidence": routing_confidence,
            "expanded_neighbours": [n for n, _ in neighbours[:MAX_NODE_DEGREE]],
            "candidate_set_size": len(cand_idxs),
            "corpus_size": self.n_chunks_indexed,
            "reduction_ratio": self.n_chunks_indexed / max(1, len(cand_idxs)),
            "embed_ms": embed_ms,
            "route_ms": route_ms,
            "expand_ms": expand_ms,
            "refine_ms": refine_ms,
            "total_ms": total_ms,
            "explanation": (
                f"routed to {routed_address} (conf {routing_confidence:.2f}); "
                f"+ {len(candidate_addrs)-1} neighbour(s); "
                f"refined over {len(cand_idxs)}/{self.n_chunks_indexed} chunks"
            ),
        }


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH) as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


if __name__ == "__main__":
    import sys
    rebuild = "--rebuild" in sys.argv
    chunks = load_chunks()
    print(f"loaded {len(chunks)} chunks")
    gac = GACSystem()
    gac.build_index(chunks, rebuild=rebuild)

    print("\n--- address space sample ---")
    for a in gac.addresses[:10]:
        n_nb = gac.graph.degree(a["address"])
        print(f"  {a['address']:50s} ({len(a['chunk_ids']):3d} chunks, {n_nb} edges) "
              f"— {a['summary'][:70]}")

    print("\n--- smoke test ---")
    for q in ["What is product X?",
              "Show me case studies for industry Y",
              "How does feature Z work?"]:
        r = gac.query(q)
        print(f"\nQ: {q}")
        print(f"   {r['explanation']}  ({r['total_ms']:.1f}ms)")
        for h in r["hits"][:3]:
            snippet = h["text"][:120].replace("\n", " ")
            print(f"   {h['score']:.3f} [{h['via_address']}] {snippet}…")
