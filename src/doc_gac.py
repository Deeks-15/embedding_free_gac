"""DocGAC — embedding-free GAC for prose documents.

Built per the user direction: "no matter what, we don't use embedding and
cosine similarity search in this new approach." DocGAC is the document
analog of DrainGAC — same architecture, different routing primitive.

Hot-path primitives (all deterministic, NO embeddings, NO cosine):

  1. Address minting (warm path, one-time per corpus):
        - YAKE extracts top-K keywords per chunk (deterministic, term-based)
        - Build address signatures = (top-3 chunk keywords ∪ section tag)
        - Cluster chunks by signature overlap (Jaccard ≥ threshold)
        - Address name = "/topic/{shared keywords}/{section}"
        - NO LLM mint required (deterministic naming from keywords)

  2. Per-query routing:
        - YAKE extracts query keywords (deterministic)
        - Score each address by:
              keyword-overlap (Jaccard with address keywords)
            + BM25 over address summary + chunks
        - Top scoring address(es) are candidates
        - If no address has any term overlap → empty result ("I don't know")

  3. Per-query reranking inside candidates:
        - BM25 over chunk text (term-based, no embeddings)
        - + section-name proximity tiebreaker
        - Top-k returned

Interface mirrors Phase1GAC / DrainGAC so the same eval harness runs unchanged:
  .bootstrap(chunks)
  .stream_ingest(chunks)   # no-op for docs
  .query(q, k=5)
"""
from __future__ import annotations
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Deliberately not importing numpy, sentence_transformers, or sklearn
# similarity primitives — the whole point of DocGAC is to demonstrate
# the architecture works without any neural embedding on the hot path.

import yake

import sys
sys.path.insert(0, str(Path(__file__).parent))

# Shared stopwords (kept same as DrainGAC for consistency)
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "from", "of", "with", "by", "as",
    "and", "or", "but", "not", "do", "does", "did", "have", "has", "had",
    "this", "that", "these", "those", "it", "its", "i", "we", "you", "they",
    "my", "your", "their", "show", "me", "tell", "what", "how", "why",
    "where", "when", "which", "who", "find", "list", "get", "give",
    "all", "any", "some", "no", "more", "less", "many", "few",
    "page", "slide", "header", "section", "introduction",
}


def _tokenize(text: str) -> List[str]:
    """Cheap deterministic tokenizer — alphanumeric, lowercased, stopwords filtered."""
    return [w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", (text or "").lower())
            if w not in _STOPWORDS]


# Singleton YAKE extractor reused per query / chunk
_YAKE = {"kw": None}


def _yake_keywords(text: str, top: int = 6) -> List[str]:
    """Extract top-K keyword phrases via YAKE (deterministic, no neural net)."""
    if _YAKE["kw"] is None:
        _YAKE["kw"] = yake.KeywordExtractor(
            lan="en", n=2, dedupLim=0.7, top=top, features=None,
        )
    if not text or len(text.strip()) < 10:
        return []
    try:
        pairs = _YAKE["kw"].extract_keywords(text[:5000])
        # YAKE returns (keyword, score) — lower score = better
        pairs.sort(key=lambda x: x[1])
        return [kw.lower() for kw, _ in pairs[:top]]
    except Exception:
        return []


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class DocGAC:
    """Embedding-free GAC for prose documents.

    Counters:
        queries_routed_deterministic  — full keyword-route, no BM25 needed
        queries_routed_bm25           — BM25 fallback
        queries_empty                 — no term overlap at all
        query_time_embed_calls        — must always be 0
    """

    def __init__(self, random_seed: int = 42,
                  cluster_jaccard_threshold: float = 0.4,
                  yake_top: int = 6):
        self.cluster_jaccard_threshold = cluster_jaccard_threshold
        self.yake_top = yake_top
        self.random_seed = random_seed

        # Address registry
        self.addresses: List[Dict[str, Any]] = []
        self.n_indexed = 0

        # Counters
        self.queries_routed_deterministic = 0
        self.queries_routed_bm25 = 0
        self.queries_empty = 0
        self.query_time_embed_calls = 0   # invariant: stays 0

        # Cartographer skipped — addresses named deterministically from
        # keywords. (DocGAC has no LLM dependency at all.)
        self.total_mint_calls = 0
        self.total_edge_calls = 0
        self.total_cartographer_usd = 0.0

    # ---------- bootstrap (warm path, ONE-TIME, no LLM, no embeddings) ----

    def bootstrap(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract keywords per chunk, cluster by signature overlap,
        create one address per cluster, name deterministically."""
        t0 = time.perf_counter()
        if not chunks:
            return {"ingest_ms": 0, "total_addresses": 0,
                    "total_mint_llm_calls": 0, "total_edge_llm_calls": 0,
                    "total_cartographer_usd": 0.0, "build_secs": 0,
                    "n_primary_clusters": 0, "n_secondpass_clusters": 0,
                    "stratum_breakdown": {}}

        # Step 1: extract keywords per chunk
        chunk_keywords: List[set] = []
        chunk_sections: List[str] = []
        chunk_docs: List[str] = []
        for c in chunks:
            kws = _yake_keywords(c["text"], top=self.yake_top)
            sig = set(kws)
            # also include tokens from doc filename + section
            for tok in _tokenize(c.get("doc", "") + " " + c.get("section", "")):
                if len(tok) > 3:
                    sig.add(tok)
            chunk_keywords.append(sig)
            chunk_sections.append(c.get("section", "?"))
            chunk_docs.append(c.get("doc", "?"))

        # Step 2: deterministic clustering by signature Jaccard
        # Greedy: walk chunks in order, assign to first cluster with
        # signature overlap >= threshold, else create new cluster.
        # Stable: same input → same output.
        clusters: List[Dict[str, Any]] = []
        for i, sig in enumerate(chunk_keywords):
            assigned = False
            best_cluster = -1
            best_jac = 0.0
            for ci, cluster in enumerate(clusters):
                jac = _jaccard(sig, cluster["signature"])
                if jac >= self.cluster_jaccard_threshold and jac > best_jac:
                    best_jac = jac
                    best_cluster = ci
            if best_cluster >= 0:
                clusters[best_cluster]["chunk_idxs"].append(i)
                # merge signatures (intersection — keeps shared theme tight)
                # OR union (more inclusive) — we use union with cap to avoid bloat
                merged = clusters[best_cluster]["signature"] | sig
                if len(merged) > 20:
                    # cap signature size to keep clusters coherent
                    merged = set(list(clusters[best_cluster]["signature"])[:15]) | set(list(sig)[:5])
                clusters[best_cluster]["signature"] = merged
                assigned = True
            if not assigned:
                clusters.append({
                    "signature": set(sig),
                    "chunk_idxs": [i],
                })

        # Step 3: build addresses from clusters (deterministic naming)
        for ci, cluster in enumerate(clusters):
            chunk_idxs = cluster["chunk_idxs"]
            if not chunk_idxs:
                continue
            sig = cluster["signature"]
            # Address name: take top 3 keywords by frequency in cluster
            kw_counter: Counter = Counter()
            for i in chunk_idxs:
                for kw in chunk_keywords[i]:
                    kw_counter[kw] += 1
            top_kws = [kw for kw, _ in kw_counter.most_common(3)]
            if not top_kws:
                top_kws = [f"cluster-{ci}"]
            # safe path component (replace non-alnum)
            safe_path = "/".join(
                re.sub(r"[^a-zA-Z0-9_]+", "-", kw.replace(" ", "-"))[:25]
                for kw in top_kws
            )
            address_name = "/doc/" + safe_path
            # Dedupe
            existing = {a["address"] for a in self.addresses}
            base = address_name
            n = 1
            while address_name in existing:
                address_name = f"{base}#{n}"
                n += 1

            # Dominant doc + section (for hint-based routing later)
            doc_counter = Counter(chunk_docs[i] for i in chunk_idxs)
            sec_counter = Counter(chunk_sections[i] for i in chunk_idxs)

            # Build summary: top 3 keywords joined
            summary = "Topic: " + ", ".join(top_kws)

            self.addresses.append({
                "address": address_name,
                "summary": summary,
                "signature": sig,           # set of keywords + tokens
                "top_keywords": top_kws,
                "chunk_ids": [f"doc{i:05d}" for i in chunk_idxs],
                "chunk_texts": [chunks[i]["text"] for i in chunk_idxs],
                "chunk_docs": [chunk_docs[i] for i in chunk_idxs],
                "chunk_sections": [chunk_sections[i] for i in chunk_idxs],
                "n_chunks": len(chunk_idxs),
                "dominant_doc": doc_counter.most_common(1)[0][0] if doc_counter else None,
                "dominant_section": sec_counter.most_common(1)[0][0] if sec_counter else None,
                "doc_purity": doc_counter.most_common(1)[0][1] / len(chunk_idxs) if doc_counter else 0,
            })

        self.n_indexed = len(chunks)
        ms = (time.perf_counter() - t0) * 1000
        print(f"  DocGAC bootstrap: {len(self.addresses)} addresses from "
              f"{len(chunks)} chunks (jaccard≥{self.cluster_jaccard_threshold}, "
              f"YAKE-top={self.yake_top}), {ms/1000:.1f}s, 0 LLM calls, "
              "0 embedding calls")

        # Precompute corpus-wide IDF for BM25 over chunk texts
        self._precompute_bm25_idf()

        return {
            "ingest_ms": ms,
            "total_addresses": len(self.addresses),
            "total_mint_llm_calls": 0,
            "total_edge_llm_calls": 0,
            "total_cartographer_usd": 0.0,
            "build_secs": ms / 1000,
            "n_primary_clusters": len(self.addresses),
            "n_secondpass_clusters": 0,
            "stratum_breakdown": {},
            "n_addresses": len(self.addresses),
            "embedding_calls": 0,
        }

    # ---------- stream ingest (no-op for static doc corpus) -------------

    def stream_ingest(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Doc corpora are not streaming. If you call this, it just runs
        bootstrap-style for the new chunks. For Phase 4 we pass all chunks
        at bootstrap, so stream_ingest gets [] and is a no-op."""
        if not chunks:
            return {"ingest_ms": 0, "n_new": 0, "embed_calls": 0}
        # Route each new chunk to nearest cluster by Jaccard
        for c in chunks:
            kws = _yake_keywords(c["text"], top=self.yake_top)
            sig = set(kws)
            for tok in _tokenize(c.get("doc", "") + " " + c.get("section", "")):
                if len(tok) > 3:
                    sig.add(tok)
            best_idx, best_jac = -1, 0.0
            for ai, a in enumerate(self.addresses):
                jac = _jaccard(sig, a["signature"])
                if jac > best_jac:
                    best_jac = jac
                    best_idx = ai
            if best_idx >= 0 and best_jac >= self.cluster_jaccard_threshold:
                a = self.addresses[best_idx]
                a["chunk_ids"].append(f"doc{self.n_indexed:05d}")
                a["chunk_texts"].append(c["text"])
                a["chunk_docs"].append(c.get("doc", "?"))
                a["chunk_sections"].append(c.get("section", "?"))
                a["n_chunks"] += 1
            else:
                # would create a new address; for simplicity, force-assign to nearest
                if self.addresses:
                    a = self.addresses[best_idx if best_idx >= 0 else 0]
                    a["chunk_ids"].append(f"doc{self.n_indexed:05d}")
                    a["chunk_texts"].append(c["text"])
                    a["chunk_docs"].append(c.get("doc", "?"))
                    a["chunk_sections"].append(c.get("section", "?"))
                    a["n_chunks"] += 1
            self.n_indexed += 1
        return {"ingest_ms": 0, "n_new": len(chunks), "embed_calls": 0}

    # ---------- BM25 ----------------------------------------------------

    def _precompute_bm25_idf(self):
        """One-time IDF over the full chunk corpus. Used by BM25 fallback and rerank."""
        self.N_docs = sum(len(a["chunk_texts"]) for a in self.addresses)
        df: Counter = Counter()
        # token to doc-frequency
        for a in self.addresses:
            for txt in a["chunk_texts"]:
                seen = set(_tokenize(txt))
                for tok in seen:
                    df[tok] += 1
        self.idf = {tok: math.log(1 + (self.N_docs - n + 0.5) / (n + 0.5))
                    for tok, n in df.items()}
        # avg doc length (in tokens) for BM25 normalisation
        total_len = 0
        n_d = 0
        for a in self.addresses:
            for txt in a["chunk_texts"]:
                total_len += len(_tokenize(txt))
                n_d += 1
        self.avg_dl = total_len / max(1, n_d)
        self.k1 = 1.5
        self.b = 0.75

    def _bm25_score(self, doc_tokens: List[str], query_tokens: List[str]) -> float:
        if not doc_tokens or not query_tokens:
            return 0.0
        tf: Counter = Counter(doc_tokens)
        dl = len(doc_tokens)
        score = 0.0
        for tk in query_tokens:
            if tk not in tf:
                continue
            f = tf[tk]
            idf = self.idf.get(tk, math.log(1 + (self.N_docs + 0.5) / 0.5))
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(1, self.avg_dl))
            score += idf * (f * (self.k1 + 1)) / denom
        return score

    # ---------- query (HOT PATH — embedding-free) ------------------------

    def query(self, q: str, k: int = 5, query_hint_svc=None, query_hint_lvl=None):
        """Per-query hot path: deterministic keyword routing + BM25 fallback.
        ZERO embedding calls. Returns empty if no term overlap anywhere."""
        t0 = time.perf_counter()
        # Extract query keywords (deterministic, no embeddings)
        q_yake = _yake_keywords(q, top=self.yake_top)
        q_tokens = _tokenize(q)
        q_signature = set(q_yake) | set(q_tokens)

        if not q_signature:
            ms = (time.perf_counter() - t0) * 1000
            self.queries_empty += 1
            return self._empty_result(t0, "no query tokens")

        # Step 1: deterministic candidate filter via Jaccard
        candidates_jac = []
        for i, a in enumerate(self.addresses):
            jac = _jaccard(q_signature, a["signature"])
            if jac > 0:
                candidates_jac.append((i, jac))

        if candidates_jac:
            self.queries_routed_deterministic += 1
            # Take top-3 candidates by Jaccard for reranking
            candidates_jac.sort(key=lambda x: -x[1])
            cand_idxs = [i for i, _ in candidates_jac[:5]]
            fallback_used = False
            fallback_kind = ""
        else:
            # Step 2: BM25 fallback over address summaries + top chunks
            self.queries_routed_bm25 += 1
            fallback_used = True
            fallback_kind = "bm25"
            scored = []
            for i, a in enumerate(self.addresses):
                # score against address summary + first chunk text + top keywords
                doc_text = (a["summary"] + " " + " ".join(a["top_keywords"]) +
                            " " + (a["chunk_texts"][0] if a["chunk_texts"] else ""))
                doc_tokens = _tokenize(doc_text)
                score = self._bm25_score(doc_tokens, q_tokens)
                if score > 0:
                    scored.append((i, score))
            if not scored:
                self.queries_empty += 1
                return self._empty_result(t0, "no BM25 overlap")
            scored.sort(key=lambda x: -x[1])
            cand_idxs = [i for i, _ in scored[:5]]

        # Step 3: deterministic chunk reranking via BM25 + section bonus
        all_chunks = []
        for ai in cand_idxs:
            a = self.addresses[ai]
            for ci, txt in enumerate(a["chunk_texts"]):
                doc_tokens = _tokenize(txt)
                bm25 = self._bm25_score(doc_tokens, q_tokens)
                # small bonus if doc/section name has query overlap
                sec_tokens = set(_tokenize(a["chunk_sections"][ci] +
                                            " " + a["chunk_docs"][ci]))
                sec_bonus = len(q_signature & sec_tokens) * 0.5
                all_chunks.append({
                    "id": a["chunk_ids"][ci],
                    "score": bm25 + sec_bonus,
                    "text": txt,
                    "via_address": a["address"],
                })
        all_chunks.sort(key=lambda x: -x["score"])
        # filter zero scores (they're not even partial matches)
        hits = [c for c in all_chunks if c["score"] > 0][:k]

        ms = (time.perf_counter() - t0) * 1000
        n_cand_chunks = sum(self.addresses[i]["n_chunks"] for i in cand_idxs)
        return {
            "hits": hits,
            "candidate_set_size": n_cand_chunks,
            "corpus_size": self.n_indexed,
            "reduction_ratio": self.n_indexed / max(1, n_cand_chunks),
            "routed_address": self.addresses[cand_idxs[0]]["address"] if cand_idxs else "",
            "expanded_neighbours": [],
            "fallback_used": fallback_used,
            "fallback_kind": fallback_kind,
            "total_ms": ms,
            "cost_usd": 0.0,    # zero LLM/API cost per query
            "embed_calls_this_query": 0,   # invariant
        }

    def _empty_result(self, t0, reason):
        ms = (time.perf_counter() - t0) * 1000
        return {
            "hits": [], "candidate_set_size": 0,
            "corpus_size": self.n_indexed, "reduction_ratio": 0,
            "routed_address": "",
            "expanded_neighbours": [], "fallback_used": True,
            "fallback_kind": f"empty: {reason}",
            "total_ms": ms, "cost_usd": 0.0,
            "embed_calls_this_query": 0,
        }


if __name__ == "__main__":
    # Smoke test on the existing doc-corpus chunks
    print("DocGAC smoke test")
    import json
    chunks = []
    chunks_path = Path(__file__).resolve().parent.parent / "data" / "chunks.jsonl"
    with open(chunks_path) as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"  loaded {len(chunks)} chunks from {len(set(c['doc'] for c in chunks))} docs")
    g = DocGAC()
    g.bootstrap(chunks)
    print(f"  built {len(g.addresses)} addresses")
    print(f"  sample addresses:")
    for a in g.addresses[:5]:
        print(f"    {a['address']}  ({a['n_chunks']} chunks, top kws: {a['top_keywords']})")
    print(f"\n  sample queries (no embeddings used):")
    for q in ["What is product X?",
              "Show me GenAI case studies",
              "prompt advisor tool",
              "AI in mobility industry"]:
        r = g.query(q)
        n_hits = len(r["hits"])
        fb = r["fallback_kind"] or "deterministic"
        embed = r["embed_calls_this_query"]
        print(f"    '{q[:35]}' → {n_hits} hits, {r['total_ms']:.2f}ms, "
              f"fallback={fb}, embed_calls={embed}, route={r['routed_address']}")
