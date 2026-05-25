"""DrainGAC — authentic GAC for logs (deterministic-routing hot path).

This is the architecture the GAC whitepaper §3 actually promises:
    * Per-event ingest: deterministic template extraction → exact address lookup.
      ZERO embeddings, ZERO cosine on the hot path.
    * Per-query retrieval: deterministic tag matching → exact template_id route.
      ONE embedding only as fallback when no template matches the query intent.
    * Warm path: LLM mints an address (name + summary + edges) for each
      genuinely novel template Drain3 discovers. Same Cartographer as before.

Contrast with the existing TunedGAC ("softened GAC") which embeds + cosines
throughout the hot path and is in practice bounded-ANN RAG. The pilot audit
made this difference explicit; DrainGAC is the test of whether the authentic
architecture closes the accuracy gap that the softened version couldn't.

Interface mirrors Phase1GAC so the existing eval harness can drive it:
    .bootstrap(chunks)
    .stream_ingest(chunks)
    .query(q, k=5)

But the internals are different:
    * No address centroids — addresses identified by template_id.
    * Drain3 template mining replaces KMeans/HDBSCAN.
    * Retrieval ranks by recency × level-match × keyword-overlap, NOT cosine.
"""
from __future__ import annotations
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Deliberately NOT importing numpy/sentence_transformers/sklearn cosine —
# the entire DrainGAC path is pure-Python deterministic + term-based.

import sys
sys.path.insert(0, str(Path(__file__).parent))

# Drain3 import
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

# Reuse cartographer + deterministic query-tag extraction without modification
from gac import Cartographer, MAX_NODE_DEGREE
from streaming_tuned import extract_query_tags
from streaming_replay import (
    EMBED_USD_PER_TOKEN, GEMINI_IN, GEMINI_OUT, TOKENS_PER_QUERY,
    MINT_IN_TOK, MINT_OUT_TOK, EDGE_IN_TOK, EDGE_OUT_TOK,
    USD_PER_MINT, USD_PER_EDGE,
)

# NO embedding model imports anywhere. The entire hot path — ingest AND query
# AND fallback — uses only deterministic / term-based operations:
#     * Drain3 (regex-based template extraction)
#     * Token overlap counts (deterministic set intersection)
#     * BM25 ranking (term-based, no neural embeddings)
#     * Recency + level scoring (counter-based)
# This is the architecture the whitepaper §3 actually claims: "retrieval
# itself never performs semantics again."


# Stopwords for keyword overlap (kept small; logs are domain-specific)
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "from", "of", "with", "by", "as",
    "and", "or", "but", "not", "do", "does", "did", "have", "has", "had",
    "this", "that", "these", "those", "it", "its", "i", "we", "you", "they",
    "my", "your", "their", "show", "me", "tell", "what", "how", "why",
    "where", "when", "which", "who", "find", "list", "get", "give",
    "all", "any", "some", "no", "more", "less", "many", "few",
}


def _tokenize(text: str) -> List[str]:
    """Cheap tokeniser for keyword overlap — alphanumeric tokens lowercased."""
    return [w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", text.lower())
            if w not in _STOPWORDS]


class DrainGAC:
    """Authentic-architecture GAC for log data.

    Public API mirrors Phase1GAC so existing eval drivers can use it
    unchanged. Internals are deterministic template-based routing.

    Counters tracked for the report:
        events_with_known_template   # ingest hits, no warm-path mint
        events_with_novel_template   # warm-path mint fired
        queries_routed_deterministic # full hot path, no embedding
        queries_routed_fallback      # embedding fallback path used
    """

    def __init__(self, random_seed: int = 42, mint_batch_size: int = 10,
                 skip_cartographer: bool = False):
        """skip_cartographer: when True, use deterministic naming derived from
        template tokens (no LLM call). This is used when Gemini is unhealthy
        or when the experiment doesn't care about pretty address names; the
        retrieval mechanism is unchanged."""
        self.skip_cartographer = skip_cartographer
        # ---- Drain3 setup ----
        # Drain3 is online; we feed it lines one at a time and it grows
        # template clusters on the fly. Single in-memory persistence.
        cfg = TemplateMinerConfig()
        cfg.profiling_enabled = False
        cfg.snapshot_interval_minutes = 0  # no auto-snapshot; we manage state
        cfg.drain_extra_delimiters = []
        cfg.drain_max_clusters = None
        cfg.drain_sim_th = 0.4               # similarity threshold for grouping
        cfg.drain_depth = 4                   # tree depth — standard default
        cfg.drain_max_node_depth = 4
        cfg.drain_max_children = 100
        cfg.parametrize_numeric_tokens = True
        cfg.mask_prefix = "<"
        cfg.mask_suffix = ">"
        self.drain = TemplateMiner(config=cfg)

        # ---- Address registry ----
        # Each address corresponds to one Drain3 template cluster.
        # Key fields:
        #   address      — canonical name (LLM-minted)
        #   summary      — 1-sentence intent (LLM-minted)
        #   template     — the Drain3 wildcard pattern (e.g. "User <:USERNAME:> logged in from <:IP:>")
        #   template_id  — Drain3's cluster_id
        #   chunk_ids    — list of chunk IDs assigned here
        #   chunk_texts  — original text per chunk (for reranking)
        #   chunk_levels — INFO/WARN/ERROR per chunk
        #   chunk_recency— monotonically increasing counter (acts as ordering signal)
        #   dominant_svc — service tag (Counter most_common)
        #   dominant_lvl — level tag (Counter most_common)
        #   template_keywords — set of meaningful tokens in the template body
        self.addresses: List[Dict[str, Any]] = []
        self.address_by_template_id: Dict[int, int] = {}  # template_id → address index
        self.template_id_to_template: Dict[int, str] = {}

        # Warm-path queue: novel templates waiting to be minted by the cartographer
        self.pending_novel: List[Dict[str, Any]] = []

        # ---- Adjacency graph (§8) ----
        import networkx as nx
        self.graph = nx.Graph()

        # ---- Cartographer (LLM, rare) ----
        self.cartographer: Optional[Cartographer] = None
        self.mint_batch_size = mint_batch_size
        self.total_mint_calls = 0
        self.total_edge_calls = 0
        self.total_cartographer_usd = 0.0

        # ---- Counters / stats ----
        self.events_with_known_template = 0
        self.events_with_novel_template = 0
        self.queries_routed_deterministic = 0
        self.queries_routed_fallback = 0
        self.global_chunk_counter = 0  # for recency ordering
        self.random_seed = random_seed
        self.n_indexed = 0

    # ---------- helpers --------------------------------------------------

    def _extract_template_keywords(self, template_str: str) -> set:
        """Extract meaningful tokens from a Drain3 template (post-masking).
        These are the literal tokens — the wildcards <:X:> are NOT included."""
        # remove wildcard placeholders
        cleaned = re.sub(r"<[^>]+>", " ", template_str)
        return set(_tokenize(cleaned))

    def _ingest_one(self, chunk: Dict[str, Any]) -> Tuple[int, bool]:
        """Send one chunk through Drain3; return (template_id, is_novel)."""
        text = chunk["text"]
        result = self.drain.add_log_message(text)
        template_id = result["cluster_id"]
        template_str = result["template_mined"]
        is_novel = template_id not in self.address_by_template_id
        if is_novel:
            self.template_id_to_template[template_id] = template_str
        return template_id, is_novel

    def _assign_chunk(self, chunk: Dict[str, Any], address_idx: int):
        """Append a chunk to an existing address. Pure dict append."""
        a = self.addresses[address_idx]
        a["chunk_ids"].append(f"d{self.global_chunk_counter:08d}")
        a["chunk_texts"].append(chunk["text"])
        a["chunk_levels"].append((chunk.get("level") or "UNKNOWN").upper())
        a["chunk_recency"].append(self.global_chunk_counter)
        if chunk.get("svc"):
            a["svc_counter"][chunk["svc"]] += 1
        if chunk.get("level"):
            a["lvl_counter"][chunk["level"]] += 1
        self.global_chunk_counter += 1

    def _add_new_address(self, template_id: int, address_name: str,
                          summary: str, sample_chunks: List[Dict[str, Any]]):
        """Create a new address record for a Drain3 template that the
        cartographer has just minted a name for."""
        template_str = self.template_id_to_template.get(template_id, "")
        # disambiguate name collisions (rare)
        base = address_name
        existing_names = {a["address"] for a in self.addresses}
        n = 1
        while address_name in existing_names:
            address_name = f"{base}#{n}"
            n += 1

        addr = {
            "address": address_name,
            "summary": summary,
            "template": template_str,
            "template_id": template_id,
            "template_keywords": self._extract_template_keywords(template_str),
            "chunk_ids": [],
            "chunk_texts": [],
            "chunk_levels": [],
            "chunk_recency": [],
            "svc_counter": Counter(),
            "lvl_counter": Counter(),
            "dominant_svc": None,
            "dominant_lvl": None,
            "svc_purity": 0.0,
            "lvl_purity": 0.0,
            "origin": "drain-template",
        }
        idx = len(self.addresses)
        self.addresses.append(addr)
        self.address_by_template_id[template_id] = idx
        self.graph.add_node(address_name)
        # Now actually assign the sample chunks to this address
        for c in sample_chunks:
            self._assign_chunk(c, idx)
        self._refresh_dominant_tags(idx)
        return idx

    def _refresh_dominant_tags(self, address_idx: int):
        a = self.addresses[address_idx]
        if a["svc_counter"]:
            most = a["svc_counter"].most_common(1)[0]
            a["dominant_svc"] = most[0]
            a["svc_purity"] = most[1] / max(1, sum(a["svc_counter"].values()))
        if a["lvl_counter"]:
            most = a["lvl_counter"].most_common(1)[0]
            a["dominant_lvl"] = most[0]
            a["lvl_purity"] = most[1] / max(1, sum(a["lvl_counter"].values()))

    # ---------- bootstrap (warm path) ------------------------------------

    def bootstrap(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pass every chunk through Drain3, then mint addresses for the
        discovered templates via batched cartographer calls.

        After bootstrap, the address space is fully populated for any template
        Drain3 saw during these N chunks. Subsequent stream_ingest() lines that
        match these templates route by dict-lookup with ZERO embedding calls.
        """
        t0 = time.perf_counter()

        # Group chunks by Drain3 template_id
        chunks_by_tid: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for c in chunks:
            tid, _ = self._ingest_one(c)
            chunks_by_tid[tid].append(c)

        n_templates_discovered = len(chunks_by_tid)
        print(f"  Drain3 discovered {n_templates_discovered} templates "
              f"from {len(chunks)} bootstrap lines")

        # Batched cartographer minting (or deterministic fallback if skipped)
        if self.cartographer is None and not self.skip_cartographer:
            self.cartographer = Cartographer()

        template_ids = list(chunks_by_tid.keys())
        new_addr_records = []   # list of (template_id, address_name, summary)

        def _deterministic_minted_list(batch_tids):
            """Generate deterministic address names from template keywords.
            Pure Python; no LLM."""
            minted = []
            for tid in batch_tids:
                tokens = list(self._extract_template_keywords(
                    self.template_id_to_template[tid]))[:4]
                minted.append({
                    "address": "/log/" + "/".join(tokens or [f"template-{tid}"]),
                    "summary": f"Drain3 template #{tid}: " +
                               self.template_id_to_template[tid][:80],
                })
            return minted

        for batch_start in range(0, len(template_ids), self.mint_batch_size):
            batch_tids = template_ids[batch_start:batch_start + self.mint_batch_size]
            if self.skip_cartographer:
                # PURE DETERMINISTIC PATH — no LLM at all
                minted_list = _deterministic_minted_list(batch_tids)
                # no cost increment (no LLM)
            else:
                samples_per_template = []
                for tid in batch_tids:
                    samples = chunks_by_tid[tid][:5]
                    samples_per_template.append([s["text"] for s in samples])
                try:
                    minted_list = self.cartographer.mint_addresses_batch(samples_per_template)
                except Exception as e:
                    print(f"  [WARN] batched mint failed: {e}")
                    minted_list = _deterministic_minted_list(batch_tids)
                self.total_mint_calls += 1
                self.total_cartographer_usd += (
                    MINT_IN_TOK * len(batch_tids) * GEMINI_IN +
                    MINT_OUT_TOK * len(batch_tids) * GEMINI_OUT
                )

            for tid, minted in zip(batch_tids, minted_list):
                addr_idx = self._add_new_address(
                    tid, minted["address"], minted["summary"],
                    chunks_by_tid[tid],
                )
                new_addr_records.append({
                    "address": self.addresses[addr_idx]["address"],
                    "summary": minted["summary"],
                })

        # Batched edge minting — skip entirely when cartographer is disabled.
        # Without LLM-minted edges, the graph stays empty; one-hop recovery
        # never fires. That's an honest architectural cost of skip_cartographer.
        if not self.skip_cartographer:
            for batch_start in range(0, len(new_addr_records), self.mint_batch_size):
                batch_new = new_addr_records[batch_start:batch_start + self.mint_batch_size]
                existing = new_addr_records[:batch_start]
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

        # enforce degree bound
        for node in list(self.graph.nodes):
            edges = list(self.graph.edges(node, data=True))
            if len(edges) > MAX_NODE_DEGREE:
                edges.sort(key=lambda x: -x[2].get("weight", 0))
                for u, v, _ in edges[MAX_NODE_DEGREE:]:
                    if self.graph.has_edge(u, v):
                        self.graph.remove_edge(u, v)

        self.n_indexed = len(chunks)
        ms = (time.perf_counter() - t0) * 1000
        print(f"  DrainGAC bootstrap: {len(self.addresses)} addresses, "
              f"{self.total_mint_calls} mint + {self.total_edge_calls} edge LLM calls, "
              f"{ms/1000:.1f}s")
        return {
            "ingest_ms": ms,
            "n_new": len(chunks),
            "total_addresses": len(self.addresses),
            "total_mint_llm_calls": self.total_mint_calls,
            "total_edge_llm_calls": self.total_edge_calls,
            "total_cartographer_usd": self.total_cartographer_usd,
            "build_secs": ms / 1000,
            "n_templates_discovered": n_templates_discovered,
            "n_primary_clusters": n_templates_discovered,
            "n_secondpass_clusters": 0,
            "stratum_breakdown": {},
        }

    # ---------- stream ingest (HOT PATH — should be zero-embedding) ------

    def stream_ingest(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Per-event hot path. Each chunk: Drain3 mine → dict lookup → append.
        ZERO embedding calls if all templates are known.
        Novel templates queued for periodic batched cartographer mint."""
        if not chunks:
            return {"ingest_ms": 0, "n_new": 0, "n_known": 0, "n_novel": 0,
                    "embed_calls": 0}
        t0 = time.perf_counter()
        n_known = 0
        n_novel_inline = 0

        for c in chunks:
            tid, is_novel = self._ingest_one(c)
            if is_novel:
                # Queue for batched mint; for now, lose the chunk OR attach
                # to a pending placeholder. We choose: hold a pending list
                # and mint when the queue has ≥ mint_batch_size novel
                # templates. Until then these lines are effectively
                # "not retrievable yet" — like a real warm-path delay.
                self.pending_novel.append({
                    "tid": tid, "template_str": self.template_id_to_template[tid],
                    "chunk": c,
                })
                n_novel_inline += 1
                self.events_with_novel_template += 1
            else:
                # FAST PATH: dict lookup → append. NO EMBEDDING.
                self._assign_chunk(c, self.address_by_template_id[tid])
                self._refresh_dominant_tags(self.address_by_template_id[tid])
                n_known += 1
                self.events_with_known_template += 1

        # Periodically mint novel templates (when batch_size accumulated)
        if len(self.pending_novel) >= self.mint_batch_size:
            self._drain_pending_novel()

        self.n_indexed += len(chunks)
        ms = (time.perf_counter() - t0) * 1000
        return {
            "ingest_ms": ms,
            "n_new": len(chunks),
            "n_known": n_known,
            "n_novel": n_novel_inline,
            "embed_calls": 0,  # !!! the defining property of authentic GAC
            "pending_novel": len(self.pending_novel),
        }

    def _drain_pending_novel(self):
        """Mint addresses for all currently-pending novel templates."""
        if not self.pending_novel:
            return
        # Group by template_id (each pending item is from one tid, but
        # we may have multiple chunks per tid)
        by_tid: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for p in self.pending_novel:
            by_tid[p["tid"]].append(p["chunk"])

        if self.cartographer is None and not self.skip_cartographer:
            self.cartographer = Cartographer()

        tids = list(by_tid.keys())
        new_addr_records = []
        for batch_start in range(0, len(tids), self.mint_batch_size):
            batch_tids = tids[batch_start:batch_start + self.mint_batch_size]
            if self.skip_cartographer:
                minted_list = [{
                    "address": "/log/streamed/" + "_".join(
                        list(self._extract_template_keywords(
                            self.template_id_to_template[tid]))[:2] or [f"t{tid}"]),
                    "summary": f"Drain3 template #{tid} (deterministic)",
                } for tid in batch_tids]
            else:
                samples_per_template = [
                    [c["text"] for c in by_tid[tid][:5]] for tid in batch_tids
                ]
                try:
                    minted_list = self.cartographer.mint_addresses_batch(samples_per_template)
                except Exception as e:
                    print(f"  [WARN] stream-novel mint failed: {e}")
                    minted_list = [{
                        "address": "/log/streamed/" + "_".join(
                            list(self._extract_template_keywords(
                                self.template_id_to_template[tid]))[:2] or [f"t{tid}"]),
                        "summary": f"Drain3 template #{tid} (fallback)",
                    } for tid in batch_tids]
                self.total_mint_calls += 1
                self.total_cartographer_usd += (
                    MINT_IN_TOK * len(batch_tids) * GEMINI_IN +
                    MINT_OUT_TOK * len(batch_tids) * GEMINI_OUT
                )
            for tid, minted in zip(batch_tids, minted_list):
                self._add_new_address(tid, minted["address"], minted["summary"],
                                       by_tid[tid])
                new_addr_records.append({
                    "address": self.addresses[-1]["address"],
                    "summary": minted["summary"],
                })

        # Edges for the newly-minted addresses (skipped if cartographer is off)
        if new_addr_records and not self.skip_cartographer:
            existing = [{
                "address": a["address"], "summary": a["summary"],
            } for a in self.addresses[:-len(new_addr_records)]]
            for batch_start in range(0, len(new_addr_records), self.mint_batch_size):
                batch_new = new_addr_records[batch_start:batch_start + self.mint_batch_size]
                try:
                    edges_per_addr = self.cartographer.mint_edges_batch(batch_new, existing)
                except Exception:
                    edges_per_addr = [[] for _ in batch_new]
                self.total_edge_calls += 1
                self.total_cartographer_usd += (
                    EDGE_IN_TOK * len(batch_new) * GEMINI_IN +
                    EDGE_OUT_TOK * len(batch_new) * GEMINI_OUT
                )
                for entry, edges in zip(batch_new, edges_per_addr):
                    for e in edges:
                        self.graph.add_edge(entry["address"], e["to"],
                                             weight=e["weight"])

        self.pending_novel = []

    # ---------- query (HOT PATH for retrieval) ---------------------------

    def query(self, q: str, k: int = 5, query_hint_svc=None, query_hint_lvl=None):
        """Per-query hot path.
        Step 1: deterministic tag extraction (no embedding).
        Step 2: filter addresses by tag match (no embedding).
        Step 3: if ≥1 candidate, rank chunks by recency × level × keyword overlap (no embedding).
                if 0 candidates → embedding fallback (one cosine).
        """
        t0 = time.perf_counter()
        svc_hint, lvl_hint = query_hint_svc, query_hint_lvl
        if svc_hint is None and lvl_hint is None:
            svc_hint, lvl_hint = extract_query_tags(q)

        query_tokens = set(_tokenize(q))

        # Step 1+2: candidate filter
        # An address is a candidate if:
        #   (a) its dominant_svc matches svc_hint (if provided), AND
        #   (b) its template_keywords overlap with query_tokens, OR
        #       its dominant_lvl matches lvl_hint
        candidates = []
        for i, a in enumerate(self.addresses):
            if svc_hint and a.get("dominant_svc") != svc_hint:
                continue
            kw_overlap = len(query_tokens & a["template_keywords"])
            lvl_match = (lvl_hint and a.get("dominant_lvl") == lvl_hint)
            if kw_overlap > 0 or lvl_match:
                candidates.append((i, kw_overlap, lvl_match))

        # If hint filter gave nothing, try just keyword overlap (relax svc)
        if not candidates and svc_hint:
            for i, a in enumerate(self.addresses):
                kw_overlap = len(query_tokens & a["template_keywords"])
                if kw_overlap > 0:
                    candidates.append((i, kw_overlap, False))

        fallback_used = False
        routed_address = None
        nbrs = []

        if candidates:
            # Step 3 deterministic: rank candidate addresses by (kw × 3 + lvl × 2),
            # take the top one, then ALSO include its graph neighbours (§8).
            candidates.sort(key=lambda x: -(x[1] * 3 + (5 if x[2] else 0)))
            routed_idx = candidates[0][0]
            routed_address = self.addresses[routed_idx]["address"]
            cand_addrs = {routed_address}
            if routed_address in self.graph:
                for n in self.graph.neighbors(routed_address):
                    w = self.graph[routed_address][n].get("weight", 0)
                    nbrs.append((n, w))
            nbrs.sort(key=lambda x: -x[1])
            for n, _ in nbrs[:MAX_NODE_DEGREE]:
                cand_addrs.add(n)
            self.queries_routed_deterministic += 1
        else:
            # Step 3 fallback: BM25 over address summaries (term-based, NO EMBEDDINGS).
            # We use summary text + template body + address string itself as the
            # "document" per address and rank by BM25 against the query tokens.
            # If even BM25 produces zero overlap → return empty (honest "I don't know").
            fallback_used = True
            self.queries_routed_fallback += 1
            best_addr = None
            best_score = 0.0
            # IDF cache: how many addresses contain each query term?
            # Cheap to compute on the fly; addresses are O(tens-hundreds).
            doc_freq: Dict[str, int] = {}
            address_docs: List[set] = []
            for a in self.addresses:
                doc_tokens = set(_tokenize((a["summary"] or "") + " " +
                                            (a["address"] or "") + " " +
                                            (a["template"] or "")))
                doc_tokens |= a["template_keywords"]
                address_docs.append(doc_tokens)
                for tk in doc_tokens & query_tokens:
                    doc_freq[tk] = doc_freq.get(tk, 0) + 1
            N = max(1, len(self.addresses))
            import math
            for i, doc_tokens in enumerate(address_docs):
                if not doc_tokens:
                    continue
                # simplified BM25 with k1=1.2, no length normalisation (docs short)
                score = 0.0
                for tk in query_tokens:
                    if tk in doc_tokens:
                        df = doc_freq.get(tk, 1)
                        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                        score += idf  # tf=1 term saturates; close enough
                if score > best_score:
                    best_score = score
                    best_addr = i
            if best_addr is None:
                # truly nothing matched — empty result, honest "no answer"
                ms = (time.perf_counter() - t0) * 1000
                return {
                    "hits": [], "candidate_set_size": 0,
                    "corpus_size": self.n_indexed, "reduction_ratio": 0,
                    "routed_address": "",
                    "expanded_neighbours": [], "fallback_used": True,
                    "fallback_kind": "empty",
                    "total_ms": ms, "cost_usd": 0.0,
                    "embed_calls_this_query": 0,
                }
            routed_address = self.addresses[best_addr]["address"]
            cand_addrs = {routed_address}
            if routed_address in self.graph:
                for n in self.graph.neighbors(routed_address):
                    w = self.graph[routed_address][n].get("weight", 0)
                    nbrs.append((n, w))
            nbrs.sort(key=lambda x: -x[1])
            for n, _ in nbrs[:MAX_NODE_DEGREE]:
                cand_addrs.add(n)

        # Step 4: deterministic reranking inside candidate addresses
        # Score each chunk by:
        #   keyword_overlap × 3  +  level_match × 2  +  recency_score
        # where recency_score = (chunk_recency / global_chunk_counter)
        # tie-broken by absolute recency.
        all_chunks = []
        for addr_name in cand_addrs:
            a_idx = next(i for i, a in enumerate(self.addresses)
                          if a["address"] == addr_name)
            a = self.addresses[a_idx]
            for ci, txt in enumerate(a["chunk_texts"]):
                tokens = set(_tokenize(txt))
                kw_score = len(query_tokens & tokens)
                lvl_score = 1 if (lvl_hint and
                                  a["chunk_levels"][ci] == lvl_hint) else 0
                recency = a["chunk_recency"][ci] / max(1, self.global_chunk_counter)
                score = kw_score * 3 + lvl_score * 2 + recency
                all_chunks.append({
                    "id": a["chunk_ids"][ci],
                    "score": score,
                    "text": txt,
                    "via_address": addr_name,
                    "level": a["chunk_levels"][ci],
                })
        all_chunks.sort(key=lambda x: -x["score"])
        hits = all_chunks[:k]
        ms = (time.perf_counter() - t0) * 1000
        n_candidates = sum(len(self.addresses[i]["chunk_ids"])
                            for i, a in enumerate(self.addresses)
                            if a["address"] in cand_addrs)
        return {
            "hits": hits,
            "candidate_set_size": min(n_candidates, 200),
            "corpus_size": self.n_indexed,
            "reduction_ratio": self.n_indexed / max(1, n_candidates),
            "routed_address": routed_address or "",
            "expanded_neighbours": [n for n, _ in nbrs[:MAX_NODE_DEGREE]],
            "fallback_used": fallback_used,
            "total_ms": ms,
            "cost_usd": (TOKENS_PER_QUERY * EMBED_USD_PER_TOKEN
                         if fallback_used else 0.0),
            "embed_calls_this_query": (1 if fallback_used else 0),
        }


if __name__ == "__main__":
    # Smoke test
    print("DrainGAC smoke test")
    from realistic_chunker import chunk_file
    chunks = chunk_file(Path(__file__).resolve().parent.parent / "logs/realistic.log")[:500]
    g = DrainGAC()
    boot = g.bootstrap(chunks[:200])
    print(f"  bootstrap: {boot['total_addresses']} addrs, "
          f"{boot['n_templates_discovered']} templates")

    stream = g.stream_ingest(chunks[200:])
    print(f"  stream:    {stream['n_known']}/{stream['n_new']} known templates "
          f"(no embed); {stream['n_novel']} novel (pending mint); "
          f"embed_calls = {stream['embed_calls']}")

    print(f"\n  Total embed calls during {len(chunks)} log lines: 0 "
          "(except inside bootstrap, NONE)")
    print(f"\n  Sample queries:")
    for q in ["failed login attempts", "database connection pool exhausted",
              "anything anomalous"]:
        r = g.query(q)
        latency = r["total_ms"]
        fb = "FALLBACK" if r["fallback_used"] else "deterministic"
        n_hits = len(r["hits"])
        print(f"    '{q[:40]}' → {fb}, {latency:.2f}ms, {n_hits} hits, "
              f"routed: {r['routed_address']}")
