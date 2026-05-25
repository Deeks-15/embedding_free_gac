"""RAG baseline: sentence-transformers + ChromaDB.

Builds an embedding index over the chunks, and exposes a `RAGSystem` class
with .index() and .query() methods used by the head-to-head harness.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

CHUNKS = Path(__file__).resolve().parent.parent / "data/chunks.jsonl"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "data/chroma_db"
EMB_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, CPU-friendly
COLL_NAME = "rag_chunks"


def load_chunks() -> List[Dict[str, Any]]:
    chunks = []
    with open(CHUNKS) as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


class RAGSystem:
    def __init__(self, model_name: str = EMB_MODEL, persist_dir: Path = CHROMA_DIR):
        self.model_name = model_name
        self.persist_dir = persist_dir
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.coll = None
        self.index_build_time_s = 0.0
        self.embed_time_s = 0.0
        self.n_indexed = 0

    def build_index(self, chunks: List[Dict[str, Any]], rebuild: bool = False):
        if rebuild:
            try:
                self.client.delete_collection(COLL_NAME)
            except Exception:
                pass
        try:
            self.coll = self.client.get_collection(COLL_NAME)
            if self.coll.count() == len(chunks) and not rebuild:
                self.n_indexed = self.coll.count()
                print(f"[RAG] reusing existing index ({self.n_indexed} chunks)")
                return
            self.client.delete_collection(COLL_NAME)
        except Exception:
            pass

        self.coll = self.client.create_collection(
            COLL_NAME, metadata={"hnsw:space": "cosine"}
        )

        t0 = time.perf_counter()
        texts = [c["text"] for c in chunks]
        embs = self.model.encode(
            texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.embed_time_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        # Chroma has insertion limits; batch
        B = 500
        for i in range(0, len(chunks), B):
            batch = chunks[i:i + B]
            self.coll.add(
                ids=[c["id"] for c in batch],
                embeddings=embs[i:i + B].tolist(),
                documents=[c["text"] for c in batch],
                metadatas=[{"doc": c["doc"], "section": c["section"]} for c in batch],
            )
        self.index_build_time_s = time.perf_counter() - t1
        self.n_indexed = len(chunks)
        print(
            f"[RAG] indexed {self.n_indexed} chunks "
            f"(embed {self.embed_time_s:.2f}s, write {self.index_build_time_s:.2f}s)"
        )

    def query(self, q: str, k: int = 5) -> Dict[str, Any]:
        t0 = time.perf_counter()
        q_emb = self.model.encode([q], normalize_embeddings=True)[0]
        embed_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        res = self.coll.query(
            query_embeddings=[q_emb.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        ann_ms = (time.perf_counter() - t1) * 1000

        hits = []
        for i in range(len(res["ids"][0])):
            hits.append({
                "id": res["ids"][0][i],
                "doc": res["metadatas"][0][i]["doc"],
                "section": res["metadatas"][0][i]["section"],
                "score": 1 - res["distances"][0][i],  # cosine sim
                "text": res["documents"][0][i],
            })
        return {
            "hits": hits,
            "candidate_set_size": self.n_indexed,  # ANN scans logically over all
            "embed_ms": embed_ms,
            "ann_ms": ann_ms,
            "total_ms": embed_ms + ann_ms,
            "explanation": (
                f"ANN over {self.n_indexed} embeddings; "
                f"top-{k} by cosine similarity to query embedding"
            ),
        }


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"loaded {len(chunks)} chunks")
    rag = RAGSystem()
    rag.build_index(chunks)
    # smoke test
    r = rag.query("What is product X?")
    print(f"\nQuery: 'What is product X?'   ({r['total_ms']:.1f}ms)")
    for h in r["hits"]:
        snippet = h["text"][:140].replace("\n", " ")
        print(f"  {h['score']:.3f}  [{h['doc']}/{h['section']}]  {snippet}…")
