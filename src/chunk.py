"""Chunk extracted corpus into retrievable units.

Strategy: split on section markers (page/slide), then merge adjacent small
fragments until each chunk is ~target_tokens. Output is a JSONL of
{id, doc, section, text}.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

CORPUS = Path(__file__).resolve().parent.parent / "corpus"
OUT = Path(__file__).resolve().parent.parent / "data/chunks.jsonl"

# rough word target per chunk; sentence-transformers MiniLM handles ~256-512 tokens well
TARGET_WORDS = 180
MIN_WORDS = 30
MAX_WORDS = 350

SECTION_RE = re.compile(r"\n--- (page|slide) (\d+) ---\n")


def split_sections(text: str):
    """Yield (label, body) tuples split on '--- page/slide N ---' markers."""
    parts = SECTION_RE.split(text)
    # parts: [pre, 'page'|'slide', '1', body, 'page'|'slide', '2', body, ...]
    if len(parts) == 1:
        yield "section 1", parts[0]
        return
    if parts[0].strip():
        yield "header", parts[0]
    i = 1
    while i + 2 < len(parts) + 1 and i + 1 < len(parts):
        kind = parts[i]
        num = parts[i + 1]
        body = parts[i + 2] if i + 2 < len(parts) else ""
        yield f"{kind} {num}", body
        i += 3


def chunkify(label: str, body: str):
    """Merge body into chunks of ~TARGET_WORDS, respecting paragraph boundaries."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if not paras:
        return
    buf = []
    buf_words = 0
    for p in paras:
        n = len(p.split())
        if buf_words + n > MAX_WORDS and buf:
            yield label, "\n\n".join(buf)
            buf = [p]
            buf_words = n
        else:
            buf.append(p)
            buf_words += n
            if buf_words >= TARGET_WORDS:
                yield label, "\n\n".join(buf)
                buf = []
                buf_words = 0
    if buf and buf_words >= MIN_WORDS:
        yield label, "\n\n".join(buf)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    chunk_id = 0
    n_docs = 0
    by_doc = {}
    with open(OUT, "w", encoding="utf-8") as out:
        for path in sorted(CORPUS.glob("*.txt")):
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            # strip metadata header
            lines = raw.split("\n")
            body_start = 0
            for i, ln in enumerate(lines[:5]):
                if not ln.startswith("#") and ln.strip():
                    body_start = i
                    break
            text = "\n".join(lines[body_start:])
            doc_chunks = 0
            for label, body in split_sections(text):
                for sec_label, chunk_text in chunkify(label, body):
                    if len(chunk_text.split()) < MIN_WORDS:
                        continue
                    record = {
                        "id": f"c{chunk_id:05d}",
                        "doc": path.stem,
                        "section": sec_label,
                        "text": chunk_text,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    chunk_id += 1
                    doc_chunks += 1
            by_doc[path.stem] = doc_chunks
            n_docs += 1
    print(f"Wrote {chunk_id} chunks from {n_docs} docs to {OUT}")
    # top contributors
    top = sorted(by_doc.items(), key=lambda x: -x[1])[:10]
    print("\nTop docs by chunk count:")
    for doc, c in top:
        print(f"  {c:4d}  {doc}")


if __name__ == "__main__":
    main()
