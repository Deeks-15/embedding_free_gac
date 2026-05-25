# corpus/ — bring your own documents

This folder is intentionally empty in the public repo.

The original DocGAC benchmark ran against a proprietary corpus (a private
set of GenAI-practice slide decks and briefs, ~31 documents). That content
cannot be republished, so this folder ships empty.

To reproduce the DocGAC vs RAG comparison on your own data:

1. Drop your source documents (PDF, PPTX, HTML, MD, TXT) into a folder.
2. Point the extractor at it and run:
   ```bash
   DOC_SOURCE_DIR=/path/to/your/docs python src/extract.py
   ```
   Extracted `.txt` files will land here, in `corpus/`.
3. Chunk them (the chunker reads from `corpus/` and writes
   `data/chunks.jsonl`):
   ```bash
   python src/chunk.py
   ```
4. Edit the `EVAL_QUERIES` list in `src/phase4_docs.py` to replace the
   bundled eval queries with ones grounded in your corpus (the bundled
   queries reference the original docs by name and will mostly miss
   against unrelated content).
5. Re-run the head-to-head:
   ```bash
   python src/phase4_docs.py
   ```

The DrainGAC (logs) benchmark in `src/phase3_drain.py` does **not** depend
on this folder — it uses the synthetic logs in `logs/` and ships runnable
out-of-the-box.
