"""Multi-line aware chunker for realistic logs.

Groups consecutive log lines into single LOGICAL entries when continuation
patterns are detected (Java stack traces, Postgres query plans, mobile
crash reports, etc.).

Strategy:
  1. Detect entry-start markers (line begins with timestamp, IP, JSON `{`, or
     known prefix).
  2. Any line NOT matching an entry-start is a continuation of the previous
     entry — append it to the entry.
  3. Truncate very-long multi-line entries to first 1200 chars to keep
     embedding cost bounded (the rerank text uses the head, which is where
     the discriminative content lives).

Also extracts (service, level) for hybrid routing:
  - JSON entries: parse and read `.service` and `.level`
  - Java: parse `[thread] LEVEL c.example.foo.BarController` → service=BarController, level=LEVEL
  - Apache: service="edge-cdn-or-nginx", level inferred from status code
  - k8s key=value: parse `source=...` and `level=...`
  - Postgres: service="postgres", level inferred (LOG/ERROR/WARNING)
  - Redis: service="redis"
  - Worker plain-text: parse `[worker-name]` and level keyword
  - Audit JSON: service="audit", level from JSON
  - CDN CSV-ish: service="cdn"
  - Malformed: service=None, level=None (rare; the consumer can handle)
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Entry-start patterns. Order matters — try most specific first.
ENTRY_START_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"),                   # ISO 8601 (most formats)
    re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}"),            # Java logback "2026-05-24 08:30:36.732"
    re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} UTC \["),     # Postgres "2026-05-24 08:30:36.500 UTC [...]"
    re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3} - -"),                # Apache "10.0.1.42 - -"
    re.compile(r"^\{[\"\w]"),                                              # JSON object
    re.compile(r"^time=\d{4}-\d{2}-\d{2}"),                                # k8s key=value
    re.compile(r"^\[\d{4}-\d{2}-\d{2}T"),                                  # Elasticsearch "[2026-05-24T...]"
    re.compile(r'^"\d{4}-\d{2}-\d{2}T'),                                   # CDN '"2026-05-24T..."'
    re.compile(r"^\[\d+\] \d{4}-\d{2}-\d{2}T"),                            # Redis "[7] 2026-05-24T..."
]

# Quick predicate
def is_entry_start(line: str) -> bool:
    if not line:
        return False  # blank line is treated as continuation/noise
    for p in ENTRY_START_PATTERNS:
        if p.match(line):
            return True
    return False


# ---------- service + level extraction ----------------------------------

# Java logback: "2026-05-24 08:30:36.732 [thread] LEVEL c.example.api.BarController - msg"
JAVA_PAT = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} \[([^\]]+)\] (\w+)\s+c\.example\.[\w.]+\.(\w+)")
# Apache: "ip - - [timestamp] \"METHOD path HTTP/1.1\" STATUS bytes ..."
APACHE_PAT = re.compile(r'^\d+\.\d+\.\d+\.\d+ - - \[[^\]]+\] "(\w+) (\S+) HTTP/[\d.]+" (\d+)')
# k8s: "time=... level=X source=Y ... event=Z"
KV_LEVEL = re.compile(r"\blevel=(\w+)")
KV_SOURCE = re.compile(r"\bsource=(\w+)")
KV_EVENT = re.compile(r"\bevent=(\w+)")
# Postgres: "TIMESTAMP UTC [pid] LEVEL: ..."
PG_PAT = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} UTC \[\d+\] (\w+):")
# ES: "[timestamp] [LEVEL] [class] [node]"
ES_PAT = re.compile(r"^\[\d{4}-\d{2}-\d{2}T[\d:.Z]+\] \[(\w+)\] \[([^\]]+)\]")
# Redis: "[idx] timestamp ROLE COMMAND ..."
REDIS_PAT = re.compile(r"^\[\d+\] \d{4}-\d{2}-\d{2}T[\d:Z]+ ([\*#\-]) (\w+)")
# Worker/plain text: "TIMESTAMP [service-name] LEVEL Msg..."
PLAIN_PAT = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.Z]+ \[([^\]]+)\]\s+(\w+)?")


def extract_meta(entry_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (service, level, format_tag).

    format_tag is one of: 'json', 'java', 'apache', 'k8s_kv', 'postgres',
    'es', 'redis', 'cdn', 'plain', 'malformed'. Useful for routing and
    for downstream chunkers that need to know what they're dealing with.
    """
    first = entry_text.split("\n", 1)[0]

    # JSON: parse the whole first line
    if first.startswith("{"):
        try:
            obj = json.loads(first)
            svc = obj.get("service") or obj.get("category") or "unknown"
            lvl = obj.get("level") or "INFO"
            return (str(svc), str(lvl).upper(), "json")
        except json.JSONDecodeError:
            return (None, None, "malformed")

    # Java
    m = JAVA_PAT.match(first)
    if m:
        thread, level, controller = m.group(1), m.group(2), m.group(3)
        # service = the controller's name, lowercased + service-ified
        svc = controller.lower().replace("controller", "-svc").replace("repository", "-repo")
        return (svc, level.upper(), "java")

    # Apache
    m = APACHE_PAT.match(first)
    if m:
        method, path, status = m.group(1), m.group(2), int(m.group(3))
        # treat path prefix as service hint
        if path.startswith("/api/"):
            svc = "api-gateway"
        elif path.startswith("/assets/") or path.startswith("/favicon"):
            svc = "static-server"
        else:
            svc = "nginx"
        # level from status
        if 500 <= status < 600:
            lvl = "ERROR"
        elif 400 <= status < 500:
            lvl = "WARN"
        else:
            lvl = "INFO"
        return (svc, lvl, "apache")

    # k8s key=value
    if first.startswith("time=") and "level=" in first:
        lvl = (KV_LEVEL.search(first).group(1) if KV_LEVEL.search(first) else "INFO").upper()
        src = KV_SOURCE.search(first).group(1) if KV_SOURCE.search(first) else "k8s"
        return (f"k8s-{src}", lvl, "k8s_kv")

    # Postgres
    m = PG_PAT.match(first)
    if m:
        raw_level = m.group(1).upper()
        # Postgres "LOG" → INFO, "ERROR" → ERROR, "WARNING" → WARN
        lvl = {"LOG": "INFO", "ERROR": "ERROR", "WARNING": "WARN",
               "FATAL": "ERROR", "DETAIL": "INFO", "HINT": "INFO"}.get(
            raw_level, "INFO"
        )
        return ("postgres", lvl, "postgres")

    # Elasticsearch
    m = ES_PAT.match(first)
    if m:
        return ("elasticsearch", m.group(1).upper(), "es")

    # Redis
    m = REDIS_PAT.match(first)
    if m:
        return ("redis", "INFO", "redis")

    # CDN '"2026-...",cf-pop=...'
    if first.startswith('"') and "cf-pop=" in first:
        if "BLOCK" in first or "ddos" in first:
            return ("cdn", "WARN", "cdn")
        return ("cdn", "INFO", "cdn")

    # Plain text: 2026-...T...Z [name] LEVEL msg  (workers, mobile, audit, etc.)
    m = PLAIN_PAT.match(first)
    if m:
        svc = m.group(1)
        lvl = (m.group(2) or "INFO").upper()
        # normalize level (CRASH → ERROR)
        if lvl == "CRASH":
            lvl = "ERROR"
        return (svc, lvl, "plain")

    return (None, None, "malformed")


# ---------- chunking ----------------------------------------------------

# Stack-trace continuations look like "\t at ..." or "  Caused by: ..." or
# (for Postgres) DETAIL lines that ARE entry-starts but conceptually belong
# to the previous LOG entry. We treat continuation as: NOT an entry-start.
#
# Special case for Postgres: a DETAIL/HINT line IS an entry-start (matches
# the timestamp pattern) but logically belongs to the previous LOG entry.
# We detect this and merge.

PG_DETAIL_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} UTC \[\d+\] (DETAIL|HINT|CONTEXT|STATEMENT):")

MAX_ENTRY_CHARS = 1200


def chunk_file(path: Path) -> List[Dict[str, Any]]:
    """Read the realistic log file and return list of LOGICAL entry chunks."""
    entries: List[Dict[str, Any]] = []
    current_lines: List[str] = []
    line_no = 0
    entry_first_line_no = 0

    def flush():
        if not current_lines:
            return
        text = "\n".join(current_lines)
        svc, lvl, fmt = extract_meta(text)
        # truncate very long multi-line entries
        truncated = False
        if len(text) > MAX_ENTRY_CHARS:
            text = text[:MAX_ENTRY_CHARS] + " ...(truncated)"
            truncated = True
        entries.append({
            "id": f"e{entry_first_line_no:07d}",
            "doc": "realistic.log",
            "section": f"line {entry_first_line_no}",
            "text": text,
            "svc": svc,
            "level": lvl,
            "format": fmt,
            "n_raw_lines": len(current_lines),
            "truncated": truncated,
        })

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_no += 1
            stripped = line.rstrip("\n")
            # Postgres DETAIL/HINT/CONTEXT continuation special-case
            if PG_DETAIL_RE.match(stripped) and current_lines and \
               extract_meta("\n".join(current_lines))[2] == "postgres":
                current_lines.append(stripped)
                continue
            if is_entry_start(stripped):
                flush()
                current_lines = [stripped]
                entry_first_line_no = line_no
            else:
                if not current_lines:
                    # orphan continuation at file start; treat as its own (malformed) entry
                    current_lines = [stripped]
                    entry_first_line_no = line_no
                else:
                    current_lines.append(stripped)
        flush()
    return entries


def summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stats about the chunked corpus — for sanity-checking."""
    from collections import Counter
    fmts = Counter(e["format"] for e in entries)
    svcs = Counter(e["svc"] for e in entries if e["svc"])
    lvls = Counter(e["level"] for e in entries if e["level"])
    multi = sum(1 for e in entries if e["n_raw_lines"] > 1)
    truncated = sum(1 for e in entries if e["truncated"])
    malformed = sum(1 for e in entries if e["format"] == "malformed")
    total_raw_lines = sum(e["n_raw_lines"] for e in entries)
    return {
        "n_entries": len(entries),
        "n_raw_lines": total_raw_lines,
        "n_multiline_entries": multi,
        "n_truncated_entries": truncated,
        "n_malformed_entries": malformed,
        "by_format": dict(fmts.most_common()),
        "by_service_top10": dict(svcs.most_common(10)),
        "by_level": dict(lvls.most_common()),
        "avg_lines_per_entry": total_raw_lines / max(1, len(entries)),
    }


if __name__ == "__main__":
    p = Path(__file__).resolve().parent.parent / "logs/realistic.log"
    entries = chunk_file(p)
    s = summary(entries)
    print(json.dumps(s, indent=2))
    # Show a few multi-line entries
    print("\n--- sample multi-line entries ---")
    multi = [e for e in entries if e["n_raw_lines"] > 1][:3]
    for e in multi:
        print(f"\n[{e['id']}] svc={e['svc']} level={e['level']} "
              f"format={e['format']} n_raw_lines={e['n_raw_lines']}")
        print(e["text"][:400] + (" ..." if len(e["text"]) > 400 else ""))
