"""Phase 4 charts — DocGAC vs RAG on doc corpus halves, plus the
cross-domain (logs vs docs) Tier-validation chart."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = Path(__file__).resolve().parent.parent / "reports/charts"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (10, 5.5), "figure.dpi": 110,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
})

RAG_COLOR = "#d62728"
GAC_COLOR = "#1f77b4"
ACCENT = "#2ca02c"
NEUTRAL = "#666"
TIER1 = "#1f77b4"
TIER2 = "#ff7f0e"


def chart_docgac_main():
    d = json.loads((DATA / "phase4_results.json").read_text())
    halves = {h["half_key"]: h for h in d["halves"]}
    a = halves["a"]
    b = halves["b"]
    a_rag = a["rag_score"]["aggregate_stats"]["hit_at_k_rate"]
    a_dg = a["docgac_score"]["aggregate_stats"]["hit_at_k_rate"]
    b_rag = b["rag_score"]["aggregate_stats"]["hit_at_k_rate"]
    b_dg = b["docgac_score"]["aggregate_stats"]["hit_at_k_rate"]

    fig, ax = plt.subplots(figsize=(10, 6))
    xpos = np.array([0, 1, 3, 4])
    means = [a_rag["mean"]*100, a_dg["mean"]*100,
             b_rag["mean"]*100, b_dg["mean"]*100]
    colors = [RAG_COLOR, GAC_COLOR, RAG_COLOR, GAC_COLOR]

    # error bars from min/max (per-system noise band)
    yerr_lo = [max(0, (m - lo)*100) for m, lo in
               [(a_rag['mean'], a_rag['min']), (a_dg['mean'], a_dg['min']),
                (b_rag['mean'], b_rag['min']), (b_dg['mean'], b_dg['min'])]]
    yerr_hi = [max(0, (hi - m)*100) for m, hi in
               [(a_rag['mean'], a_rag['max']), (a_dg['mean'], a_dg['max']),
                (b_rag['mean'], b_rag['max']), (b_dg['mean'], b_dg['max'])]]

    bars = ax.bar(xpos, means, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.85,
                  yerr=[yerr_lo, yerr_hi], capsize=6)
    for b_, m in zip(bars, means):
        ax.text(b_.get_x() + b_.get_width()/2, b_.get_height() - 2,
                f"{m:.1f}%", ha="center", va="top", color="white",
                fontsize=12, fontweight="bold")

    a_delta = (a_dg["mean"] - a_rag["mean"]) * 100
    b_delta = (b_dg["mean"] - b_rag["mean"]) * 100
    a_nf = a_rag["range"] * 100
    b_nf = b_rag["range"] * 100
    ax.annotate(f"Δ = {a_delta:+.1f}pp\n(noise {a_nf:.1f}pp)",
                xy=(0.5, max(a_rag["max"], a_dg["max"])*100 + 5),
                ha="center", fontsize=11, fontweight="bold",
                color=ACCENT if a_delta > a_nf else (RAG_COLOR if -a_delta > a_nf else NEUTRAL))
    ax.annotate(f"Δ = {b_delta:+.1f}pp\n(noise {b_nf:.1f}pp)",
                xy=(3.5, max(b_rag["max"], b_dg["max"])*100 + 5),
                ha="center", fontsize=11, fontweight="bold",
                color=ACCENT if b_delta > b_nf else (RAG_COLOR if -b_delta > b_nf else NEUTRAL))

    ax.annotate("doc-a (16 docs, 297 chunks)",
                xy=(0.5, max(a_rag["max"], a_dg["max"])*100 + 13),
                ha="center", fontsize=11, color="#555")
    ax.annotate("doc-b held-out (15 docs, 278 chunks)",
                xy=(3.5, max(b_rag["max"], b_dg["max"])*100 + 13),
                ha="center", fontsize=11, color="#555")

    ax.set_xticks(xpos)
    ax.set_xticklabels(["RAG", "DocGAC\n(YAKE + BM25,\nzero embeddings)",
                        "RAG", "DocGAC"], fontsize=10)
    ax.set_ylabel("Hit@5 (%) — 3-pass mean with min/max")
    ax.set_title("Phase 4 — DocGAC vs RAG on prose docs (split corpus)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, 100)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=RAG_COLOR, label="RAG (embeddings + ANN)"),
        Patch(facecolor=GAC_COLOR, label="DocGAC (YAKE keyword + BM25, zero embeddings)"),
    ], loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "G_docgac_vs_rag_main.png")
    plt.close(fig)
    print(f"  [G] docgac_vs_rag_main.png")


def chart_tier_validation():
    """The big cross-domain story: logs vs docs, validating the §1.1 Tier framework."""
    p3 = json.loads((DATA / "phase3_results.json").read_text())
    p4 = json.loads((DATA / "phase4_results.json").read_text())
    p3_c = {c["corpus_key"]: c for c in p3["corpora"]}
    p4_h = {h["half_key"]: h for h in p4["halves"]}

    log_a = p3_c["a"]["delta_hit_at_k_mean"] * 100
    log_b = p3_c["b"]["delta_hit_at_k_mean"] * 100
    doc_a = p4_h["a"]["delta_hit_at_k_mean"] * 100
    doc_b = p4_h["b"]["delta_hit_at_k_mean"] * 100

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(4)
    deltas = [log_a, log_b, doc_a, doc_b]
    colors = [TIER1, TIER1, TIER2, TIER2]
    bars = ax.bar(x, deltas, color=colors, edgecolor="white", linewidth=0.5)

    for b_, v in zip(bars, deltas):
        y_text = b_.get_height() + (1 if v >= 0 else -3)
        va = "bottom" if v >= 0 else "top"
        color = ACCENT if v > 0 else RAG_COLOR
        ax.text(b_.get_x() + b_.get_width()/2, y_text,
                f"{v:+.1f}pp", ha="center", va=va,
                color=color, fontsize=12, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["logs (a)\nDrainGAC", "logs (b)\nDrainGAC",
                        "docs (a)\nDocGAC", "docs (b)\nDocGAC"],
                       fontsize=10)
    ax.set_ylabel("GAC vs RAG hit@5 delta (pp)")
    ax.set_title("§1.1 Tier framework validated — GAC wins on Tier 1 (logs), "
                 "loses on Tier 2 (docs)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(-20, 18)

    # group labels
    ax.text(0.5, 16, "Tier 1 — templated\n(DrainGAC: regex + BM25)",
            ha="center", fontsize=11, color=TIER1, fontweight="bold")
    ax.text(2.5, 16, "Tier 2 — prose\n(DocGAC: keyword + BM25)",
            ha="center", fontsize=11, color=TIER2, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "G_tier_validation.png")
    plt.close(fig)
    print(f"  [G] tier_validation.png")


def chart_routing_breakdown_docs():
    d = json.loads((DATA / "phase4_results.json").read_text())
    halves = {h["half_key"]: h for h in d["halves"]}
    a_r = halves["a"]["docgac_query_routing"]
    b_r = halves["b"]["docgac_query_routing"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(2)
    w = 0.5
    det = [a_r["deterministic"], b_r["deterministic"]]
    bm25 = [a_r["fallback_bm25"], b_r["fallback_bm25"]]
    empty = [a_r["empty_no_match"], b_r["empty_no_match"]]

    p1 = ax.bar(x, det, w, color=ACCENT, label="deterministic (Jaccard keyword match)")
    p2 = ax.bar(x, bm25, w, bottom=det, color=TIER2,
                label="BM25 fallback (term-based)")
    p3 = ax.bar(x, empty, w, bottom=np.array(det)+np.array(bm25),
                color=RAG_COLOR, label="empty (no overlap — 'I don't know')")

    for i in range(2):
        if det[i] > 0:
            ax.text(i, det[i]/2, str(det[i]), ha="center", color="white",
                    fontsize=12, fontweight="bold")
        if bm25[i] > 0:
            ax.text(i, det[i] + bm25[i]/2, str(bm25[i]), ha="center",
                    color="white", fontsize=12, fontweight="bold")
        if empty[i] > 0:
            ax.text(i, det[i] + bm25[i] + empty[i]/2, str(empty[i]),
                    ha="center", color="white", fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["doc-a", "doc-b (held-out)"], fontsize=11)
    ax.set_ylabel("Queries (out of 42)")
    ax.set_title("DocGAC query routing on docs — zero embeddings used")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, 50)
    fig.tight_layout()
    fig.savefig(OUT / "G_routing_breakdown_docs.png")
    plt.close(fig)
    print(f"  [G] routing_breakdown_docs.png")


if __name__ == "__main__":
    print("Generating Phase 4 charts...")
    chart_docgac_main()
    chart_tier_validation()
    chart_routing_breakdown_docs()
    print(f"All → {OUT}")
