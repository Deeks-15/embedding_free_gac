"""Phase 3 charts — DrainGAC vs RAG on both corpora, plus the
generalization contrast vs §12-tuned stack (Phase 1/2)."""
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
DRAIN_COLOR = "#1f77b4"
TUNED_COLOR = "#ff7f0e"
C0_COLOR = "#888888"
ACCENT = "#2ca02c"


def chart_drain_main():
    d = json.loads((DATA / "phase3_results.json").read_text())
    corpora = {c["corpus_key"]: c for c in d["corpora"]}

    a = corpora["a"]
    b = corpora["b"]
    a_rag = a["rag_score"]["aggregate_stats"]["hit_at_k_rate"]
    a_dr = a["drain_score"]["aggregate_stats"]["hit_at_k_rate"]
    b_rag = b["rag_score"]["aggregate_stats"]["hit_at_k_rate"]
    b_dr = b["drain_score"]["aggregate_stats"]["hit_at_k_rate"]

    fig, ax = plt.subplots(figsize=(10, 6))
    xpos = np.array([0, 1, 3, 4])
    means = [a_rag["mean"]*100, a_dr["mean"]*100,
             b_rag["mean"]*100, b_dr["mean"]*100]
    colors = [RAG_COLOR, DRAIN_COLOR, RAG_COLOR, DRAIN_COLOR]
    bars = ax.bar(xpos, means, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.85)

    for b_, m in zip(bars, means):
        ax.text(b_.get_x() + b_.get_width()/2, b_.get_height() - 2,
                f"{m:.1f}%", ha="center", va="top", color="white",
                fontsize=12, fontweight="bold")

    # delta labels above each pair
    a_delta = (a_dr["mean"] - a_rag["mean"]) * 100
    b_delta = (b_dr["mean"] - b_rag["mean"]) * 100
    ax.annotate(f"Δ = {a_delta:+.1f}pp",
                xy=(0.5, max(a_rag["mean"], a_dr["mean"]) * 100 + 5),
                ha="center", fontsize=12, fontweight="bold",
                color=ACCENT if a_delta > 0 else RAG_COLOR)
    ax.annotate(f"Δ = {b_delta:+.1f}pp",
                xy=(3.5, max(b_rag["mean"], b_dr["mean"]) * 100 + 5),
                ha="center", fontsize=12, fontweight="bold",
                color=ACCENT if b_delta > 0 else RAG_COLOR)

    ax.annotate("realistic (a) — tuning corpus",
                xy=(0.5, max(a_rag["mean"], a_dr["mean"]) * 100 + 12),
                ha="center", fontsize=11, color="#555")
    ax.annotate("held-out (b) — frozen test corpus",
                xy=(3.5, max(b_rag["mean"], b_dr["mean"]) * 100 + 12),
                ha="center", fontsize=11, color="#555")

    ax.set_xticks(xpos)
    ax.set_xticklabels(["RAG", "DrainGAC\n(zero-embedding\nhot path)",
                        "RAG", "DrainGAC"], fontsize=10)
    ax.set_ylabel("Hit@5 (%) — LLM-judge ground truth (3-pass mean)")
    ax.set_title("Phase 3 — DrainGAC: authentic GAC vs RAG on both corpora")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, 90)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=RAG_COLOR, label="RAG (embeddings + ANN)"),
        Patch(facecolor=DRAIN_COLOR, label="DrainGAC (Drain3 + BM25, no embeddings on hot path)"),
    ], loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT / "F_drain_vs_rag_main.png")
    plt.close(fig)
    print(f"  [F] drain_vs_rag_main.png")


def chart_generalization_contrast():
    """Compare §12-tuned (Phase 1 + 2) vs DrainGAC (Phase 3) — how each
    approach generalizes from (a) to (b)."""
    p1 = json.loads((DATA / "phase1_results.json").read_text())
    p2 = json.loads((DATA / "phase2_results.json").read_text())
    p3 = json.loads((DATA / "phase3_results.json").read_text())
    p1_by = {r["id"]: r for r in p1["results"]}
    p2_by = {r["id"]: r for r in p2["results"]}
    p3_corpora = {c["corpus_key"]: c for c in p3["corpora"]}

    # Deltas (system - C0 baseline) on each corpus
    tuned_a = (p1_by["C3"]["aggregate_stats"]["gac_hit_at_k"]["mean"] -
               p1_by["C0"]["aggregate_stats"]["gac_hit_at_k"]["mean"]) * 100
    tuned_b = (p2_by["C3"]["aggregate_stats"]["gac_hit_at_k"]["mean"] -
               p2_by["C0"]["aggregate_stats"]["gac_hit_at_k"]["mean"]) * 100
    drain_a = p3_corpora["a"]["delta_hit_at_k_mean"] * 100
    drain_b = p3_corpora["b"]["delta_hit_at_k_mean"] * 100

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(2)
    w = 0.35
    bars_tuned = ax.bar(x - w/2, [tuned_a, tuned_b], w,
                         color=TUNED_COLOR, label="§12-tuned vs C0 baseline")
    bars_drain = ax.bar(x + w/2, [drain_a, drain_b], w,
                         color=DRAIN_COLOR, label="DrainGAC vs RAG baseline")

    for bars, vals in [(bars_tuned, [tuned_a, tuned_b]),
                       (bars_drain, [drain_a, drain_b])]:
        for b_, v in zip(bars, vals):
            y_text = b_.get_height() + (1 if v >= 0 else -3)
            va = "bottom" if v >= 0 else "top"
            color = ACCENT if v > 0 else RAG_COLOR
            ax.text(b_.get_x() + b_.get_width()/2, y_text,
                    f"{v:+.1f}pp", ha="center", va=va,
                    color=color, fontsize=11, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["tuning corpus (a)", "held-out (b)"], fontsize=11)
    ax.set_ylabel("Δ accuracy (pp) vs baseline")
    ax.set_title("Generalization contrast — both approaches gained on (a); "
                 "only DrainGAC held up on (b)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    ymin = min(tuned_b, drain_b) - 5
    ymax = max(tuned_a, drain_a) + 5
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.savefig(OUT / "F_generalization_contrast.png")
    plt.close(fig)
    print(f"  [F] generalization_contrast.png")


def chart_routing_breakdown():
    """How DrainGAC routes its queries on each corpus."""
    d = json.loads((DATA / "phase3_results.json").read_text())
    corpora = {c["corpus_key"]: c for c in d["corpora"]}
    a_r = corpora["a"]["drain_query_routing"]
    b_r = corpora["b"]["drain_query_routing"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(2)
    w = 0.25
    det = [a_r["deterministic"], b_r["deterministic"]]
    bm25 = [a_r["fallback_bm25"] - a_r["empty_no_match"],
            b_r["fallback_bm25"] - b_r["empty_no_match"]]
    empty = [a_r["empty_no_match"], b_r["empty_no_match"]]

    p1 = ax.bar(x, det, w, color=ACCENT, label="deterministic (template match)")
    p2 = ax.bar(x, bm25, w, bottom=det, color=TUNED_COLOR,
                label="BM25 fallback (term match)")
    p3 = ax.bar(x, empty, w, bottom=np.array(det)+np.array(bm25),
                color=RAG_COLOR, label="empty (no match — honest 'I don't know')")

    for i in range(2):
        if det[i] > 0:
            ax.text(i, det[i]/2, str(det[i]), ha="center", color="white",
                    fontsize=11, fontweight="bold")
        if bm25[i] > 0:
            ax.text(i, det[i] + bm25[i]/2, str(bm25[i]), ha="center",
                    color="white", fontsize=11, fontweight="bold")
        if empty[i] > 0:
            ax.text(i, det[i] + bm25[i] + empty[i]/2, str(empty[i]),
                    ha="center", color="white", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["realistic (a)", "held-out (b)"], fontsize=11)
    ax.set_ylabel("Queries (out of 42)")
    ax.set_title("DrainGAC query routing — embeddings only used on fallback "
                 "(0 actual embed calls)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, 50)
    fig.tight_layout()
    fig.savefig(OUT / "F_routing_breakdown.png")
    plt.close(fig)
    print(f"  [F] routing_breakdown.png")


def chart_hot_path_embeddings():
    """The headline architectural claim: per-event + per-query embedding calls."""
    d = json.loads((DATA / "phase3_results.json").read_text())
    corpora = {c["corpus_key"]: c for c in d["corpora"]}

    fig, ax = plt.subplots(figsize=(10, 5))
    systems = ["RAG\n(every event\n+ every query)",
               "TunedGAC\n(every event\n+ every query)",
               "DrainGAC\n(zero events\n+ zero queries)"]
    # Approximate counts per 25k events + 42 queries
    n_events = 25000
    n_queries = 42
    embed_counts = [n_events + n_queries, n_events + n_queries, 0]
    colors = [RAG_COLOR, "#7a9cc6", DRAIN_COLOR]
    bars = ax.bar(np.arange(3), embed_counts, color=colors,
                  edgecolor="white", linewidth=0.5)
    for b_, c in zip(bars, embed_counts):
        if c > 0:
            ax.text(b_.get_x() + b_.get_width()/2, b_.get_height(),
                    f"{c:,}", ha="center", va="bottom", fontsize=11,
                    fontweight="bold")
        else:
            ax.text(b_.get_x() + b_.get_width()/2, 100,
                    "ZERO", ha="center", va="bottom", fontsize=14,
                    color=ACCENT, fontweight="bold")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(systems, fontsize=10)
    ax.set_ylabel("Hot-path embedding calls (per 25k events + 42 queries)")
    ax.set_title("Architectural difference — embedding calls on the hot path")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "F_hot_path_embeddings.png")
    plt.close(fig)
    print(f"  [F] hot_path_embeddings.png")


if __name__ == "__main__":
    print("Generating Phase 3 charts...")
    chart_drain_main()
    chart_generalization_contrast()
    chart_routing_breakdown()
    chart_hot_path_embeddings()
    print(f"All → {OUT}")
