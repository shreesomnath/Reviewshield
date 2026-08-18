"""Additional statistical rigor pass on top of the existing absolute-score
comparison: effect sizes (not just p-values), per-paper scatter (not just
means), bootstrap CIs, and a cross-model/condition heatmap.

Uses only already-computed score data (abs_score_progress_*.jsonl for
Qwen, D*_meta-llama_..._progress.jsonl for Llama, D*_gemini-..._progress
.jsonl for Gemini) - no new generation needed.

Usage (inside the container):
    python /workspace/scripts/analysis/additional_stats.py
"""
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
})
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

EVAL_DIR = Path("/workspace/outputs/eval")
FIG_DIR = Path("/workspace/outputs/figures")
FAMILIES = ["F1", "F2", "F3", "F4", "F5"]


def rank_biserial(clean, injected):
    """Matched-pairs rank-biserial effect size for Wilcoxon signed-rank:
    (n_pos - n_neg) / n_nonzero, where pos/neg is the sign of
    (injected - clean) per pair. +1 = injection always raises the score,
    -1 = always lowers it, 0 = no consistent direction."""
    diffs = [i - c for c, i in zip(clean, injected)]
    nonzero = [d for d in diffs if d != 0]
    if not nonzero:
        return 0.0
    n_pos = sum(1 for d in nonzero if d > 0)
    n_neg = sum(1 for d in nonzero if d < 0)
    return (n_pos - n_neg) / len(nonzero)


def bootstrap_ci(values, n_boot=2000, ci=0.95):
    if len(values) < 2:
        return (None, None)
    rng = random.Random(0)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((1 - ci) / 2 * n_boot)
    hi_idx = int((1 + ci) / 2 * n_boot)
    return (means[lo_idx], means[hi_idx])


# ---------------------------------------------------------------------------
# Load per-paper, per-family (clean, injected) pairs
# ---------------------------------------------------------------------------

def load_qwen_pairs(cond):
    """From abs_score_progress_{cond}.jsonl: one clean/injected pair per
    paper per family, already deduplicated (single generation per family)."""
    path = EVAL_DIR / f"abs_score_progress_{cond}.jsonl"
    pairs = {f: [] for f in FAMILIES}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            for fam, d in rec["per_family"].items():
                if fam in pairs:
                    pairs[fam].append((d["clean"], d["injected"]))
    return pairs


def load_local_pairs(cond, model_id):
    """From D{cond}_{model_id}_progress.jsonl: filter to injection_only
    variant so each paper contributes exactly one pair per family (avoids
    conflating the injection_plus_flaw variant, a different manipulation)."""
    path = EVAL_DIR / f"D{cond[1:]}_{model_id.replace('/', '_')}_progress.jsonl" if cond.startswith("D") else None
    path = EVAL_DIR / f"{cond}_{model_id.replace('/', '_')}_progress.jsonl"
    pairs = {f: [] for f in FAMILIES}
    if not path.exists():
        return pairs
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            r = rec["result"]
            if r.get("skipped"):
                continue
            for item in r["injected"]:
                if item["variant_type"] != "injection_only":
                    continue
                fam = item["family"]
                if fam in pairs and item["clean_score"] is not None and item["injected_score"] is not None:
                    pairs[fam].append((item["clean_score"], item["injected_score"]))
    return pairs


def main():
    qwen_pairs = {"D0": load_qwen_pairs("D0"), "D1": load_qwen_pairs("D1"), "D2": load_qwen_pairs("D2"),
                  "D3": load_local_pairs("D3", "Qwen/Qwen2.5-14B-Instruct")}
    llama_pairs = {"D0": load_local_pairs("D0", "meta-llama/Llama-3.1-8B-Instruct"),
                   "D1": load_local_pairs("D1", "meta-llama/Llama-3.1-8B-Instruct"),
                   "D2": load_local_pairs("D2", "meta-llama/Llama-3.1-8B-Instruct")}
    gemini_pairs = {"D0": load_local_pairs("D0", "gemini-flash-lite-latest"),
                     "D1": load_local_pairs("D1", "gemini-flash-lite-latest")}

    # ---------------- effect sizes + significance ----------------
    print("=== Wilcoxon signed-rank + rank-biserial effect size, D0 (undefended), per family ===")
    print(f"{'Family':<8}{'Model':<10}{'n':<5}{'p-value':<12}{'effect r':<10}{'meaning'}")
    for model_name, pairs_by_cond in [("Qwen", qwen_pairs), ("Llama", llama_pairs)]:
        for fam in FAMILIES:
            pairs = pairs_by_cond["D0"][fam]
            if len(pairs) < 3:
                continue
            clean = [c for c, i in pairs]
            injected = [i for c, i in pairs]
            try:
                stat, p = wilcoxon([i - c for c, i in pairs])
            except ValueError:
                p = float("nan")
            r = rank_biserial(clean, injected)
            meaning = "strong, consistent inflation" if r > 0.5 else ("mild/mixed" if r > 0.1 else "~no consistent effect")
            print(f"{fam:<8}{model_name:<10}{len(pairs):<5}{p:<12.2e}{r:<10.2f}{meaning}")

    # ---------------- bootstrap CIs ----------------
    print("\n=== Bootstrap 95% CI on mean injected score, per family ===")
    print(f"{'Family':<8}{'Model/Cond':<16}{'mean':<8}{'95% CI'}")
    for label, pairs_by_cond in [("Qwen", qwen_pairs), ("Llama", llama_pairs), ("Gemini", gemini_pairs)]:
        for cond, pairs_by_fam in pairs_by_cond.items():
            for fam in FAMILIES:
                pairs = pairs_by_fam[fam]
                if len(pairs) < 3:
                    continue
                injected_scores = [i for c, i in pairs]
                mean = sum(injected_scores) / len(injected_scores)
                lo, hi = bootstrap_ci(injected_scores)
                print(f"{fam:<8}{label + ' ' + cond:<16}{mean:<8.2f}[{lo:.2f}, {hi:.2f}]")

    # ---------------- per-paper scatter: Qwen + Llama, D0 vs D2 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    colors = {"F1": "tab:red", "F2": "tab:orange", "F3": "tab:green", "F4": "tab:blue", "F5": "tab:purple"}
    rng = random.Random(0)

    def jitter(vals, spread=0.18):
        return [v + rng.uniform(-spread, spread) for v in vals]

    for ax, (model_name, pairs_by_cond) in zip(axes, [("Qwen2.5-14B", qwen_pairs), ("Llama-3.1-8B", llama_pairs)]):
        for fam in FAMILIES:
            d0 = pairs_by_cond["D0"][fam]
            d2 = pairs_by_cond["D2"][fam]
            # Scores are integers, so raw (clean, injected) pairs collapse
            # onto a handful of grid points across n=50 papers - jitter is
            # purely visual (doesn't change the underlying integer scores)
            # so the true sample density at each point is actually visible
            # instead of misleadingly looking like a few dozen points.
            ax.scatter(jitter([c for c, i in d0]), jitter([i for c, i in d0]), marker="o", alpha=0.35,
                       color=colors[fam], label=f"{fam} D0 (undefended)" if fam == "F1" else None, s=22)
            ax.scatter(jitter([c for c, i in d2]), jitter([i for c, i in d2]), marker="^", alpha=0.35,
                       color=colors[fam], label=f"{fam} D2 (trained)" if fam == "F1" else None, s=22)
        ax.plot([0, 10], [0, 10], color="gray", linestyle="--", linewidth=1, alpha=0.6, label="y=x (no change)")
        ax.set_xlabel("Clean-paper score")
        ax.set_ylabel("Injected-paper score")
        ax.set_title(f"{model_name}: per-paper score shift\n(circle=D0, triangle=D2, color=family)")
        ax.set_xlim(0, 10.5); ax.set_ylim(0, 10.5)
        handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[f], label=f, markersize=8) for f in FAMILIES]
        handles.append(plt.Line2D([0], [0], color="gray", linestyle="--", label="y=x"))
        ax.legend(handles=handles, fontsize=7, loc="upper left")
    fig.suptitle("Per-paper clean vs. injected score, jittered for visibility (n=50/family/condition)\n(points above the diagonal = score inflated by the attack)", y=1.04)
    fig.tight_layout()
    out1 = FIG_DIR / "per_paper_scatter_d0_d2.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"\nWrote {out1}")

    # ---------------- heatmap: model x condition x family ----------------
    rows = []
    row_labels = []
    combos = [
        ("Qwen D0", qwen_pairs["D0"]), ("Qwen D1", qwen_pairs["D1"]), ("Qwen D2", qwen_pairs["D2"]), ("Qwen D3", qwen_pairs["D3"]),
        ("Llama D0", llama_pairs["D0"]), ("Llama D1", llama_pairs["D1"]), ("Llama D2", llama_pairs["D2"]),
        ("Gemini D0", gemini_pairs["D0"]), ("Gemini D1", gemini_pairs["D1"]),
    ]
    for label, pairs_by_fam in combos:
        row = []
        has_data = False
        for fam in FAMILIES:
            pairs = pairs_by_fam[fam]
            if pairs:
                has_data = True
                row.append(sum(i for c, i in pairs) / len(pairs))
            else:
                row.append(np.nan)
        if has_data:
            rows.append(row)
            row_labels.append(label)

    fig2, ax2 = plt.subplots(figsize=(8, 0.6 * len(rows) + 2))
    data = np.array(rows)
    im = ax2.imshow(data, cmap="RdYlGn_r", vmin=5, vmax=10, aspect="auto")
    ax2.set_xticks(range(len(FAMILIES))); ax2.set_xticklabels(FAMILIES)
    ax2.set_yticks(range(len(row_labels))); ax2.set_yticklabels(row_labels)
    for r in range(len(rows)):
        for c in range(len(FAMILIES)):
            v = data[r, c]
            if not np.isnan(v):
                ax2.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=9)
            else:
                ax2.text(c, r, "n/a", ha="center", va="center", fontsize=8, color="gray")
    ax2.set_title("Mean injected score by model x condition x family\n(darker red = more inflated/less robust)")
    fig2.colorbar(im, ax=ax2, label="Mean score under attack")
    fig2.tight_layout()
    out2 = FIG_DIR / "cross_model_heatmap.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
