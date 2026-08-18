"""Real statistical significance testing on the paired absolute-score data
(D0 vs D2, and D0 vs D1) already collected in abs_score_progress_*.jsonl,
plus a combined 3-way (D0/D1/D2) absolute-score plot.

Uses a paired Wilcoxon signed-rank test (non-parametric, appropriate for
bounded integer LLM scores rather than assuming normality) on the
per-paper mean injected score (averaged across the 5 families) for each
condition pair, since the same 50 papers were reviewed under both
conditions - a paired design.

Usage (inside the container):
    python /workspace/scripts/analysis/significance_and_3way_plot.py
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
})
import matplotlib.pyplot as plt
from scipy import stats

EVAL_DIR = Path("/workspace/outputs/eval")
FIG_DIR = Path("/workspace/outputs/figures")
FAMILIES = ["F1", "F2", "F3", "F4", "F5"]
FAMILY_LABELS = ["F1\nDirect\noverride", "F2\nStealth/\nrendering",
                  "F3\nReviewer\nimpersonation", "F4\nFabricated\nauthority",
                  "F5\nSycophancy"]


def load_progress(cond):
    path = EVAL_DIR / f"abs_score_progress_{cond}.jsonl"
    per_paper_mean = {}
    per_family = {f: [] for f in FAMILIES}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            scores = [v["injected"] for v in rec["per_family"].values() if v["injected"] is not None]
            if scores:
                per_paper_mean[rec["aid"]] = statistics.mean(scores)
            for fam, v in rec["per_family"].items():
                if v["injected"] is not None:
                    per_family[fam].append(v["injected"])
    return per_paper_mean, per_family


def main():
    d0_mean, d0_fam = load_progress("D0")
    d1_mean, d1_fam = load_progress("D1")
    d2_mean, d2_fam = load_progress("D2")

    print(f"D0: {len(d0_mean)} papers, D1: {len(d1_mean)} papers, D2: {len(d2_mean)} papers")

    common_ids_d2 = sorted(set(d0_mean) & set(d2_mean))
    common_ids_d1 = sorted(set(d0_mean) & set(d1_mean))

    d0_paired_d2 = [d0_mean[a] for a in common_ids_d2]
    d2_paired = [d2_mean[a] for a in common_ids_d2]
    d0_paired_d1 = [d0_mean[a] for a in common_ids_d1]
    d1_paired = [d1_mean[a] for a in common_ids_d1]

    print(f"\n=== Paired significance test: D0 vs D2 (n={len(common_ids_d2)}) ===")
    w_stat, p_val = stats.wilcoxon(d0_paired_d2, d2_paired)
    print(f"Wilcoxon signed-rank: statistic={w_stat:.2f}, p={p_val:.6f}")
    print(f"Mean D0 injected: {statistics.mean(d0_paired_d2):.3f}, Mean D2 injected: {statistics.mean(d2_paired):.3f}")
    print(f"Significant at alpha=0.05: {p_val < 0.05}")

    print(f"\n=== Paired significance test: D0 vs D1 (n={len(common_ids_d1)}) ===")
    w_stat1, p_val1 = stats.wilcoxon(d0_paired_d1, d1_paired)
    print(f"Wilcoxon signed-rank: statistic={w_stat1:.2f}, p={p_val1:.6f}")
    print(f"Mean D0 injected: {statistics.mean(d0_paired_d1):.3f}, Mean D1 injected: {statistics.mean(d1_paired):.3f}")
    print(f"Significant at alpha=0.05: {p_val1 < 0.05}")

    # --- 3-way plot ---
    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(FAMILIES))
    width = 0.26
    d0_inj = [statistics.mean(d0_fam[f]) for f in FAMILIES]
    d1_inj = [statistics.mean(d1_fam[f]) for f in FAMILIES]
    d2_inj = [statistics.mean(d2_fam[f]) for f in FAMILIES]

    ax.bar([i - width for i in x], d0_inj, width, label="D0 (undefended)", color="tab:red", alpha=0.85)
    ax.bar([i for i in x], d1_inj, width, label="D1 (prompting defense)", color="tab:orange", alpha=0.85)
    ax.bar([i + width for i in x], d2_inj, width, label="D2 (trained, DPO)", color="tab:green", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(FAMILY_LABELS)
    ax.set_ylabel("Mean absolute score under attack")
    ax.set_title(f"Absolute injected-paper score, all conditions (n=50 papers each)\n"
                  f"D0 vs D2: Wilcoxon p={p_val:.4f}  |  D0 vs D1: Wilcoxon p={p_val1:.4f}")
    ax.axvline(2.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path = FIG_DIR / "three_way_absolute_score.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
