"""Review content/style visualization: word count and critique-theme
frequency per condition, from the real stored text samples.

Usage (inside the container):
    python /workspace/scripts/analysis/plot_review_style.py
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

EVAL_DIR = Path("/workspace/outputs/eval")
FIG_DIR = Path("/workspace/outputs/figures")

THEME_KEYWORDS = {
    "Empirical/\nexperimental\nrigor": ["empirical", "experiment", "ablation", "validation"],
    "Generalizability": ["generaliz", "broader applicability"],
    "Practical\nrelevance": ["practical"],
    "Scalability/\nefficiency": ["scalab", "efficien", "computational complexity"],
    "Baseline\ncomparison": ["baseline", "comparison", "state-of-the-art", "existing method"],
}
CONDITIONS = ["D0", "D1", "D2"]
COLORS = {"D0": "tab:red", "D1": "tab:orange", "D2": "tab:green"}


def load(cond):
    path = EVAL_DIR / f"{cond}_Qwen_Qwen2.5-14B-Instruct_progress.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            out.append(rec["result"].get("clean_review_text_sample", ""))
    return out


def main():
    data = {c: load(c) for c in CONDITIONS}
    for c in CONDITIONS:
        print(f"{c}: {len(data[c])} reviews")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- panel 1: word count distribution (box plot) ---
    lengths = [[len(t.split()) for t in data[c] if t] for c in CONDITIONS]
    bp = ax1.boxplot(lengths, tick_labels=CONDITIONS, patch_artist=True)
    for patch, c in zip(bp["boxes"], CONDITIONS):
        patch.set_facecolor(COLORS[c])
        patch.set_alpha(0.7)
    ax1.set_ylabel("Review length (words)")
    ax1.set_title("Review length distribution")
    ax1.grid(axis="y", alpha=0.3)

    # --- panel 2: theme mention frequency, grouped bars ---
    x = range(len(THEME_KEYWORDS))
    width = 0.26
    for i, c in enumerate(CONDITIONS):
        texts = data[c]
        freqs = []
        for theme, kws in THEME_KEYWORDS.items():
            hits = sum(1 for t in texts if any(kw in t.lower() for kw in kws))
            freqs.append(100 * hits / len(texts) if texts else 0)
        offset = (i - 1) * width
        ax2.bar([xi + offset for xi in x], freqs, width, label=c, color=COLORS[c], alpha=0.85)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(list(THEME_KEYWORDS.keys()))
    ax2.set_ylabel("% of reviews mentioning theme")
    ax2.set_title("Critique-theme frequency by condition")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Review content/style comparison across defense conditions", y=1.02)
    fig.tight_layout()
    out_path = FIG_DIR / "review_style_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
