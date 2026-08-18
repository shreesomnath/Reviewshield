"""D0 vs D2 comparison for Llama-3.1-8B: absolute injected score AND
decision-flip ASR side by side, since the two metrics tell a genuinely
mixed story (lower scores under attack, but more decision flips) that
needs to be shown together, not cherry-picked.

Usage (inside the container):
    python /workspace/scripts/analysis/plot_llama_d0_d2.py
"""
import json
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
FAMILIES = ["F1", "F2", "F3", "F4", "F5"]
FAMILY_LABELS = ["F1\nDirect\noverride", "F2\nStealth/\nrendering",
                  "F3\nReviewer\nimpersonation", "F4\nFabricated\nauthority",
                  "F5\nSycophancy"]


def main():
    d0 = json.loads((EVAL_DIR / "D0_meta-llama_Llama-3.1-8B-Instruct.json").read_text())["summary"]
    d2 = json.loads((EVAL_DIR / "D2_meta-llama_Llama-3.1-8B-Instruct.json").read_text())["summary"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    x = range(len(FAMILIES))
    width = 0.35

    # panel 1: absolute injected score
    ax = axes[0]
    d0_scores = [d0["per_family"][f]["mean_injected_score"] for f in FAMILIES]
    d2_scores = [d2["per_family"][f]["mean_injected_score"] for f in FAMILIES]
    d0_clean = d0["per_family"]["F1"]["mean_clean_score"]
    d2_clean = d2["per_family"]["F1"]["mean_clean_score"]
    ax.bar([i - width/2 for i in x], d0_scores, width, label="D0 (undefended)", color="tab:red")
    ax.bar([i + width/2 for i in x], d2_scores, width, label="D2 (trained)", color="tab:green")
    ax.axhline(d0_clean, color="tab:red", linestyle="--", linewidth=1, alpha=0.6, label=f"D0 clean ({d0_clean:.2f})")
    ax.axhline(d2_clean, color="tab:green", linestyle="--", linewidth=1, alpha=0.6, label=f"D2 clean ({d2_clean:.2f})")
    ax.set_xticks(list(x)); ax.set_xticklabels(FAMILY_LABELS)
    ax.set_ylabel("Mean absolute score under attack")
    ax.set_title("Absolute score (lower = better for D2)")
    ax.axvline(2.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.legend(fontsize=8)

    # panel 2: decision-flip ASR
    ax = axes[1]
    d0_flip = [d0["per_family"][f]["decision_flip_asr"] * 100 for f in FAMILIES]
    d2_flip = [d2["per_family"][f]["decision_flip_asr"] * 100 for f in FAMILIES]
    ax.bar([i - width/2 for i in x], d0_flip, width, label="D0 (undefended)", color="tab:red")
    ax.bar([i + width/2 for i in x], d2_flip, width, label="D2 (trained)", color="tab:green")
    ax.set_xticks(list(x)); ax.set_xticklabels(FAMILY_LABELS)
    ax.set_ylabel("Decision-flip ASR (%)")
    ax.set_title("Decision flip Reject->Accept (higher = worse for D2)")
    ax.axvline(2.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.legend()

    fig.suptitle("Llama-3.1-8B: D0 vs D2, absolute score vs decision-flip - a genuinely mixed result", y=1.03)
    fig.tight_layout()
    out_path = FIG_DIR / "llama_d0_d2_score_vs_decisionflip.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
