"""Grouped bar chart visualizing the DPO hyperparameter-robustness check
(Table dpo-sweep in the manuscript): DefenseEffect(config vs D0) per
family, for D2's original config (beta=0.1, lr=5e-6) alongside the two
sweep configs (beta=0.3; lr=2e-5). Matches the style conventions used in
parse_and_plot_absolute_scores.py.

Usage (inside the container):
    python /workspace/scripts/analysis/plot_dpo_sweep.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
})
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("/workspace/outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

FAMILIES = ["F1", "F2", "F3", "F4", "F5"]
FAMILY_LABELS = ["F1\nDirect\noverride", "F2\nStealth/\nrendering",
                  "F3\nReviewer\nimpersonation", "F4\nFabricated\nauthority",
                  "F5\nSycophancy"]

# DefenseEffect(config vs D0) and paired Wilcoxon p, from did_analysis.py
# (original D2) and did_analysis_sweep.py (beta03, lr2e5) -- real computed
# values, copied here for plotting only, not recomputed.
DATA = {
    "D2 (default: $\\beta$=0.1, lr=5e-6)": {
        "F1": (0.10, 0.32), "F2": (0.20, 0.077), "F3": (0.18, 0.039),
        "F4": (-0.10, 0.35), "F5": (-0.22, 0.070),
    },
    "$\\beta$=0.3 (lr unchanged)": {
        "F1": (0.08, 0.40), "F2": (0.12, 0.16), "F3": (0.18, 0.020),
        "F4": (0.04, 0.62), "F5": (-0.06, 0.58),
    },
    "lr=2e-5 ($\\beta$ unchanged)": {
        "F1": (-0.52, 9.7e-3), "F2": (-0.48, 1.1e-2), "F3": (0.06, 0.41),
        "F4": (-0.24, 1.4e-2), "F5": (-0.64, 1.7e-4),
    },
}
COLORS = ["#4C72B0", "#DD8452", "#55A868"]

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(FAMILIES))
width = 0.26

# Fix the y-range explicitly first (with headroom for asterisks) so the
# per-bar offset below is guaranteed to land inside the visible axes,
# regardless of how large any single bar's magnitude is.
all_effects = [v[f][0] for v in DATA.values() for f in FAMILIES]
y_min, y_max = min(all_effects), max(all_effects)
y_span = y_max - y_min
pad = 0.22 * y_span
ax.set_ylim(y_min - pad, y_max + pad)
offset_mag = 0.05 * y_span

for i, (label, vals) in enumerate(DATA.items()):
    offsets = x + (i - 1) * width
    effects = [vals[f][0] for f in FAMILIES]
    pvals = [vals[f][1] for f in FAMILIES]
    bars = ax.bar(offsets, effects, width, label=label, color=COLORS[i], edgecolor="black", linewidth=0.5)
    for bar, p, eff in zip(bars, pvals, effects):
        if p < 0.05:
            y = eff + (offset_mag if eff >= 0 else -offset_mag)
            ax.text(bar.get_x() + bar.get_width() / 2, y, "*", ha="center",
                    va="bottom" if eff >= 0 else "top", fontsize=16, fontweight="bold",
                    clip_on=False)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(FAMILY_LABELS)
ax.set_ylabel("DefenseEffect vs. D0\n(negative = more robust than undefended)")
ax.set_title("DPO hyperparameter-robustness check (Qwen2.5-14B, paired DID vs. D0)\n"
              "* = statistically significant, paired Wilcoxon $p<0.05$ ($n=50$ papers/family)")
ax.legend(loc="upper right", framealpha=0.95)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()

out_path = FIG_DIR / "dpo_sweep_defense_effect.png"
fig.savefig(out_path, dpi=220)
print(f"Wrote {out_path}")
