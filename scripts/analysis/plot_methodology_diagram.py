"""Publication methodology/pipeline figure for ReviewShield.

Iterated for legibility, fit, AND spacing. The canvas is taller than wide
enough to give the training band real vertical room, so Preference pairs
sits cleanly between ReviewShield-Bench (above) and the DPO/SFT fork
(below) with visible gaps and proper fork arrows -- no overlaps. Fonts are
large enough to stay legible at single-column \\textwidth. Boxes are
grid-aligned: DPO/SFT sit directly above D2/D3 so those arrows are exact
verticals; the two branch arrows (train split, test split) run through
empty channels with their labels on them.

Usage: python plot_methodology_diagram.py (matplotlib only).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(11, 9.0))
ax.set_xlim(0, 22)
ax.set_ylim(0, 18)
ax.axis("off")

PAD = 0.10
BOX_STYLE = dict(boxstyle=f"round,pad={PAD},rounding_size=0.14", linewidth=1.6)
COL = {"data": "#c9ddf0", "attack": "#f3c9cc", "train": "#cfe6c9", "eval": "#fdeec2"}
EDGE = "#3a3a3a"


def box(x, y, w, h, title, sub, color, tfs=14.0, sfs=11.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=color, edgecolor=EDGE, **BOX_STYLE))
    if sub is None:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color="#111111")
    else:
        ax.text(x + w / 2, y + h - 0.48, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color="#111111")
        ax.text(x + w / 2, y + (h - 0.80) / 2, sub, ha="center", va="center",
                fontsize=sfs, color="#1e1e1e", linespacing=1.4)
    return (x, y, w, h)


def arrow(p1, p2, label=None, lpos=None):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=18,
                                 shrinkA=0, shrinkB=0, linewidth=1.7, color=EDGE))
    if label:
        lx, ly = lpos if lpos else ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=11.0,
                style="italic", color="#444444",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.9))


# ---- Data sourcing (stacked, top-left) ----
b_corpus = box(0.4, 15.0, 3.7, 1.5, "arXiv corpus", "~800 CS/ML\nmanuscripts", COL["data"], tfs=13.5, sfs=11.0)
b_flaws  = box(0.4, 13.3, 3.7, 1.5, "Planted flaws", "fabricated stats,\nmissing baselines", COL["data"], tfs=13.5, sfs=11.0)

# ---- Attack construction (wide, real gap) ----
b_tax   = box(4.4, 13.6, 6.0, 2.7, "Attack taxonomy",
              "F1 override · F2 stealth\nF3 impersonation\nF4 authority · F5 sycophancy", COL["attack"])
b_bench = box(11.0, 13.6, 6.0, 2.7, "ReviewShield-Bench",
              "clean / injected / flawed\ntrain+val: F1–F3\ntest: all F1–F5", COL["attack"])

# ---- Training band (pref stacked above the DPO/SFT fork, with gaps) ----
b_pref = box(14.0, 11.0, 4.4, 1.75, "Preference pairs",
             "chosen vs rejected\n1,057 train · 150 val", COL["train"], sfs=11.0)
b_dpo  = box(11.35, 7.6, 4.6, 2.5, "DPO training",
             "chosen + rejected\nQwen-14B + LoRA → D2", COL["train"], sfs=11.0)
b_sft  = box(16.45, 7.6, 4.6, 2.5, "SFT training",
             "chosen only (ablation)\nsame data + LoRA → D3", COL["train"], sfs=11.0)

# ---- Evaluation container ----
ax.add_patch(FancyBboxPatch((0.7, 3.4), 20.8, 3.9, facecolor="#fbfbfb",
                            edgecolor="#7a7a7a", linestyle=(0, (5, 3)),
                            boxstyle="round,pad=0.10,rounding_size=0.14", linewidth=1.3))
ax.text(1.05, 6.95, "Evaluation — 50 held-out test papers × F1–F5, clean + injected",
        ha="left", va="center", fontsize=12.5, fontweight="bold", color="#222222")
ax.text(1.05, 3.78, "Reviewer models:  Qwen2.5-14B (primary, D0–D3)   ·   "
                    "Llama-3.1-8B (secondary, D0–D2)   ·   gemini-flash-lite (frontier, D0–D1)",
        ha="left", va="center", fontsize=10.5, style="italic", color="#555555")

b_d0 = box(1.1, 4.2, 4.5, 2.1, "D0 · undefended", "base weights\nneutral prompt", COL["eval"], tfs=12.5, sfs=11.0)
b_d1 = box(6.2, 4.2, 4.5, 2.1, "D1 · prompting", "base weights +\ndefensive prompt", COL["eval"], tfs=12.5, sfs=11.0)
b_d2 = box(11.4, 4.2, 4.5, 2.1, "D2 · DPO (ours)", "DPO adapter\nneutral prompt", COL["eval"], tfs=12.5, sfs=11.0)
b_d3 = box(16.5, 4.2, 4.5, 2.1, "D3 · SFT ablation", "SFT adapter\nneutral prompt", COL["eval"], tfs=12.5, sfs=11.0)

# ---- Metrics ----
box(0.7, 0.5, 20.8, 2.1, "Metric suite",
    "score-inflation ASR · decision-flip ASR · bias-corrected absolute score (Wilcoxon + effect size)\n"
    "flaw-detection recall · over-refusal rate · seen (F1–F3) vs held-out (F4–F5) generalization gap",
    "#eaeaea", tfs=13.5, sfs=11.5)

# ---- Arrows ----
arrow((4.1, 15.75), (4.4, 15.35))                      # corpus  -> taxonomy
arrow((4.1, 14.05), (4.4, 14.45))                      # flaws   -> taxonomy
arrow((10.4, 14.95), (11.0, 14.95))                    # taxonomy -> bench
arrow((14.7, 13.6), (16.0, 12.77),                     # bench -> preference pairs (train split)
      label="train split\nF1–F3", lpos=(18.4, 13.55))
arrow((15.2, 11.0), (13.9, 10.14))                     # pref -> DPO (fork left)
arrow((17.2, 11.0), (18.5, 10.14))                     # pref -> SFT (fork right)
arrow((13.65, 7.6), (13.65, 6.32))                     # DPO -> D2 (vertical)
arrow((18.75, 7.6), (18.75, 6.32))                     # SFT -> D3 (vertical)
arrow((11.5, 13.6), (7.0, 7.32),                       # bench -> evaluation (test split)
      label="test split\nF1–F5", lpos=(8.1, 10.3))
arrow((11.1, 3.4), (11.1, 2.62))                       # evaluation -> metrics

# ---- Legend ----
legend_elems = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COL["data"], markeredgecolor=EDGE, markersize=15, label="Data sourcing"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COL["attack"], markeredgecolor=EDGE, markersize=15, label="Attack construction"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COL["train"], markeredgecolor=EDGE, markersize=15, label="Training"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COL["eval"], markeredgecolor=EDGE, markersize=15, label="Evaluation conditions"),
]
ax.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          ncol=4, frameon=False, fontsize=12.5, handletextpad=0.5, columnspacing=1.8)

fig.tight_layout()
fig.savefig("/workspace/outputs/figures/methodology_pipeline.png", dpi=220, bbox_inches="tight")
print("Wrote methodology_pipeline.png")
