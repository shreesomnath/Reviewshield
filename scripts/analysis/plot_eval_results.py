"""Evaluation-results figure + table, built from whatever outputs/eval/*.json
files exist at run time. Safe to re-run as more conditions (D0-local, D2,
...) finish - it just picks up every JSON file present.

Produces:
  - outputs/figures/eval_asr_by_family.png: grouped bars, score-inflation
    ASR per attack family, one bar group per condition run so far.
  - outputs/tables/eval_summary.md: a markdown table of the headline
    numbers (seen-family ASR, held-out ASR, FDR, over-refusal) per
    condition, for direct use in the manuscript.

Usage (inside the container):
    python /workspace/scripts/analysis/plot_eval_results.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
})
import matplotlib.pyplot as plt
import numpy as np

EVAL_DIR = Path("/workspace/outputs/eval")
FIG_DIR = Path("/workspace/outputs/figures")
TABLE_DIR = Path("/workspace/outputs/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

FAMILIES = ["F1", "F2", "F3", "F4", "F5"]
FAMILY_LABELS = ["F1\nDirect\noverride", "F2\nStealth/\nrendering",
                  "F3\nReviewer\nimpersonation", "F4\nFabricated\nauthority",
                  "F5\nSycophancy"]

# human-readable label per (condition, backend, model_id) combo
def label_for(run):
    cond = run["condition"]
    backend = run["backend"]
    model = run["model_id"]
    if backend == "frontier":
        return f"{cond} (frontier: {model})"
    short_model = model.split("/")[-1]
    return f"{cond} (local: {short_model})"


def main():
    # Only real condition-result files (D0_/D1_/D2_/D3_*.json) - excludes
    # absolute_score_sample.json, which is a different structure (no
    # condition/backend/model_id top-level keys) from a separate ad hoc
    # analysis, not a run_eval.py condition output.
    files = sorted(f for f in EVAL_DIR.glob("*.json") if f.name[:3] in ("D0_", "D1_", "D2_", "D3_"))
    if not files:
        print("No eval outputs found yet in outputs/eval/ - nothing to plot.")
        return
    runs = [json.loads(f.read_text()) for f in files]
    print(f"Found {len(runs)} eval run(s): {[label_for(r) for r in runs]}")

    # --- grouped bar chart: score-inflation ASR by family ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    n_runs = len(runs)
    bar_width = 0.8 / max(n_runs, 1)
    x = np.arange(len(FAMILIES))
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_runs, 1)))

    for i, run in enumerate(runs):
        per_family = run["summary"]["per_family"]
        vals = [per_family.get(f, {}).get("score_inflation_asr", 0.0) * 100
                for f in FAMILIES]
        offset = (i - (n_runs - 1) / 2) * bar_width
        ax.bar(x + offset, vals, bar_width, label=label_for(run), color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(FAMILY_LABELS)
    ax.set_ylabel("Score-inflation ASR (%)")
    ax.set_title("Attack success rate by family and condition\n"
                  "(F4/F5 = held out of training; lower is better)")
    ax.axvline(2.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(1.0, ax.get_ylim()[1] * 0.95, "seen (train)", ha="center", fontsize=9, color="gray")
    ax.text(3.5, ax.get_ylim()[1] * 0.95, "held out", ha="center", fontsize=9, color="gray")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    out_png = FIG_DIR / "eval_asr_by_family.png"
    fig.savefig(out_png, dpi=150)
    print(f"Wrote {out_png}")

    # --- summary table ---
    lines = ["| Condition | Seen ASR (F1-F3) | Held-out ASR (F4-F5) | "
             "Decision-flip ASR | Flaw-detect recall | Over-refusal |",
             "|---|---|---|---|---|---|"]
    for run in runs:
        s = run["summary"]
        seen = s["seen_families_avg"]["score_inflation_asr"] * 100
        held = s["heldout_families_avg"]["score_inflation_asr"] * 100
        dflip = np.mean([s["seen_families_avg"]["decision_flip_asr"],
                          s["heldout_families_avg"]["decision_flip_asr"]]) * 100
        fdr = s["flaw_only_fdr"] * 100
        overref = s["over_refusal_rate"] * 100
        lines.append(f"| {label_for(run)} | {seen:.1f}% | {held:.1f}% | "
                      f"{dflip:.1f}% | {fdr:.1f}% | {overref:.1f}% |")
    table_md = "\n".join(lines)
    out_table = TABLE_DIR / "eval_summary.md"
    out_table.write_text(table_md + "\n")
    print(f"Wrote {out_table}\n")
    print(table_md)


if __name__ == "__main__":
    main()
