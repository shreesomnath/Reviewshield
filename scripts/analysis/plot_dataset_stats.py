"""Dataset-characteristics figure for RevGuard-Bench's generated preference
pairs - producible now, without waiting for training/eval, since it only
uses the already-complete data-generation output (train_pairs.jsonl,
val_pairs.jsonl) plus the benchmark's raw variant counts.

Plots per-family "yield rate": informative pairs produced / raw attack
attempts. A LOW yield rate for a family means the judge model (Gemini
flash-lite, acting as both generator and behavioral proxy during data
generation) already resisted that attack in BOTH the neutral and defensive
framings most of the time - i.e., the attack rarely succeeded on either
side, so there was no contrast to learn from. A HIGH yield rate means the
attack more often succeeded on at least one framing, producing real
contrastive training signal. This is itself a real, reportable finding
about baseline attack difficulty, not just a data-processing statistic.

Usage (inside the container):
    python /workspace/scripts/analysis/plot_dataset_stats.py
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
})
import matplotlib.pyplot as plt

DATA_DIR = Path("/workspace/data/processed")
FIG_DIR = Path("/workspace/outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

FAMILY_NAMES = {
    "F1": "F1\nDirect\noverride",
    "F2": "F2\nStealth/\nrendering",
    "F3": "F3\nReviewer\nimpersonation",
}


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def main():
    train = load_jsonl(DATA_DIR / "preference_pairs/train_pairs.jsonl")
    val = load_jsonl(DATA_DIR / "preference_pairs/val_pairs.jsonl")
    all_pairs = train + val

    train_bench = load_jsonl(DATA_DIR / "revguard_bench/train.jsonl")
    val_bench = load_jsonl(DATA_DIR / "revguard_bench/val.jsonl")
    all_bench = train_bench + val_bench

    pair_fam_counts = Counter(p["family"] for p in all_pairs if p["family"] is not None)
    raw_fam_counts = Counter(v["family"] for v in all_bench if v["family"] is not None)

    families = ["F1", "F2", "F3"]  # only seen families are in train/val by design
    yield_rates = [100 * pair_fam_counts[f] / raw_fam_counts[f] for f in families]
    labels = [FAMILY_NAMES[f] for f in families]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel (a): yield rate per family
    ax = axes[0]
    bars = ax.bar(labels, yield_rates, color="#0072B2", edgecolor="black", linewidth=0.5)
    for bar, f in zip(bars, families):
        h = bar.get_height()
        ax.annotate(f"{pair_fam_counts[f]}/{raw_fam_counts[f]}", (bar.get_x() + bar.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Informative-pair yield rate (%)")
    ax.set_title("(a) Preference-pair yield by attack family\n(train+val, seen families only)")
    ax.grid(alpha=0.25, axis="y")
    ax.set_ylim(0, max(yield_rates) * 1.3)

    # Panel (b): variant-type composition of the full dataset
    ax = axes[1]
    type_counts = Counter(p["variant_type"] for p in all_pairs)
    type_labels = {"clean_anti_over_refusal": "Clean\n(anti-over-refusal)",
                   "injection_only": "Injection\nonly",
                   "injection_plus_flaw": "Injection\n+ flaw"}
    labels2 = [type_labels[t] for t in type_counts]
    values2 = list(type_counts.values())
    colors2 = ["#009E73", "#D55E00", "#CC79A7"]
    ax.bar(labels2, values2, color=colors2[:len(labels2)], edgecolor="black", linewidth=0.5)
    for i, v in enumerate(values2):
        ax.annotate(str(v), (i, v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Preference pairs")
    ax.set_title(f"(b) Final dataset composition\n(train={len(train)}, val={len(val)}, total={len(all_pairs)})")
    ax.grid(alpha=0.25, axis="y")

    plt.tight_layout()
    out_path = FIG_DIR / "dataset_yield_and_composition.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")
    print(f"Yield rates: {dict(zip(families, [f'{y:.1f}%' for y in yield_rates]))}")


if __name__ == "__main__":
    main()
