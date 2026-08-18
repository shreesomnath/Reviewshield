"""Cross-model vulnerability comparison: D0 (undefended) on Qwen2.5-14B
vs Llama-3.1-8B, using the already-complete full 50-paper evaluations for
both. Real, both conditions finished - no new generation needed.

Panel 2 (absolute score) uses the injection_only variant consistently for
BOTH models (one pair per paper per family, n=50) -- an earlier version
pulled Qwen's absolute score from absolute_score_sample.json
(injection_only only) but Llama's from the official run_eval aggregate
(which mixes injection_only + injection_plus_flaw, n=100), an
apples-to-oranges comparison despite appearing in the same panel. Fixed
to match additional_stats.py's methodology.

Usage (inside the container):
    python /workspace/scripts/analysis/plot_cross_model_comparison.py
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


def load(cond, model_file):
    path = EVAL_DIR / f"{cond}_{model_file}.json"
    return json.loads(path.read_text())["summary"]


def load_qwen_pairs(cond):
    path = EVAL_DIR / f"abs_score_progress_{cond}.jsonl"
    pairs = {f: [] for f in FAMILIES}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            for fam, d in rec["per_family"].items():
                if fam in pairs:
                    pairs[fam].append((d["clean"], d["injected"]))
    return pairs


def load_llama_pairs(cond, model_file):
    path = EVAL_DIR / f"{cond}_{model_file}_progress.jsonl"
    pairs = {f: [] for f in FAMILIES}
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


def mean_of(pairs, idx):
    return [sum(p[idx] for p in pairs[f]) / len(pairs[f]) if pairs[f] else None for f in FAMILIES]


def main():
    qwen = load("D0", "Qwen_Qwen2.5-14B-Instruct")
    llama = load("D0", "meta-llama_Llama-3.1-8B-Instruct")
    qwen_pairs = load_qwen_pairs("D0")
    llama_pairs = load_llama_pairs("D0", "meta-llama_Llama-3.1-8B-Instruct")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # panel 1: relative ASR (official aggregate, mixed variant, consistent for both models)
    ax = axes[0]
    x = range(len(FAMILIES))
    width = 0.35
    qwen_asr = [qwen["per_family"][f]["score_inflation_asr"] * 100 for f in FAMILIES]
    llama_asr = [llama["per_family"][f]["score_inflation_asr"] * 100 for f in FAMILIES]
    ax.bar([i - width/2 for i in x], qwen_asr, width, label="Qwen2.5-14B", color="tab:blue")
    ax.bar([i + width/2 for i in x], llama_asr, width, label="Llama-3.1-8B", color="tab:orange")
    ax.set_xticks(list(x))
    ax.set_xticklabels(FAMILY_LABELS)
    ax.set_ylabel("Score-inflation ASR (%)")
    ax.set_title("Undefended (D0) relative ASR by base model")
    ax.axvline(2.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.legend()

    # panel 2: absolute injected score (injection_only, n=50, consistent for both models)
    ax = axes[1]
    qwen_abs = mean_of(qwen_pairs, 1)
    llama_abs = mean_of(llama_pairs, 1)
    qwen_clean = sum(p[0] for p in qwen_pairs["F1"]) / len(qwen_pairs["F1"])
    llama_clean = sum(p[0] for p in llama_pairs["F1"]) / len(llama_pairs["F1"])
    ax.bar([i - width/2 for i in x], qwen_abs, width, label="Qwen2.5-14B", color="tab:blue")
    ax.bar([i + width/2 for i in x], llama_abs, width, label="Llama-3.1-8B", color="tab:orange")
    ax.axhline(qwen_clean, color="tab:blue", linestyle="--", linewidth=1, alpha=0.6, label=f"Qwen clean ({qwen_clean:.2f})")
    ax.axhline(llama_clean, color="tab:orange", linestyle="--", linewidth=1, alpha=0.6, label=f"Llama clean ({llama_clean:.2f})")
    ax.set_xticks(list(x))
    ax.set_xticklabels(FAMILY_LABELS)
    ax.set_ylabel("Mean absolute score under attack")
    ax.set_title("Undefended (D0) absolute score by base model\n(injection_only, n=50/family)")
    ax.axvline(2.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.legend(fontsize=8)

    fig.suptitle("Cross-model vulnerability comparison, undefended baseline (n=50 papers each)", y=1.02)
    fig.tight_layout()
    out_path = FIG_DIR / "cross_model_d0_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")

    print("\n=== Summary ===")
    print(f"Qwen  seen avg ASR: {qwen['seen_families_avg']['score_inflation_asr']*100:.1f}%  "
          f"held-out avg ASR: {qwen['heldout_families_avg']['score_inflation_asr']*100:.1f}%")
    print(f"Llama seen avg ASR: {llama['seen_families_avg']['score_inflation_asr']*100:.1f}%  "
          f"held-out avg ASR: {llama['heldout_families_avg']['score_inflation_asr']*100:.1f}%")
    print("\n=== Panel 2 numbers (injection_only, consistent) ===")
    print(f"{'Family':<8}{'Qwen abs':<10}{'Llama abs':<10}")
    for i, f in enumerate(FAMILIES):
        print(f"{f:<8}{qwen_abs[i]:<10.2f}{llama_abs[i]:<10.2f}")


if __name__ == "__main__":
    main()
