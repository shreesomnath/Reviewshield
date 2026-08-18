"""The key comparison this whole Gemini exercise was for: does our
trained local defense (Qwen2.5-14B D2) actually compete with or beat
what you'd get from a frontier model, undefended or with a simple
prompting defense? All real, absolute, bias-corrected scores.

Uses the injection_only variant consistently for all three models/
conditions (one pair per paper per family) -- an earlier version of this
script pulled Gemini's numbers from the official run_eval aggregate
(which mixes injection_only + injection_plus_flaw, n=100/family) while
pulling Qwen D2's numbers from absolute_score_sample.json
(injection_only only, n=50/family). Same underlying data, different
variant-mix, so the two were not directly comparable despite appearing
side by side in the same table. Fixed to match additional_stats.py's
methodology throughout.

Reads Gemini D0/D1 from eval_gemini_fix/, not eval/: the original Gemini
eval runs left the frontier backend's do_sample flag at its client
default (True, temperature=0.7) instead of the do_sample=False used
everywhere else, making Gemini's results non-deterministic. This was
corrected in run_eval_gemini_combined.py and the corrected data lives in
eval_gemini_fix/ (see dev_scripts/plot_gemini_vs_trained_PRE_SAMPLING_FIX.py
for the pre-correction version, kept for provenance only -- do not use it
to reproduce the paper's reported numbers).

Usage (inside the container):
    python /workspace/scripts/analysis/plot_gemini_vs_trained.py
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

EVAL_DIR = Path("/workspace/outputs/eval")
FIXED_DIR = Path("/workspace/outputs/eval_gemini_fix")
FIG_DIR = Path("/workspace/outputs/figures")
FAMILIES = ["F1", "F2", "F3", "F4", "F5"]
FAMILY_LABELS = ["F1\nDirect\noverride", "F2\nStealth/\nrendering",
                  "F3\nReviewer\nimpersonation", "F4\nFabricated\nauthority",
                  "F5\nSycophancy"]


def load_local_pairs(cond, model_id):
    path = FIXED_DIR / f"{cond}_{model_id.replace('/', '_')}_progress.jsonl"
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


def mean_injected(pairs_by_fam):
    return [sum(i for c, i in pairs_by_fam[f]) / len(pairs_by_fam[f]) if pairs_by_fam[f] else None for f in FAMILIES]


def main():
    gem_d0_pairs = load_local_pairs("D0", "gemini-flash-lite-latest")
    gem_d1_pairs = load_local_pairs("D1", "gemini-flash-lite-latest")
    qwen_d2_pairs = load_qwen_pairs("D2")

    gem_d0_scores = mean_injected(gem_d0_pairs)
    gem_d1_scores = mean_injected(gem_d1_pairs)
    qwen_d2_scores = mean_injected(qwen_d2_pairs)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(FAMILIES))
    width = 0.26

    ax.bar([i - width for i in x], gem_d0_scores, width, label="Gemini D0 (undefended frontier)", color="tab:red")
    ax.bar([i for i in x], gem_d1_scores, width, label="Gemini D1 (prompting defense)", color="tab:orange")
    ax.bar([i + width for i in x], qwen_d2_scores, width, label="Qwen2.5-14B D2 (our trained defense)", color="tab:green")

    ax.set_xticks(list(x)); ax.set_xticklabels(FAMILY_LABELS)
    ax.set_ylabel("Mean absolute score under attack")
    ax.set_title("Does our trained local defense compete with a frontier model?\n(lower = more robust; all real, bias-corrected absolute scores, injection_only variant, n=50/family)")
    ax.axvline(2.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path = FIG_DIR / "gemini_vs_trained_qwen.png"  # overwrites with corrected (deterministic) Gemini data
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")

    print("\n=== Table (injection_only, n=50/family, consistent across all three) ===")
    print(f"{'Family':<8}{'Gem D0':<10}{'Gem D1':<10}{'Qwen D2':<10}")
    for i, f in enumerate(FAMILIES):
        print(f"{f:<8}{gem_d0_scores[i]:<10}{gem_d1_scores[i]:<10}{qwen_d2_scores[i]:<10}")


if __name__ == "__main__":
    main()
