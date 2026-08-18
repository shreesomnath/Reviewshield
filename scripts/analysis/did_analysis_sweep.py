"""DID robustness check: does the null D2-vs-D0 result survive at
different DPO hyperparameters, or is it an artifact of the specific
(beta=0.1, lr=5e-6) choice used for the paper's main D2? Computed exactly
like the original did_analysis.py (same AttackEffect/DefenseEffect
formula, same paired Wilcoxon + bootstrap CI), on the same 50 test
papers, against the same D0 baseline, for two new configs:
  - beta=0.3,  lr=5e-6   (higher KL penalty, same LR)
  - beta=0.1,  lr=2e-5   (same beta, 4x higher LR)
"""
import json
import random
from pathlib import Path
from scipy.stats import wilcoxon

EVAL_DIR = Path("/workspace/outputs/eval")
SWEEP_DIRS = {
    "beta03": Path("/workspace/outputs/eval_sweep_beta03"),
    "lr2e5": Path("/workspace/outputs/eval_sweep_lr2e5"),
}
FAMILIES = ["F1", "F2", "F3", "F4", "F5"]


def load_d0():
    path = EVAL_DIR / "abs_score_progress_D0.jsonl"
    out = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            out[rec["aid"]] = {fam: (d["clean"], d["injected"]) for fam, d in rec["per_family"].items()}
    return out


def load_sweep(sweep_dir):
    path = sweep_dir / "D2_Qwen_Qwen2.5-14B-Instruct_progress.jsonl"
    out = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            r = rec["result"]
            if r.get("skipped"):
                continue
            aid = rec["arxiv_id"]
            out[aid] = {}
            for item in r["injected"]:
                if item["variant_type"] != "injection_only":
                    continue
                if item["clean_score"] is not None and item["injected_score"] is not None:
                    out[aid][item["family"]] = (item["clean_score"], item["injected_score"])
    return out


def load_original_d2():
    path = EVAL_DIR / "abs_score_progress_D2.jsonl"
    out = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            out[rec["aid"]] = {fam: (d["clean"], d["injected"]) for fam, d in rec["per_family"].items()}
    return out


def bootstrap_ci(values, n_boot=4000, ci=0.95):
    rng = random.Random(0)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return means[int((1 - ci) / 2 * n_boot)], means[int((1 + ci) / 2 * n_boot)]


def did(data_a, data_b, label):
    common = sorted(set(data_a) & set(data_b))
    print(f"\n=== DID: {label}, n_papers={len(common)} ===")
    print(f"{'Family':<8}{'n':<5}{'mean DefenseEff':<18}{'p-value':<12}{'95% CI'}")
    for fam in FAMILIES:
        pairs = []
        for aid in common:
            if fam not in data_a[aid] or fam not in data_b[aid]:
                continue
            ca, ia = data_a[aid][fam]
            cb, ib = data_b[aid][fam]
            defense_eff = (ib - cb) - (ia - ca)
            pairs.append(defense_eff)
        if len(pairs) < 3:
            print(fam, "insufficient data")
            continue
        mean_defense = sum(pairs) / len(pairs)
        try:
            _, p = wilcoxon(pairs)
        except ValueError:
            p = float("nan")
        lo, hi = bootstrap_ci(pairs)
        sig = "*" if p < 0.05 else " "
        print(f"{fam:<8}{len(pairs):<5}{mean_defense:<18.3f}{p:<12.2e}{sig}[{lo:.3f}, {hi:.3f}]")


d0 = load_d0()
d2_orig = load_original_d2()
beta03 = load_sweep(SWEEP_DIRS["beta03"])
lr2e5 = load_sweep(SWEEP_DIRS["lr2e5"])

print(f"D0 papers: {len(d0)}, original D2: {len(d2_orig)}, beta03: {len(beta03)}, lr2e5: {len(lr2e5)}")

did(d0, beta03, "beta=0.3 (sweep) vs D0 -- does higher-beta DPO help?")
did(d0, lr2e5, "lr=2e-5 (sweep) vs D0 -- does higher-LR DPO help?")
did(d2_orig, beta03, "beta=0.3 (sweep) vs original D2 (beta=0.1) -- effect of beta alone")
did(d2_orig, lr2e5, "lr=2e-5 (sweep) vs original D2 (lr=5e-6) -- effect of LR alone")
