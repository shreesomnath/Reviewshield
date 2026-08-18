"""Same paired DID analysis as did_analysis.py, extended to Llama (D0/D1/D2)
and the Gemini frontier comparison (D0/D1 vs Qwen D2), to check whether the
same calibration confound affects those sections of the paper too.

The Llama results below are canonical and unaffected by the frontier
sampling bug (local backend, always deterministic). The Gemini results
in this file read from the ORIGINAL, non-deterministic eval/ directory
and are superseded -- use did_analysis_gemini.py for the corrected,
reported Gemini numbers instead.
"""
import json
import random
from pathlib import Path
from scipy.stats import wilcoxon

EVAL_DIR = Path("/workspace/outputs/eval")
FAMILIES = ["F1", "F2", "F3", "F4", "F5"]


def load_local_paper_family_scores(cond, model_file):
    path = EVAL_DIR / f"{cond}_{model_file}_progress.jsonl"
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
                if item.get("clean_score") is not None and item.get("injected_score") is not None:
                    out[aid][item["family"]] = (item["clean_score"], item["injected_score"])
    return out


def load_qwen_paper_family_scores(cond):
    path = EVAL_DIR / f"abs_score_progress_{cond}.jsonl"
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
    common_aids = sorted(set(data_a) & set(data_b))
    print(f"\n=== DID: {label}, n_papers={len(common_aids)} ===")
    print(f"{'Family':<8}{'n':<5}{'mean DefenseEff':<18}{'p-value':<12}{'95% CI'}")
    for fam in FAMILIES:
        pairs = []
        for aid in common_aids:
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


# ---- Llama ----
llama_d0 = load_local_paper_family_scores("D0", "meta-llama_Llama-3.1-8B-Instruct")
llama_d1 = load_local_paper_family_scores("D1", "meta-llama_Llama-3.1-8B-Instruct")
llama_d2 = load_local_paper_family_scores("D2", "meta-llama_Llama-3.1-8B-Instruct")
did(llama_d0, llama_d2, "Llama: D2 vs D0 (does DPO help on Llama?)")
did(llama_d0, llama_d1, "Llama: D1 vs D0 (does prompting help on Llama?)")

# ---- Gemini frontier comparison ----
gem_d0 = load_local_paper_family_scores("D0", "gemini-flash-lite-latest")
gem_d1 = load_local_paper_family_scores("D1", "gemini-flash-lite-latest")
did(gem_d0, gem_d1, "Gemini: D1 vs D0 (does prompting help on the frontier model?)")

# Cross-model-family comparison isn't directly paired (different papers/models can't share an aid
# assumption safely) but Qwen D2 vs Gemini D1 IS comparable since both evaluate the same 50 papers.
qwen_d0 = load_qwen_paper_family_scores("D0")
qwen_d2 = load_qwen_paper_family_scores("D2")
did(gem_d0, qwen_d2, "Cross-model: Qwen D2's attack-effect vs Gemini D0's attack-effect (each own baseline)")
