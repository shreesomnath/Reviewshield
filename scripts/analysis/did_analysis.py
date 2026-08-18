"""Real paired difference-in-differences (DID) analysis, exactly as the
independent review's Section 4.1 requests, computed on real per-paper
data rather than approximated from global means.

AttackEffect(c, p, f)  = Score_attacked(c, p, f) - Score_clean(c, p, f)
DefenseEffect(D2vD0,p,f) = AttackEffect(D2,p,f) - AttackEffect(D0,p,f)

Reports mean/median DefenseEffect per family with a paired Wilcoxon test
against 0 and a bootstrap CI, for both D2-vs-D0 and D3-vs-D2 (the SFT
ablation comparison the reviewer also raised in the same section).
"""
import json
import random
from pathlib import Path
from scipy.stats import wilcoxon

EVAL_DIR = Path("/workspace/outputs/eval")
FAMILIES = ["F1", "F2", "F3", "F4", "F5"]


def load_paper_family_scores(cond):
    """Returns {aid: {fam: (clean, injected)}} from abs_score_progress_{cond}.jsonl
    (Qwen D0/D1/D2) or from the D3 progress file for D3."""
    if cond in ("D0", "D1", "D2"):
        path = EVAL_DIR / f"abs_score_progress_{cond}.jsonl"
        out = {}
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                out[rec["aid"]] = {fam: (d["clean"], d["injected"]) for fam, d in rec["per_family"].items()}
        return out
    else:  # D3
        path = EVAL_DIR / "D3_Qwen_Qwen2.5-14B-Instruct_progress.jsonl"
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


def bootstrap_ci(values, n_boot=4000, ci=0.95):
    rng = random.Random(0)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((1 - ci) / 2 * n_boot)]
    hi = means[int((1 + ci) / 2 * n_boot)]
    return lo, hi


def did(cond_a, cond_b, label):
    """DefenseEffect = AttackEffect(cond_b) - AttackEffect(cond_a), i.e.
    does cond_b reduce the attack-induced score change relative to cond_a?
    Negative = cond_b better (smaller attack-induced rise)."""
    data_a = load_paper_family_scores(cond_a)
    data_b = load_paper_family_scores(cond_b)
    common_aids = sorted(set(data_a) & set(data_b))
    print(f"\n=== DID: {label} ({cond_b} vs {cond_a}), n_papers={len(common_aids)} ===")
    print(f"{'Family':<8}{'n':<5}{'mean AttackEff '+cond_a:<20}{'mean AttackEff '+cond_b:<20}{'mean DefenseEff':<18}{'p-value':<12}{'95% CI'}")
    for fam in FAMILIES:
        pairs = []
        for aid in common_aids:
            if fam not in data_a[aid] or fam not in data_b[aid]:
                continue
            ca, ia = data_a[aid][fam]
            cb, ib = data_b[aid][fam]
            attack_eff_a = ia - ca
            attack_eff_b = ib - cb
            defense_eff = attack_eff_b - attack_eff_a
            pairs.append((attack_eff_a, attack_eff_b, defense_eff))
        if len(pairs) < 3:
            print(fam, "insufficient data")
            continue
        mean_a = sum(p[0] for p in pairs) / len(pairs)
        mean_b = sum(p[1] for p in pairs) / len(pairs)
        defense_effs = [p[2] for p in pairs]
        mean_defense = sum(defense_effs) / len(defense_effs)
        try:
            _, p = wilcoxon(defense_effs)
        except ValueError:
            p = float("nan")
        lo, hi = bootstrap_ci(defense_effs)
        sig = "*" if p < 0.05 else " "
        print(f"{fam:<8}{len(pairs):<5}{mean_a:<20.3f}{mean_b:<20.3f}{mean_defense:<18.3f}{p:<12.2e}{sig}[{lo:.3f}, {hi:.3f}]")


did("D0", "D2", "Does DPO training reduce the attack-induced score change vs undefended?")
did("D2", "D3", "Does SFT (D3) reduce the attack-induced score change vs DPO (D2)?")
did("D0", "D3", "Does SFT training reduce the attack-induced score change vs undefended?")
did("D0", "D1", "Does prompting (D1) reduce the attack-induced score change vs undefended?")
