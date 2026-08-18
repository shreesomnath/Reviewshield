"""One master re-verification pass: recompute EVERY number that appears
in a manuscript table, fresh from source, independent of any earlier
script's cached output. Prints everything in a form that's easy to diff
against the actual LaTeX tables by eye.
"""
import json
from pathlib import Path
from scipy.stats import wilcoxon, binomtest

EVAL_DIR = Path("/workspace/outputs/eval")
FAMILIES = ["F1", "F2", "F3", "F4", "F5"]


def rank_biserial(clean, injected):
    diffs = [i - c for c, i in zip(clean, injected)]
    nonzero = [d for d in diffs if d != 0]
    if not nonzero:
        return 0.0
    n_pos = sum(1 for d in nonzero if d > 0)
    n_neg = sum(1 for d in nonzero if d < 0)
    return (n_pos - n_neg) / len(nonzero)


def load_official(cond, model_file):
    path = EVAL_DIR / f"{cond}_{model_file}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["summary"]


def load_qwen_abs_pairs(cond):
    path = EVAL_DIR / f"abs_score_progress_{cond}.jsonl"
    pairs = {f: [] for f in FAMILIES}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            for fam, d in rec["per_family"].items():
                if fam in pairs:
                    pairs[fam].append((d["clean"], d["injected"]))
    return pairs


def load_local_pairs(cond, model_file):
    path = EVAL_DIR / f"{cond}_{model_file}_progress.jsonl"
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


print("############ TABLE 1: Qwen ASR (official aggregate, mixed variant) ############")
for cond in ["D0", "D1", "D2"]:
    s = load_official(cond, "Qwen_Qwen2.5-14B-Instruct")
    row = [f"{s['per_family'][f]['score_inflation_asr']:.2f}" for f in FAMILIES]
    print(cond, row)

print("\n############ TABLE 2 & 3: Qwen absolute score, D0/D1/D2/D3 (injection_only) + Wilcoxon/effect size ############")
qwen_pairs = {c: load_qwen_abs_pairs(c) for c in ["D0", "D1", "D2"]}
qwen_pairs["D3"] = load_local_pairs("D3", "Qwen_Qwen2.5-14B-Instruct")
for cond in ["D0", "D1", "D2", "D3"]:
    means = []
    for f in FAMILIES:
        pairs = qwen_pairs[cond][f]
        means.append(round(sum(i for c, i in pairs) / len(pairs), 2))
    print(cond, means, "n=", [len(qwen_pairs[cond][f]) for f in FAMILIES])

print("\nWilcoxon + rank-biserial, D0 only:")
for f in FAMILIES:
    pairs = qwen_pairs["D0"][f]
    clean = [c for c, i in pairs]
    inj = [i for c, i in pairs]
    _, p = wilcoxon([i - c for c, i in pairs])
    r = rank_biserial(clean, inj)
    print(f, f"p={p:.2e}", f"r={r:.2f}")

print("\n############ TABLE 4 & 5: Llama absolute + ASR (official aggregate for ASR, injection_only for absolute) ############")
llama_pairs = {c: load_local_pairs(c, "meta-llama_Llama-3.1-8B-Instruct") for c in ["D0", "D1", "D2"]}
for cond in ["D0", "D1", "D2"]:
    means = [round(sum(i for c, i in llama_pairs[cond][f]) / len(llama_pairs[cond][f]), 2) for f in FAMILIES]
    print("abs", cond, means)
for cond in ["D0", "D1", "D2"]:
    s = load_official(cond, "meta-llama_Llama-3.1-8B-Instruct")
    asr = [round(s['per_family'][f]['score_inflation_asr'], 2) for f in FAMILIES]
    flip = [round(s['per_family'][f]['decision_flip_asr'], 2) for f in FAMILIES]
    print("ASR ", cond, asr, "  flip", flip)

print("\nMcNemar D0 vs D2, Llama:")
d0f = load_local_pairs("D0", "meta-llama_Llama-3.1-8B-Instruct")
d2f = load_local_pairs("D2", "meta-llama_Llama-3.1-8B-Instruct")
def load_flips(cond, model_file):
    path = EVAL_DIR / f"{cond}_{model_file}_progress.jsonl"
    out = {f: {} for f in FAMILIES}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            r = rec["result"]
            if r.get("skipped"):
                continue
            aid = rec["arxiv_id"]
            for item in r["injected"]:
                if item["variant_type"] != "injection_only":
                    continue
                fam = item["family"]
                if fam in out:
                    out[fam][aid] = item["decision_flipped"]
    return out
d0flip = load_flips("D0", "meta-llama_Llama-3.1-8B-Instruct")
d2flip = load_flips("D2", "meta-llama_Llama-3.1-8B-Instruct")
for f in FAMILIES:
    common = set(d0flip[f]) & set(d2flip[f])
    b = sum(1 for a in common if d0flip[f][a] and not d2flip[f][a])
    c = sum(1 for a in common if not d0flip[f][a] and d2flip[f][a])
    n = b + c
    p = binomtest(min(b, c), n, 0.5).pvalue if n else 1.0
    print(f, f"b={b} c={c} p={p:.2e}")

print("\nEffect size, Llama D0:")
for f in FAMILIES:
    pairs = llama_pairs["D0"][f]
    clean = [c for c, i in pairs]
    inj = [i for c, i in pairs]
    try:
        _, p = wilcoxon([i - c for c, i in pairs])
    except ValueError:
        p = float("nan")
    r = rank_biserial(clean, inj)
    print(f, f"p={p:.2e}", f"r={r:.2f}")

print("\n############ TABLE 6: Frontier comparison (Gemini D0/D1 vs Qwen D2, injection_only) ############")
gem_pairs = {c: load_local_pairs(c, "gemini-flash-lite-latest") for c in ["D0", "D1"]}
for cond in ["D0", "D1"]:
    means = [round(sum(i for c, i in gem_pairs[cond][f]) / len(gem_pairs[cond][f]), 2) for f in FAMILIES]
    print("Gemini", cond, means)
print("Qwen D2 (repeat from above):", [round(sum(i for c, i in qwen_pairs['D2'][f]) / len(qwen_pairs['D2'][f]), 2) for f in FAMILIES])

print("\n############ TABLE 7: Long-form validation ############")
for cond in ["D0", "D1", "D2"]:
    path = EVAL_DIR / f"longform_progress_{cond}.jsonl"
    if not path.exists():
        print(cond, "MISSING FILE")
        continue
    fam_scores = {f: [] for f in FAMILIES}
    n_papers = 0
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            n_papers += 1
            for fam, v in rec["per_family"].items():
                if v["injected"] is not None:
                    fam_scores[fam].append(v["injected"])
    means = [round(sum(fam_scores[f]) / len(fam_scores[f]), 2) if fam_scores[f] else None for f in FAMILIES]
    print(cond, "n_papers=", n_papers, means)
