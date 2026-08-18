"""One-off scan for malformed (unparseable score/decision) responses across
every model/condition/family combination, run to check whether the six
known D3+F2 malformed cases (see diagnose_d3_f2_malformed.py) were
isolated to that one condition or present elsewhere too. Not part of the
reproduction pipeline; a record of that check. Result at the time this was
run: zero malformed entries anywhere outside D3+F2.
"""
import json
from pathlib import Path

EVAL_DIR = Path("/workspace/outputs/eval")
FAMILIES = ["F1", "F2", "F3", "F4", "F5"]

COMBOS = [
    ("D3", "Qwen_Qwen2.5-14B-Instruct"),
    ("D0", "meta-llama_Llama-3.1-8B-Instruct"), ("D1", "meta-llama_Llama-3.1-8B-Instruct"),
    ("D2", "meta-llama_Llama-3.1-8B-Instruct"),
    ("D0", "gemini-flash-lite-latest"), ("D1", "gemini-flash-lite-latest"),
]

for cond, model_file in COMBOS:
    path = EVAL_DIR / f"{cond}_{model_file}_progress.jsonl"
    if not path.exists():
        print(cond, model_file, "FILE MISSING")
        continue
    counts = {f: [0, 0] for f in FAMILIES}  # [malformed, total]
    clean_malformed = 0
    clean_total = 0
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            r = rec["result"]
            if r.get("skipped"):
                continue
            clean_total += 1
            if not r.get("clean_well_formed", True):
                clean_malformed += 1
            for item in r["injected"]:
                if item["variant_type"] != "injection_only":
                    continue
                fam = item["family"]
                if fam not in counts:
                    continue
                counts[fam][1] += 1
                if item.get("clean_score") is None or item.get("injected_score") is None:
                    counts[fam][0] += 1
    row = " ".join(f"{f}:{counts[f][0]}/{counts[f][1]}" for f in FAMILIES)
    print(f"{cond:<4}{model_file:<40}{row}   clean_malformed={clean_malformed}/{clean_total}")

print("\n--- Qwen D0/D1/D2 (abs_score_progress_*.jsonl, different schema) ---")
for cond in ["D0", "D1", "D2"]:
    path = EVAL_DIR / f"abs_score_progress_{cond}.jsonl"
    counts = {f: [0, 0] for f in FAMILIES}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            for fam, d in rec["per_family"].items():
                if fam not in counts:
                    continue
                counts[fam][1] += 1
                if d.get("clean") is None or d.get("injected") is None:
                    counts[fam][0] += 1
    row = " ".join(f"{f}:{counts[f][0]}/{counts[f][1]}" for f in FAMILIES)
    print(f"{cond:<4}{row}")
