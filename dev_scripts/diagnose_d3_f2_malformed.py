"""Diagnostic: regenerate the exact D3+F2 prompts that produced unparseable
output during the real eval run, using the identical settings run_eval.py
used (brief=True, EVAL_MAX_NEW_TOKENS=150, do_sample=False, D3's SFT
adapter), to see the actual raw text and understand why 6/50 papers
failed to produce a parseable SCORE line under this one combination.

Usage (inside the container):
    python /workspace/scripts/diagnose_d3_f2_malformed.py
"""
import json
import sys
from pathlib import Path

from unsloth import FastLanguageModel
sys.path.insert(0, str(Path("/workspace/scripts")))
sys.path.insert(0, str(Path("/workspace/scripts/generation")))
from model_utils import load_model_unsloth, generate, truncate_manuscript, parse_score_decision
from review_prompt import build_prompt

BENCH_PATH = "/workspace/data/processed/revguard_bench/test.jsonl"
TARGET_IDS = ["2608.05769v1", "2608.05588v1", "2608.05144v1", "2608.06009v1", "2608.05673v1", "2608.05893v1"]
EVAL_MAX_NEW_TOKENS = 150


def main():
    variants = {}
    with open(BENCH_PATH) as f:
        for line in f:
            v = json.loads(line)
            if v["arxiv_id"] in TARGET_IDS and v["family"] == "F2" and v["variant_type"] == "injection_only":
                variants[v["arxiv_id"]] = v["manuscript_text"]

    gen = load_model_unsloth("Qwen/Qwen2.5-14B-Instruct", adapter_path="/workspace/outputs/checkpoints/sft_qwen14b_v1/final")

    out_lines = []
    for aid in TARGET_IDS:
        manuscript = truncate_manuscript(variants[aid])
        prompt = build_prompt(manuscript, defense=None, brief=True)
        text = generate(gen, prompt, max_new_tokens=EVAL_MAX_NEW_TOKENS, do_sample=False)
        sd = parse_score_decision(text) if text else {"score": None, "decision": None, "well_formed": False}
        print(f"=== {aid} ===")
        print("well_formed:", sd["well_formed"], "score:", sd["score"])
        print("RAW TEXT:")
        print(repr(text))
        print()
        out_lines.append({"aid": aid, "well_formed": sd["well_formed"], "score": sd["score"], "text": text})

    Path("/workspace/outputs/eval/diagnose_d3_f2_malformed.json").write_text(json.dumps(out_lines, indent=2))
    print("Wrote diagnose_d3_f2_malformed.json")


if __name__ == "__main__":
    main()
