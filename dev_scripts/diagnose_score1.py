"""One-off diagnostic: regenerate the exact prompt that produced the
anomalous score=1 in longform D1 (paper 2608.05769v1, family F2,
injected variant), with full text saved this time, to determine whether
it's a real defensive overcorrection or a parsing artifact.

Usage (inside the container):
    python /workspace/scripts/diagnose_score1.py
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
TARGET_AID = "2608.05769v1"
TARGET_FAMILY = "F2"

def main():
    injected_text = None
    with open(BENCH_PATH) as f:
        for line in f:
            v = json.loads(line)
            if v["arxiv_id"] == TARGET_AID and v["family"] == TARGET_FAMILY and v["variant_type"] == "injection_only":
                injected_text = v["manuscript_text"]
                break
    assert injected_text is not None, "variant not found"

    gen = load_model_unsloth("Qwen/Qwen2.5-14B-Instruct")
    prompt = build_prompt(truncate_manuscript(injected_text), defense="D1", longform=True)
    out = generate(gen, prompt, max_new_tokens=600, do_sample=False)
    sd = parse_score_decision(out)
    print("=== PARSED ===")
    print(sd)
    print("=== FULL TEXT ===")
    print(out)
    Path("/workspace/outputs/eval/diagnose_score1_output.txt").write_text(out or "")
    print("\nWrote /workspace/outputs/eval/diagnose_score1_output.txt")

if __name__ == "__main__":
    main()
