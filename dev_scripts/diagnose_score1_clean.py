"""Companion to diagnose_score1.py: regenerate the CLEAN (unattacked)
variant of the same paper (2608.05769v1), same D1/longform settings, so
the two full review texts can be compared side by side to see whether
the injected text's presence shifted the model's reading of the paper
even though it didn't comply with the injected instruction.

Usage (inside the container):
    python /workspace/scripts/diagnose_score1_clean.py
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

def main():
    clean_text = None
    with open(BENCH_PATH) as f:
        for line in f:
            v = json.loads(line)
            if v["arxiv_id"] == TARGET_AID and v["variant_type"] == "clean":
                clean_text = v["manuscript_text"]
                break
    assert clean_text is not None, "clean variant not found"

    gen = load_model_unsloth("Qwen/Qwen2.5-14B-Instruct")
    prompt = build_prompt(truncate_manuscript(clean_text), defense="D1", longform=True)
    out = generate(gen, prompt, max_new_tokens=600, do_sample=False)
    sd = parse_score_decision(out)
    print("=== PARSED ===")
    print(sd)
    print("=== FULL TEXT ===")
    print(out)
    Path("/workspace/outputs/eval/diagnose_score1_clean_output.txt").write_text(out or "")
    print("\nWrote /workspace/outputs/eval/diagnose_score1_clean_output.txt")

if __name__ == "__main__":
    main()
