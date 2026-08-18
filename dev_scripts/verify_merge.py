"""Correctness + speed check for merging the D2 LoRA adapter into the base
model weights (merge_and_unload()) instead of running it as a wrapped
PeftModel, to see if it removes the per-step PEFT/LoRA overhead observed
in the real D2 eval run (paper 1: 26.8min, paper 2: 20.1min, vs D0's
steady ~15.5-16.5min/paper with no adapter at all).

Two things checked:
  1. Does merging even work on our 4-bit-quantized base model? (bnb 4-bit
     linear layers don't always support in-place weight merging - this is
     a real open question, not assumed.) If not, falls back to loading
     the base in bf16 (matching how train_dpo.py trained it) and merging
     there instead.
  2. Does the merged model produce the same review as the unmerged
     PeftModel for the same prompt, and is it actually faster?

Usage (inside the container):
    python /workspace/scripts/verify_merge.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "generation"))
from model_utils import load_model_4bit, generate, truncate_manuscript, parse_score_decision
from review_prompt import build_prompt

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
ADAPTER_PATH = "/workspace/outputs/checkpoints/dpo_qwen14b_v2/final"
BENCH_PATH = "/workspace/data/processed/revguard_bench/test.jsonl"


def load_one_manuscript():
    with open(BENCH_PATH) as f:
        for line in f:
            v = json.loads(line)
            if v["variant_type"] == "clean":
                return v["manuscript_text"]


def main():
    manuscript = load_one_manuscript()
    prompt = build_prompt(truncate_manuscript(manuscript), defense=None, brief=True)

    print("=== Loading base (4-bit) + adapter (unmerged, current real approach) ===")
    gen = load_model_4bit(MODEL_ID)
    from peft import PeftModel
    gen.model = PeftModel.from_pretrained(gen.model, ADAPTER_PATH)

    t0 = time.time()
    unmerged_out = generate(gen, prompt, max_new_tokens=150, do_sample=False)
    t_unmerged = time.time() - t0
    print(f"Unmerged (current): {t_unmerged:.1f}s")
    print(f"  text: {unmerged_out[:150]!r}")

    print("\n=== Attempting merge_and_unload() on the 4-bit base ===")
    try:
        gen.model = gen.model.merge_and_unload()
        merge_worked = True
    except Exception as e:
        print(f"Merge on 4-bit FAILED: {type(e).__name__}: {e}")
        merge_worked = False

    if merge_worked:
        t0 = time.time()
        merged_out = generate(gen, prompt, max_new_tokens=150, do_sample=False)
        t_merged = time.time() - t0
        print(f"Merged (4-bit base): {t_merged:.1f}s")
        print(f"  text: {merged_out[:150]!r}")
        sd_u = parse_score_decision(unmerged_out)
        sd_m = parse_score_decision(merged_out)
        print(f"\nUnmerged: well_formed={sd_u['well_formed']} score={sd_u['score']} decision={sd_u['decision']}")
        print(f"Merged:   well_formed={sd_m['well_formed']} score={sd_m['score']} decision={sd_m['decision']}")
        print(f"Exact text match: {unmerged_out.strip() == merged_out.strip()}")
        print(f"Speedup: {t_unmerged/t_merged:.2f}x")
    else:
        print("Will need to fall back to a bf16 (non-4bit) base for merging - "
              "not tested in this run, report back before implementing.")


if __name__ == "__main__":
    main()
