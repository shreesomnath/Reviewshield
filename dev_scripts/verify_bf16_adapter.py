"""Test whether loading the base model in bf16 (matching how train_dpo.py
actually trained the adapter - load_in_4bit=False) instead of 4-bit fixes
the garbage-generation problem seen when applying the D2 adapter on top of
a 4-bit-quantized base (verify_merge.py: repetitive/degenerate output,
well_formed=False on real eval papers too).

Usage (inside the container):
    python /workspace/scripts/verify_bf16_adapter.py
"""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "generation"))
from model_utils import generate, truncate_manuscript, parse_score_decision, LoadedModel
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

    print("=== Loading base in bf16 (NOT 4-bit) + adapter, matching train_dpo.py ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    gen = LoadedModel(model=model, tokenizer=tokenizer, model_id=MODEL_ID)

    t0 = time.time()
    out = generate(gen, prompt, max_new_tokens=150, do_sample=False)
    t_bf16 = time.time() - t0
    sd = parse_score_decision(out)
    print(f"bf16 + adapter: {t_bf16:.1f}s, well_formed={sd['well_formed']} score={sd['score']} decision={sd['decision']}")
    print(f"  text: {out[:300]!r}")


if __name__ == "__main__":
    main()
