"""Rigorous root-cause diagnostic for the garbage-output bug found in real
eval (brief-mode, 150-token, use_cache=False local generation on the real
truncated-manuscript prompt). Reuses the same manual-stepwise method that
found the original KV-cache-reuse bug: check logits health at each
generation step directly, instead of trusting generate()'s sampled output.

Usage (inside the container):
    python /workspace/scripts/diagnose_garbage.py
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "generation"))
from model_utils import load_model_4bit, truncate_manuscript
from review_prompt import build_prompt

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
BENCH_PATH = "/workspace/data/processed/revguard_bench/test.jsonl"


def load_manuscript(aid):
    with open(BENCH_PATH) as f:
        for line in f:
            v = json.loads(line)
            if v["variant_type"] == "clean" and v["arxiv_id"] == aid:
                return v["manuscript_text"]


def main():
    manuscript = load_manuscript("2608.04243v1")
    prompt = build_prompt(truncate_manuscript(manuscript), defense=None, brief=True)
    print(f"Prompt length (chars): {len(prompt)}")

    gen = load_model_4bit(MODEL_ID)
    messages = [{"role": "user", "content": prompt}]
    input_ids = gen.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(gen.model.device)
    print(f"Prompt length (tokens): {input_ids.shape[1]}")
    attention_mask = torch.ones_like(input_ids)

    generated = input_ids
    with torch.no_grad():
        for step in range(15):
            out = gen.model(
                input_ids=generated,
                attention_mask=torch.ones_like(generated),
                use_cache=False,
            )
            logits = out.logits[0, -1, :]  # last position's logits
            n_nan = torch.isnan(logits).sum().item()
            n_inf = torch.isinf(logits).sum().item()
            finite = logits[torch.isfinite(logits)]
            max_finite = finite.max().item() if finite.numel() else float("nan")
            min_finite = finite.min().item() if finite.numel() else float("nan")
            next_token = torch.argmax(logits).item()
            next_token_text = gen.tokenizer.decode([next_token])
            print(f"step={step} nan={n_nan} inf={n_inf} max_finite={max_finite:.2f} "
                  f"min_finite={min_finite:.2f} argmax_token={next_token!r} "
                  f"({next_token_text!r})")
            generated = torch.cat([generated, torch.tensor([[next_token]], device=generated.device)], dim=1)


if __name__ == "__main__":
    main()
