"""Decisive test: does the last-position-of-long-sequence zero/Inf logits
bug (see diagnose_generation.py) disappear WITHOUT 4-bit quantization at
all? Both the RoPE/context-length and special-token hypotheses were ruled
out by cheap config/tokenizer checks (max_position_embeddings=32768, no
rope_scaling, sequence is only 5309 tokens; the last tokens are just the
standard chat-template suffix). The bf16-compute and fp32-compute 4-bit
variants both failed differently (all-zero vs 37.5% Inf) at exactly the
same position, which points at the 4-bit quantization/dequantization path
itself rather than a general attention or precision issue.

Loads Qwen2.5-72B-Instruct in plain bf16 (no BitsAndBytesConfig), spread
across both GPUs via device_map="auto" (~144GB total, fits in the combined
~196GB across 2x RTX PRO 6000). If this is clean, the fix is simply: do not
4-bit-quantize the local generator/judge model.

Usage (inside the container):
    python /workspace/scripts/generation/diagnose_no_quant.py
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from review_prompt import build_prompt
from model_utils import truncate_manuscript

MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"


def report(pos_logits, label):
    f32 = pos_logits.float()
    n_nan = torch.isnan(f32).sum().item()
    n_inf = torch.isinf(f32).sum().item()
    n_total = f32.numel()
    finite = f32[torch.isfinite(f32)]
    if finite.numel() > 0:
        print(f"  [{label}] n={n_total} NaN={n_nan} Inf={n_inf} "
              f"min={finite.min().item():.6e} max={finite.max().item():.6e} "
              f"mean={finite.mean().item():.6e} abs_sum={finite.abs().sum().item():.6e}",
              flush=True)
    else:
        print(f"  [{label}] n={n_total} NaN={n_nan} Inf={n_inf} - NO finite values at all", flush=True)
    return n_nan == 0 and n_inf == 0


def main():
    corpus_path = Path("/workspace/data/interim/arxiv_corpus.jsonl")
    with open(corpus_path) as f:
        rec = json.loads(f.readline())
    manuscript = truncate_manuscript(rec["full_text"])
    prompt = build_prompt(manuscript, defense=None)

    print(f"=== Loading {MODEL_ID} in plain bf16, NO quantization, across both GPUs ===", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    input_ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)
    print(f"input_ids shape: {input_ids.shape}")

    print("\n=== Raw forward pass, no-quant bf16 ===", flush=True)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    clean = report(out.logits[0, -1, :], "no-quant bf16, pos=-1 (last)")

    if clean:
        print("\n=> CONFIRMED: without 4-bit quantization, the last-position "
              "logits are clean. The bug is specific to the 4-bit quantization/"
              "dequantization path (bitsandbytes NF4), not a general attention, "
              "RoPE, or transformers-version issue. Production fix: use plain "
              "bf16 (no BitsAndBytesConfig) for the local generator/judge model.")
    else:
        print("\n=> The bug persists even WITHOUT 4-bit quantization - not a "
              "bitsandbytes-specific issue. Needs further investigation "
              "(possibly a transformers/attention-kernel bug independent of "
              "quantization).")

    print("\n=== Testing actual generation (the real production call) ===", flush=True)
    try:
        with torch.no_grad():
            gen_out = model.generate(
                input_ids, attention_mask=attention_mask, max_new_tokens=60,
                do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(gen_out[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"GENERATION SUCCEEDED: {text[:200]!r}")
    except RuntimeError as e:
        print(f"GENERATION FAILED: {type(e).__name__}: {str(e)[:300]}")

    print("\n=== Diagnosis complete ===")


if __name__ == "__main__":
    main()
