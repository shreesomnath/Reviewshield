"""Test whether the last-position-of-long-sequence zero-logits bug is a
PyTorch SDPA (scaled_dot_product_attention) kernel issue, not a
quantization issue.

The bug has now reproduced identically across three configurations that
vary precision, quantization, AND GPU count: bf16+4bit (single GPU,
all-zero), fp32+4bit (single GPU, 37.5% Inf), and bf16+no-quant (split
across 2 GPUs, all-zero, and a real .generate() call crashed the same way).
The one constant across all three is a long sequence (5309 tokens) hitting
the LAST query position under transformers' default attention
implementation, which for recent transformers versions is SDPA unless
overridden. SDPA's fused/memory-efficient backends have documented
numerical edge cases for certain sequence-length/causal-mask combinations.

This test uses the cheapest reproducer config (bf16 compute, 4-bit
quantized, fits on a single GPU, fastest to load) and only changes
attn_implementation from the default to "eager" (the straightforward,
unfused reference implementation - slower, but without the optimized
kernel's edge cases).

Usage (inside the container):
    python /workspace/scripts/generation/diagnose_eager_attn.py
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    all_exactly_zero = bool((f32 == 0).all().item())
    if finite.numel() > 0:
        print(f"  [{label}] n={n_total} NaN={n_nan} Inf={n_inf} all_zero={all_exactly_zero} "
              f"min={finite.min().item():.6e} max={finite.max().item():.6e} "
              f"mean={finite.mean().item():.6e} abs_sum={finite.abs().sum().item():.6e}",
              flush=True)
    else:
        print(f"  [{label}] n={n_total} NaN={n_nan} Inf={n_inf} all_zero={all_exactly_zero} - NO finite values", flush=True)
    # a config only counts as genuinely healthy if there's no NaN/Inf AND
    # it isn't degenerately all-zero either (the mistake in earlier scripts)
    return n_nan == 0 and n_inf == 0 and not all_exactly_zero


def main():
    corpus_path = Path("/workspace/data/interim/arxiv_corpus.jsonl")
    with open(corpus_path) as f:
        rec = json.loads(f.readline())
    manuscript = truncate_manuscript(rec["full_text"])
    prompt = build_prompt(manuscript, defense=None)

    print(f"=== Loading {MODEL_ID}, 4-bit bf16 compute, attn_implementation='eager' ===", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto",
        torch_dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.eval()
    print(f"model.config._attn_implementation = {getattr(model.config, '_attn_implementation', 'unknown')}")

    messages = [{"role": "user", "content": prompt}]
    input_ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)
    print(f"input_ids shape: {input_ids.shape}")

    print("\n=== Raw forward pass, eager attention ===", flush=True)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    clean = report(out.logits[0, -1, :], "eager attn, pos=-1 (last)")

    if clean:
        print("\n=> CONFIRMED: eager attention fixes the last-position bug. "
              "Production fix: pass attn_implementation='eager' when loading "
              "the generator/judge model.")
    else:
        print("\n=> eager attention did NOT fix it either.")

    print("\n=== Testing actual generation ===", flush=True)
    try:
        with torch.no_grad():
            gen_out = model.generate(
                input_ids, attention_mask=attention_mask, max_new_tokens=80,
                do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(gen_out[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"GENERATION SUCCEEDED: {text[:300]!r}")
    except RuntimeError as e:
        print(f"GENERATION FAILED: {type(e).__name__}: {str(e)[:300]}")

    print("\n=== Diagnosis complete ===")


if __name__ == "__main__":
    main()
