"""Bisect: is the last-position corruption purely LENGTH-dependent (any
sufficiently long sequence triggers it), or specific to this one paper's
exact token content? Quantization, precision, attention implementation,
GPU count, and text degeneracy have all been ruled out by prior diagnostics
(see process.md Step 13's sub-entries) - the bug reproduces identically
regardless of all of those, always at the last position of the same
5309-token prompt.

Loads the model ONCE (cheapest reproducer config: bf16 compute, 4-bit,
single GPU, default attention), then runs forward passes (safe - never
calls .generate()/multinomial, cannot trigger the CUDA assert) over:
  (a) the SAME paper's manuscript truncated to several different lengths,
      to find whether there is a length threshold;
  (b) a DIFFERENT paper's manuscript at a similarly long length, to test
      whether the bug is specific to this one paper's content.

Usage (inside the container):
    python /workspace/scripts/generation/diagnose_length_sweep.py
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from review_prompt import build_prompt

MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"


def report(pos_logits, label):
    f32 = pos_logits.float()
    n_nan = torch.isnan(f32).sum().item()
    n_inf = torch.isinf(f32).sum().item()
    all_exactly_zero = bool((f32 == 0).all().item())
    abs_max = f32.abs().max().item()
    healthy = (n_nan == 0 and n_inf == 0 and not all_exactly_zero and abs_max < 100)
    print(f"  [{label}] NaN={n_nan} Inf={n_inf} all_zero={all_exactly_zero} "
          f"abs_max={abs_max:.4e}  -> {'HEALTHY' if healthy else 'CORRUPTED'}", flush=True)
    return healthy


def check_prompt(model, tok, text, label):
    prompt = build_prompt(text, defense=None)
    messages = [{"role": "user", "content": prompt}]
    input_ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    return report(out.logits[0, -1, :], f"{label} ({input_ids.shape[1]} tokens)")


def main():
    corpus_path = Path("/workspace/data/interim/arxiv_corpus.jsonl")
    with open(corpus_path) as f:
        rec_a = json.loads(f.readline())
        rec_b = json.loads(f.readline())

    print(f"=== Loading {MODEL_ID}, 4-bit bf16, single GPU, default attention ===", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model.eval()

    print("\n=== (a) Same paper, sweeping length ===", flush=True)
    full_a = rec_a["full_text"]
    for target_chars in [3000, 6000, 9000, 12000, 15000, 18000]:
        check_prompt(model, tok, full_a[:target_chars], f"paper A, first {target_chars} chars")

    print("\n=== (b) Different paper, similar long length ===", flush=True)
    full_b = rec_b["full_text"]
    # use the same head+tail truncation shape as production (model_utils.truncate_manuscript)
    max_chars = 18000
    head = int(max_chars * 0.75)
    tail = max_chars - head
    truncated_b = full_b[:head] + "\n\n[...manuscript truncated for length...]\n\n" + full_b[-tail:]
    check_prompt(model, tok, truncated_b, f"paper B ({rec_b['arxiv_id']}), production truncation")

    print("\n=== (c) Re-check paper A with the exact production truncation, for direct comparison ===", flush=True)
    truncated_a = full_a[:head] + "\n\n[...manuscript truncated for length...]\n\n" + full_a[-tail:]
    check_prompt(model, tok, truncated_a, f"paper A ({rec_a['arxiv_id']}), production truncation")

    print("\n=== Sweep complete ===")


if __name__ == "__main__":
    main()
