"""Does the corruption appear in the STATIC prompt (length known and
already tested safe by the length sweep), or does it appear only as the
sequence GROWS during multi-token generation (prompt + newly generated
tokens together crossing some threshold)? generate_preference_pairs.py
crashed again even after truncating to 8,000 chars (2,227 prompt tokens,
well under the 3,379-token confirmed-healthy point from the length sweep)
- but CUDA errors surface asynchronously, so the traceback alone cannot
tell us whether the actual bad computation happened on the first generated
token or a later one.

This does MANUAL greedy decoding via repeated forward() calls with KV
caching and argmax token selection - never calling .generate() or
torch.multinomial, so it cannot trigger the device-side assert regardless
of outcome, and it reports the logits health at every single step, so the
exact failure point (if any) is directly visible.

Usage (inside the container):
    python /workspace/scripts/generation/diagnose_stepwise.py
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
MAX_STEPS = 100


def is_healthy(logits_1d):
    f32 = logits_1d.float()
    n_nan = torch.isnan(f32).sum().item()
    n_inf = torch.isinf(f32).sum().item()
    all_zero = bool((f32 == 0).all().item())
    abs_max = f32.abs().max().item()
    return (n_nan == 0 and n_inf == 0 and not all_zero and abs_max < 100), n_nan, n_inf, all_zero, abs_max


def main():
    # Use the exact same variant that crashed in production.
    bench_path = Path("/workspace/data/processed/revguard_bench/val.jsonl")
    with open(bench_path) as f:
        v = json.loads(f.readline())
    print(f"Using variant: {v['variant_type']} / {v['arxiv_id']}")
    manuscript = truncate_manuscript(v["manuscript_text"])
    prompt = build_prompt(manuscript, defense=None)

    print(f"=== Loading {MODEL_ID}, 4-bit bf16, single GPU (matches production config) ===", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    input_ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    print(f"initial prompt length: {input_ids.shape[1]} tokens")

    generated = input_ids
    past_key_values = None
    first_bad_step = None

    with torch.no_grad():
        for step in range(MAX_STEPS):
            if past_key_values is None:
                out = model(input_ids=generated, attention_mask=torch.ones_like(generated), use_cache=True)
            else:
                out = model(
                    input_ids=next_token, attention_mask=torch.ones_like(generated),
                    past_key_values=past_key_values, use_cache=True,
                )
            past_key_values = out.past_key_values
            last_logits = out.logits[0, -1, :]
            healthy, n_nan, n_inf, all_zero, abs_max = is_healthy(last_logits)

            if not healthy:
                print(f"  step {step} (total_len={generated.shape[1]}): CORRUPTED "
                      f"NaN={n_nan} Inf={n_inf} all_zero={all_zero} abs_max={abs_max:.4e}", flush=True)
                first_bad_step = step
                break
            else:
                if step % 10 == 0 or step < 5:
                    print(f"  step {step} (total_len={generated.shape[1]}): healthy, abs_max={abs_max:.4e}", flush=True)

            next_token = last_logits.argmax().view(1, 1)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tok.eos_token_id:
                print(f"  step {step}: EOS reached, stopping")
                break

    if first_bad_step is None:
        print(f"\n=> HEALTHY through all {MAX_STEPS} greedy steps "
              f"(final total length {generated.shape[1]} tokens). The bug did "
              f"NOT reproduce here via manual greedy decoding - suggests the "
              f"production crash may be specific to torch.multinomial's "
              f"sampling path (do_sample=True) rather than the underlying "
              f"logits themselves, or specific to something else not "
              f"replicated by this manual loop.")
    else:
        total_len_at_failure = input_ids.shape[1] + first_bad_step
        print(f"\n=> Corruption first appeared at generation step {first_bad_step}, "
              f"i.e. total sequence length {total_len_at_failure} tokens "
              f"(prompt was {input_ids.shape[1]} tokens). "
              f"{'This means the STATIC PROMPT ALONE is already unsafe.' if first_bad_step == 0 else 'This means the prompt alone is safe, but the CUMULATIVE length during generation (prompt + generated tokens) crossing this point is what breaks it - the fix must account for max_new_tokens, not just prompt length.'}")

    print("\n=== Diagnosis complete ===")


if __name__ == "__main__":
    main()
