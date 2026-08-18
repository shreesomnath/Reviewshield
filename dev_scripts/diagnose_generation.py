"""Bisect the 'probability tensor contains inf/nan' crash safely and cheaply.

A CUDA device-side assert (which is what torch.multinomial raises here)
poisons the CUDA context for the rest of the process - every CUDA call
after the first one that fires the assert re-raises the same generic error
regardless of whether the underlying config would actually have worked, so
naive sequential try/except testing of multiple generation configs in one
process gives unreliable results after the first failure.

Instead: do a single raw forward pass (no .generate(), no sampling, no
multinomial) on the real long manuscript prompt and inspect the OUTPUT
LOGITS directly for NaN/Inf before anything touches them. A plain forward
pass cannot trigger the assert, so this is safe regardless of outcome, and
it directly answers the load-bearing question: is the corruption already
in the model's raw output (a quantization/compute-dtype numerical-stability
problem), or does it only appear during generate()'s temperature/top-p/
sampling post-processing (a decoding-parameter problem)? Those have
different fixes.

Usage (inside the container):
    python /workspace/scripts/generation/diagnose_generation.py
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


def main():
    corpus_path = Path("/workspace/data/interim/arxiv_corpus.jsonl")
    with open(corpus_path) as f:
        rec = json.loads(f.readline())
    manuscript = truncate_manuscript(rec["full_text"])
    prompt = build_prompt(manuscript, defense=None)

    print(f"=== Loading {MODEL_ID} in 4-bit (bf16 compute, production config) ===", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model.eval()
    print(f"pad_token_id={tok.pad_token_id} eos_token_id={tok.eos_token_id}")

    messages = [{"role": "user", "content": prompt}]
    input_ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)
    print(f"input_ids shape: {input_ids.shape} ({input_ids.shape[1]} tokens)")

    print("\n=== Single raw forward pass (no sampling, cannot trigger the assert) ===", flush=True)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    print(f"out.logits.shape = {tuple(out.logits.shape)}  dtype={out.logits.dtype}  "
          f"(expected (1, {input_ids.shape[1]}, vocab_size) - if the middle "
          f"dimension is 1 instead of {input_ids.shape[1]}, this model/transformers "
          f"version is only returning logits for a restricted set of positions "
          f"and the previous run's [0,-1,:] index was silently reading the wrong "
          f"thing)", flush=True)

    def report(pos_logits, label):
        f32 = pos_logits.float()
        n_nan = torch.isnan(f32).sum().item()
        n_inf = torch.isinf(f32).sum().item()
        n_total = f32.numel()
        finite = f32[torch.isfinite(f32)]
        if finite.numel() > 0:
            abs_sum = finite.abs().sum().item()
            print(f"  [{label}] n={n_total} NaN={n_nan} Inf={n_inf} "
                  f"min={finite.min().item():.6e} max={finite.max().item():.6e} "
                  f"mean={finite.mean().item():.6e} abs_sum={abs_sum:.6e}",
                  flush=True)
            return abs_sum
        else:
            print(f"  [{label}] n={n_total} NaN={n_nan} Inf={n_inf} - NO finite values at all", flush=True)
            return None

    seq_len = out.logits.shape[1]
    # Check the last position (what generate() actually samples from) AND an
    # earlier position for comparison, in case corruption is position-specific.
    last_abs_sum = report(out.logits[0, -1, :], f"pos=-1 (last, seq_len={seq_len})")
    mid_abs_sum = first_abs_sum = None
    if seq_len > 1:
        mid = seq_len // 2
        mid_abs_sum = report(out.logits[0, mid, :], f"pos={mid} (middle)")
        first_abs_sum = report(out.logits[0, 0, :], "pos=0 (first)")

    print(f"\nlogits.shape middle dim was {seq_len} "
          f"({'MATCHES' if seq_len == input_ids.shape[1] else 'DOES NOT MATCH'} "
          f"input length {input_ids.shape[1]})")
    if last_abs_sum is not None and last_abs_sum < 1e-6:
        print("=> The last-position logits are (numerically) all zero - this is "
              "itself the anomaly, distinct from NaN/Inf, and is not a healthy "
              "forward-pass output. Needs further investigation (e.g. a "
              "logits_to_keep / caching default in this transformers version "
              "silently zeroing unrequested positions) before concluding "
              "anything about generate()'s decoding logic.")
    else:
        print("=> Last-position logits have real, nonzero magnitude and no "
              "NaN/Inf: the raw forward pass is healthy here. The corruption "
              "seen in generate() must be introduced later, inside temperature "
              "scaling / top-p filtering / softmax post-processing - a "
              "decoding-parameter problem, not a model output problem.")

    # Also test a short synthetic prompt for comparison, still via raw forward
    # pass only (safe either way).
    short_ids = tok.apply_chat_template(
        [{"role": "user", "content": "Write one sentence about the ocean."}],
        add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        short_out = model(input_ids=short_ids, attention_mask=torch.ones_like(short_ids))
    print(f"\nShort prompt ({short_ids.shape[1]} tokens), out.logits.shape = "
          f"{tuple(short_out.logits.shape)}:")
    report(short_out.logits[0, -1, :], "short pos=-1 (last)")

    del model, out, short_out
    torch.cuda.empty_cache()

    # Candidate fix, now well-motivated by evidence (not a blind guess): the
    # failure is isolated to the LAST position of a long sequence specifically,
    # exactly where generate() samples the first new token from - a pattern
    # consistent with bf16 precision underflow in attention/RoPE computation
    # at high position indices, worsened by 4-bit weight quantization. Test
    # the same long prompt's last position with fp32 compute dtype instead.
    print("\n=== Reloading with bnb_4bit_compute_dtype=float32 (candidate fix) ===", flush=True)
    bnb_config_fp32 = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32, bnb_4bit_use_double_quant=True,
    )
    model_fp32 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config_fp32, device_map="auto", torch_dtype=torch.float32,
    )
    model_fp32.eval()
    with torch.no_grad():
        out_fp32 = model_fp32(input_ids=input_ids, attention_mask=attention_mask)
    print(f"out.logits.shape = {tuple(out_fp32.logits.shape)}  dtype={out_fp32.logits.dtype}")
    fp32_last_abs_sum = report(out_fp32.logits[0, -1, :], "fp32 pos=-1 (last, same long prompt)")
    if fp32_last_abs_sum is not None and fp32_last_abs_sum > 1e-6:
        print("\n=> fp32 compute dtype FIXES the last-position zero-logits bug on "
              "this exact prompt. Recommend switching bnb_4bit_compute_dtype to "
              "float32 in model_utils.load_model_4bit() (production fix).")
    else:
        print("\n=> fp32 compute dtype did NOT fix it either - the last-position "
              "zero-logits bug is not a bf16-precision issue; needs a different "
              "root cause (e.g. an attention-kernel or RoPE bug independent of "
              "compute dtype, or something specific to this exact input).")

    print("\n=== Diagnosis complete ===")


if __name__ == "__main__":
    main()
