"""Shared local-model loading and generation helpers.

One local model, Qwen2.5-72B-Instruct in 4-bit (plan.md sec:4/9), serves as
both the synthetic-data generator and the primary judge. Loading is
separated from generation so a single loaded model can be reused across many
prompts in a batch job without repeated multi-minute load times.
"""
import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

GENERATOR_MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
# Primary/secondary reviewer base models we defend (plan.md sec:5, DECIDED).
PRIMARY_REVIEWER_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
SECONDARY_REVIEWER_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    model_id: str


def load_model_4bit(model_id: str, device_map: str = "auto") -> LoadedModel:
    """Load any causal LM in 4-bit NF4 (bitsandbytes). Used for the 72B
    generator/judge; the 14B/8B reviewer bases are small enough to also load
    in bf16 directly when we want full-precision training (see train_dpo.py),
    but 4-bit is fine and memory-cheaper for pure inference/judging."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer, model_id=model_id)


def load_model_unsloth(model_id: str, adapter_path: str | None = None,
                        max_seq_length: int = 6000) -> LoadedModel:
    """Load a reviewer model (14B/8B) for local eval via Unsloth's
    FastLanguageModel instead of plain transformers AutoModelForCausalLM.

    Root cause (found via layer-by-layer activation hooking on a real
    failing prompt): plain transformers' attention computation on this
    model/transformers==4.56.2/RTX PRO 6000 Blackwell stack produces
    NaN/exploding logits starting in layer 0's self-attention output for
    ANY sufficiently long prompt (~500+ tokens) - reproduced identically
    across 4-bit, bf16, and fp32 precision, and across sdpa/eager/forced-
    math attention-kernel backends, so it is not a quantization or kernel-
    selection issue. This silently corrupted every local D0/D1/D2/D3 eval
    review at real manuscript lengths (all produced garbage/unparseable
    text, confirmed directly by inspecting real output, not just process-
    alive or file-progress signals - the earlier "19.5s/review confirmed
    usable" check upstream of this function only ever verified speed).

    Unsloth's FastLanguageModel, in contrast, uses its own patched
    attention/RoPE kernels and was independently verified (same layer-
    hooking method) to produce healthy, finite logits and real well-formed
    reviews at these same lengths - this is also the reason DPO training
    (train_dpo.py, which has always used FastLanguageModel) produced
    sensible, monotonically-improving loss/accuracy curves rather than the
    corruption this function's previous plain-transformers path exhibited.

    adapter_path may be a LoRA checkpoint dir (D2/D3): Unsloth loads it
    directly via model_name=adapter_path, since the checkpoint's own
    adapter_config.json already points back to the base model - verified
    directly to produce a real, adapter-specific (different-scoring)
    review, not a silent fallback to the unmodified base."""
    from unsloth import FastLanguageModel
    load_target = adapter_path if adapter_path else model_id
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_target, max_seq_length=max_seq_length, dtype=None, load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    return LoadedModel(model=model, tokenizer=tokenizer, model_id=model_id)


def generate(loaded: LoadedModel, prompt: str, max_new_tokens: int = 350,
             temperature: float = 0.7, do_sample: bool = False) -> str:
    # do_sample=False (greedy) AND use_cache=False (below) are both needed.
    # Root cause (process.md Steps 13-16), confirmed by a manual stepwise
    # decode: a single full forward pass over the prompt is healthy, but the
    # SECOND forward call - which reuses the cached KV state from the first,
    # exactly what generate()'s internal autoregressive loop always does -
    # corrupts to ~95% NaN logits on literally the very first incremental
    # step, independent of prompt length, quantization, precision, attention
    # implementation, or GPU count. This is a KV-cache-reuse bug on this
    # Qwen2.5-72B-Instruct + transformers==4.56.2 + torch 2.7.0 stack, not a
    # sampling-parameter issue - greedy decoding alone does not fix it, since
    # argmax over ~95%-NaN logits produces silently-wrong tokens rather than
    # a loud crash, which is worse than crashing. use_cache=False forces
    # every generation step to be a full, fresh forward pass (no cache
    # reuse), matching the confirmed-healthy "step 0" case every time, at
    # the cost of O(n^2) instead of O(n) generation cost - reduced
    # max_new_tokens from 700 to 350 to keep that cost practical. Revisit
    # both workarounds if the library stack is upgraded and this turns out
    # to be a since-fixed upstream bug.
    messages = [{"role": "user", "content": prompt}]
    input_ids = loaded.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(loaded.model.device)
    # Explicit attention_mask, not left for transformers to infer: when
    # pad_token_id == eos_token_id (true for Qwen2.5's tokenizer here),
    # unset attention_mask made generation numerically unstable on long
    # prompts with this 4-bit-quantized model - observed directly as a
    # `torch.multinomial` crash ("probability tensor contains inf, nan")
    # on the very first real generation call, with transformers' own
    # warning pointing at this exact cause. Kept as defense in depth.
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = loaded.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=False,
            temperature=temperature if do_sample else None,
            top_p=0.9 if do_sample else None,
            pad_token_id=loaded.tokenizer.eos_token_id,
        )
    completion = out[0][input_ids.shape[1]:]
    return loaded.tokenizer.decode(completion, skip_special_tokens=True)


def generate_batch(loaded: LoadedModel, prompts: list[str], max_new_tokens: int = 150,
                    temperature: float = 0.7, do_sample: bool = False) -> list[str]:
    """Same generation path as generate() (use_cache=False, same root-cause
    reasoning above), but processes multiple prompts in one model.generate()
    call. Under use_cache=False, cost is dominated by re-running a full
    forward pass over the whole sequence at every output step; batching
    several independent prompts together amortizes that per-step forward
    pass across all of them at once instead of paying it once per prompt,
    which is where the real wall-clock cost was going (confirmed via
    nvidia-smi showing GPU1 at only 0-48% utilization during sequential
    single-prompt eval - real spare capacity, not a saturated GPU).

    Left-padding is required for correct batched causal-LM generation (all
    real tokens must end at the same position so every sequence in the
    batch starts generating from the same relative step); right-padding
    (transformers' default) would corrupt generation for the padded
    sequences. Verified before use (see verify_batching.py) by checking
    batch-of-1 output matches generate()'s output exactly, and that a
    batch of different prompts doesn't cross-contaminate."""
    tok = loaded.tokenizer
    original_padding_side = tok.padding_side
    tok.padding_side = "left"
    try:
        texts = [
            tok.apply_chat_template([{"role": "user", "content": p}],
                                     add_generation_prompt=True, tokenize=False)
            for p in prompts
        ]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(loaded.model.device)
        with torch.no_grad():
            out = loaded.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                use_cache=False,
                temperature=temperature if do_sample else None,
                top_p=0.9 if do_sample else None,
                pad_token_id=tok.eos_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        completions = [
            tok.decode(out[i][input_len:], skip_special_tokens=True)
            for i in range(len(prompts))
        ]
        return completions
    finally:
        tok.padding_side = original_padding_side


def truncate_manuscript(text: str, max_chars: int = 8000) -> str:
    """Cap manuscript length so generation stays fast, within context, and
    (empirically) numerically stable on this model/transformers/torch stack.

    max_chars was reduced from an original 18000 after a real, reproduced
    bug: with Qwen2.5-72B-Instruct on transformers==4.56.2, prompts whose
    total token count crosses a threshold between ~3,400 and ~4,100 tokens
    produce corrupted (zero/Inf/overflowing) logits at the LAST sequence
    position specifically - exactly the position generate() samples the
    first new token from - reliably crashing generation. Confirmed via a
    length sweep + a second, unrelated paper at similar length (both
    corrupted at ~4,100-4,500 tokens; both healthy at <=3,379 tokens), and
    ruled out as a cause of anything else: reproduces identically regardless
    of 4-bit quantization, bf16 vs fp32 compute, eager vs default (SDPA)
    attention, and single- vs dual-GPU placement, and the input text itself
    was checked and is not garbled/degenerate. This points to a length-
    dependent bug in this specific software stack for this model, not a
    fundamental limitation (Qwen2.5-72B's own config declares a 32,768-token
    context window) - worth revisiting if the library stack is upgraded
    later, but not blocking now.

    8000 chars was chosen with real margin below the confirmed-healthy
    3,379-token point (~3.55 chars/token observed => ~2,250 manuscript
    tokens, plus review-instruction and chat-template overhead => roughly
    2,400 total prompt tokens, versus the failure band starting around
    3,400+).

    Keeps the head (title/abstract/intro, where most injections and the
    paper's core claims live) and a tail slice (conclusion, where the
    before_conclusion insertion point lands)."""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.75)
    tail = max_chars - head
    return text[:head] + "\n\n[...manuscript truncated for length...]\n\n" + text[-tail:]


def parse_score_decision(review_text: str) -> dict:
    """Extract SCORE/DECISION lines the review prompt requires. Returns None
    for a field if the model didn't follow the format."""
    score_m = re.search(r"SCORE:\s*(\d+)", review_text)
    decision_m = re.search(r"DECISION:\s*(Accept|Reject)", review_text, re.IGNORECASE)
    return {
        "score": int(score_m.group(1)) if score_m else None,
        "decision": decision_m.group(1).capitalize() if decision_m else None,
        "well_formed": bool(score_m and decision_m),
    }
