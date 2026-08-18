"""One-off correctness check for model_utils.generate_batch(), run before
trusting it in the real eval pipeline. Checks two things:
  1. batch-of-1 output is IDENTICAL to the existing single-item generate()
     (both greedy/do_sample=False, so this must match exactly - any
     difference means the batching path has a real bug, e.g. bad padding).
  2. a batch of 3 different real prompts produces 3 different, sane
     completions with no cross-contamination between sequences.

Usage (inside the container):
    python /workspace/scripts/verify_batching.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "generation"))
from model_utils import load_model_4bit, generate, generate_batch, truncate_manuscript

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

PROMPTS = [
    "Reply with exactly the single word: APPLE",
    "Reply with exactly the single word: BANANA",
    "Reply with exactly the single word: CHERRY",
]


def main():
    print(f"Loading {MODEL_ID}...")
    loaded = load_model_4bit(MODEL_ID)

    print("\n=== Test 1: batch-of-1 vs single-item generate() ===")
    single_out = generate(loaded, PROMPTS[0], max_new_tokens=20, do_sample=False)
    batch1_out = generate_batch(loaded, [PROMPTS[0]], max_new_tokens=20, do_sample=False)[0]
    print(f"single:  {single_out!r}")
    print(f"batch-1: {batch1_out!r}")
    match = single_out.strip() == batch1_out.strip()
    print(f"MATCH: {match}")
    if not match:
        print("FAIL - batching path diverges from the known-good single-item path, do not use yet")
        sys.exit(1)

    print("\n=== Test 2: batch of 3 different prompts, check no cross-contamination ===")
    batch3_out = generate_batch(loaded, PROMPTS, max_new_tokens=20, do_sample=False)
    for p, o in zip(PROMPTS, batch3_out):
        print(f"prompt={p!r} -> output={o!r}")
    all_different = len(set(o.strip() for o in batch3_out)) == len(batch3_out)
    print(f"ALL DIFFERENT (no cross-contamination): {all_different}")
    if not all_different:
        print("FAIL - outputs are not distinct, possible cross-contamination in batched decode")
        sys.exit(1)

    print("\n=== Test 3: realistic-length prompt (real truncated manuscript-sized) ===")
    import time
    fake_manuscript = ("This paper proposes a novel method. " * 400)  # ~15k chars before truncation
    truncated = truncate_manuscript(fake_manuscript, max_chars=8000)
    real_prompt = f"Summarize this in one sentence:\n\n{truncated}"
    t0 = time.time()
    single_real = generate(loaded, real_prompt, max_new_tokens=150, do_sample=False)
    t_single = time.time() - t0
    t0 = time.time()
    batch_real = generate_batch(loaded, [real_prompt] * 4, max_new_tokens=150, do_sample=False)
    t_batch4 = time.time() - t0
    print(f"single-item time (1 review): {t_single:.1f}s")
    print(f"batch-of-4 time (4 reviews): {t_batch4:.1f}s -> {t_batch4/4:.1f}s/review effective")
    print(f"speedup: {(t_single*4)/t_batch4:.2f}x")
    same_as_single = single_real.strip() == batch_real[0].strip()
    print(f"batch item 0 matches single-item output: {same_as_single}")

    print("\nALL CHECKS PASSED" if match and all_different else "\nCHECKS FAILED")


if __name__ == "__main__":
    main()
