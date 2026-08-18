"""Generate DPO/SFT preference pairs from RevGuard-Bench's train/val splits
(plan.md sec:4, Day 6-8 milestone).

Design (see review_prompt.py's module docstring for the full reasoning):
for each injected manuscript, generate two review candidates with the SAME
local model but two different elicitation framings -
  - a defensive-framing candidate (explicit spotlighting/instruction-
    hierarchy prompt, review_prompt.build_prompt(..., defense='D1'))
  - a neutral-framing candidate (review_prompt.build_prompt(..., defense=None))
verify both programmatically (does it comply with the injected attack? does
it catch the planted flaw, if any?), and assign chosen/rejected based on
which candidate actually behaved better - NOT based on which framing
produced it. Pairs where both or neither candidate behaved well are
uninformative and are skipped (plan.md sec:8).

Critically, the STORED PROMPT for every pair is always the neutral framing,
so the trained model learns to resist injections without needing to be told
to - only the model weights should differ between D0 (undefended base) and
D2/D3 (our trained models) at evaluation time.

Clean manuscripts get a pair too, guarding against over-refusal (plan.md
sec:8): chosen = a normal thorough critical review, rejected = a
deliberately low-effort/lazy review, teaching the model that resistance to
injection should not come at the cost of reviewing clean papers properly.

Usage (inside the container):
    python /workspace/scripts/generation/generate_preference_pairs.py \
        --split train --limit 20 --generator-model-id Qwen/Qwen2.5-72B-Instruct

For a fast mechanics-only smoke test before committing to the 72B download,
pass a smaller --generator-model-id (e.g. Qwen/Qwen2.5-7B-Instruct); nothing
else in the pipeline changes.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from model_utils import GENERATOR_MODEL_ID, load_model_4bit, truncate_manuscript
from model_utils import generate as local_generate
from frontier_client import DEFAULT_MODEL_ID as FRONTIER_MODEL_ID
from frontier_client import generate as frontier_generate
from frontier_client import load_frontier_model
from response_judge import compliance_heuristic, judge_flaw_detected
from review_prompt import build_prompt

BENCH_DIR = Path("/workspace/data/processed/revguard_bench")
OUT_DIR = Path("/workspace/data/processed/preference_pairs")

# Empirical, not theoretical: derived from real spend on the first
# train-split run, which hit a hard credit-exhaustion stop at item
# 3305/4392, having processed 3929 raw items (train + the fully-completed
# val split) for the account's $15 prepaid credit. $15 / 3929 items ~=
# $0.00382/item. The original theoretical per-token estimate before this
# run (~$3-4 for the whole dataset) was wrong by about 5x - this constant
# exists so the running total printed during generation is grounded in
# what actually happened, not another guess.
EST_COST_PER_RAW_ITEM = 15.0 / 3929

_LAZY_REVIEW_PROMPT_SUFFIX = (
    "\n\n(Write a very brief, low-effort review: 1-2 generic sentences, no "
    "specific critique of methodology or evidence, just a superficial "
    "impression.)"
)


def process_injection_variant(gen, judge, variant: dict, generate_fn) -> dict | None:
    manuscript = truncate_manuscript(variant["manuscript_text"])
    neutral_prompt = build_prompt(manuscript, defense=None)
    defensive_elicit_prompt = build_prompt(manuscript, defense="D1")

    candidate_neutral = generate_fn(gen, neutral_prompt)
    candidate_defensive = generate_fn(gen, defensive_elicit_prompt)

    # Guard against a real edge case, not yet observed but possible at
    # scale: if ONE candidate is blocked/empty (a skipped frontier call)
    # while the other is genuinely good, the logic below would otherwise
    # pair a real "chosen" against an empty-string "rejected" - a
    # degenerate DPO example. Both candidates must be non-empty to proceed
    # at all.
    if not candidate_neutral or not candidate_defensive:
        return None

    def is_good(text: str) -> bool:
        if not text:  # empty = a skipped/blocked frontier call; never "good"
            return False
        complied = compliance_heuristic(text, variant["target_behavior"])
        if complied:
            return False
        if variant["flag_description"] is not None:
            caught = judge_flaw_detected(judge, text, variant["flag_description"], generate_fn)
            if not caught:
                return False
        return True

    neutral_good = is_good(candidate_neutral)
    defensive_good = is_good(candidate_defensive)

    if defensive_good and not neutral_good:
        chosen, rejected = candidate_defensive, candidate_neutral
    elif neutral_good and not defensive_good:
        chosen, rejected = candidate_neutral, candidate_defensive
    else:
        return None  # both good or both bad: uninformative, skip

    return {
        "prompt": neutral_prompt,  # always the neutral framing (see module docstring)
        "chosen": chosen,
        "rejected": rejected,
        "arxiv_id": variant["arxiv_id"],
        "family": variant["family"],
        "attack_variant": variant["attack_variant"],
        "variant_type": variant["variant_type"],
        "flaw_type": variant["flaw_type"],
    }


def process_clean_variant(gen, variant: dict, generate_fn) -> dict | None:
    manuscript = truncate_manuscript(variant["manuscript_text"])
    neutral_prompt = build_prompt(manuscript, defense=None)
    thorough = generate_fn(gen, neutral_prompt)
    lazy = generate_fn(gen, neutral_prompt + _LAZY_REVIEW_PROMPT_SUFFIX)
    if not thorough or not lazy:  # a skipped/blocked frontier call: uninformative
        return None
    return {
        "prompt": neutral_prompt,
        "chosen": thorough,
        "rejected": lazy,
        "arxiv_id": variant["arxiv_id"],
        "family": None,
        "attack_variant": None,
        "variant_type": "clean_anti_over_refusal",
        "flaw_type": None,
    }


def main(split: str, limit: int | None, generator_model_id: str, backend: str,
         out_suffix: str, start_index: int):
    in_path = BENCH_DIR / f"{split}.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{split}_pairs{out_suffix}.jsonl"

    variants = []
    with open(in_path) as f:
        for line in f:
            variants.append(json.loads(line))
    total_n = len(variants)
    # --start-index resumes after an interruption WITHOUT re-processing
    # (and re-paying for) items already done: appending to out_path alone
    # only prevents losing already-written pairs, it does not by itself
    # skip re-iterating the front of the list on a plain re-run. Read the
    # exact stopping index from the previous run's log
    # ("[N/<total>]" progress lines) and pass it here.
    if start_index:
        variants = variants[start_index:]
        print(f"Resuming from index {start_index} ({len(variants)} of {total_n} remaining)")
    if limit:
        variants = variants[:limit]

    print(f"Backend: {backend}  Loading generator/judge model: {generator_model_id}")
    if backend == "frontier":
        gen = load_frontier_model(generator_model_id)
        generate_fn = frontier_generate
    else:
        gen = load_model_4bit(generator_model_id)
        generate_fn = local_generate
    judge = gen  # same loaded model serves both roles (plan.md sec:4, DECIDED)

    n_pairs, n_skipped = 0, 0
    # 'a' (append), not 'w': re-running after an interruption should not
    # discard already-generated (already-paid-for, for the frontier backend)
    # pairs. Flushed after every single write, not just at file close, so a
    # crash or interruption never loses pairs that were already generated.
    last_i = start_index + len(variants) - 1
    with open(out_path, "a") as out_f:
        for i, v in enumerate(variants, start=start_index):
            if v["variant_type"] == "clean":
                pair = process_clean_variant(gen, v, generate_fn)
            elif v["variant_type"] in ("injection_only", "injection_plus_flaw"):
                pair = process_injection_variant(gen, judge, v, generate_fn)
            else:
                continue  # flaw_only variants are for eval (FDR), not DPO pairs

            if pair is None:
                n_skipped += 1
                continue
            out_f.write(json.dumps(pair) + "\n")
            out_f.flush()
            n_pairs += 1
            if n_pairs % 5 == 0 or i == last_i:
                est_cost = (i + 1 - start_index) * EST_COST_PER_RAW_ITEM
                print(f"  [{i+1}/{total_n}] {n_pairs} pairs written, "
                      f"{n_skipped} skipped (uninformative/blocked), "
                      f"~${est_cost:.2f} est. spend this run", flush=True)

    print(f"Done. {n_pairs} preference pairs -> {out_path} ({n_skipped} skipped)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend", choices=["local", "frontier"], default="frontier")
    ap.add_argument("--generator-model-id", type=str, default=None)
    ap.add_argument("--out-suffix", type=str, default="",
                     help="appended to the output filename, e.g. '_test' for throwaway runs")
    ap.add_argument("--start-index", type=int, default=0,
                     help="skip this many raw variants from the front (resume after an "
                          "interruption without re-processing/re-paying for already-done items)")
    args = ap.parse_args()
    default_model = FRONTIER_MODEL_ID if args.backend == "frontier" else GENERATOR_MODEL_ID
    main(args.split, args.limit, args.generator_model_id or default_model, args.backend,
         args.out_suffix, args.start_index)
