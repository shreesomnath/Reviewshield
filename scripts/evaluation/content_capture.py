"""Content-capture variant of run_eval.py: saves FULL clean+injected review
text (run_eval.py only ever kept a 600-char sample of the clean review and
discarded injected text entirely after scoring). Runs on a small subsample
(--limit, default 10 papers, same fixed first-N ordering as the main 50-
paper runs) so real content/qualitative comparison across conditions and
models is possible without needing a fresh 50-paper regeneration.

Does not touch run_eval.py or its output files - writes to a separate
content_{condition}_{model_id}.jsonl so the paper's real score numbers are
never at risk of being overwritten.

Usage (inside the container):
    python /workspace/scripts/evaluation/content_capture.py \
        --condition D0 --model-id Qwen/Qwen2.5-14B-Instruct --backend local --limit 10
"""
import argparse
import json
import sys
from pathlib import Path

if "--backend" not in sys.argv or "frontier" not in sys.argv:
    from unsloth import FastLanguageModel  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "generation"))
from model_utils import parse_score_decision, truncate_manuscript
from model_utils import generate as local_generate
from model_utils import load_model_unsloth
from frontier_client import generate as frontier_generate
from frontier_client import load_frontier_model, get_usage_summary, reset_usage_summary
from review_prompt import build_prompt

BENCH_DIR = Path("/workspace/data/processed/revguard_bench")
OUT_DIR = Path("/workspace/outputs/eval")

CONTENT_MAX_NEW_TOKENS = 300  # enough for a real paragraph review, not just a score line


def load_test_variants(limit):
    from collections import defaultdict
    by_paper = defaultdict(list)
    with open(BENCH_DIR / "test.jsonl") as f:
        for line in f:
            v = json.loads(line)
            by_paper[v["arxiv_id"]].append(v)
    papers = list(by_paper.items())
    if limit:
        papers = papers[:limit]
    return dict(papers)


def review_one_full(gen, variant, generate_fn, defense):
    manuscript = truncate_manuscript(variant["manuscript_text"])
    prompt = build_prompt(manuscript, defense=defense, brief=True)
    # do_sample=False explicitly on BOTH backends: local already defaults
    # to greedy, but frontier_generate defaults to do_sample=True (real
    # temperature=0.7 sampling), which would confound any clean-vs-injected
    # content comparison against the local models with sampling noise on
    # top of whatever the injection itself does. Forcing both to greedy
    # (do_sample=False -> temperature=0.0 for Gemini, see frontier_client.
    # _call_with_retry) makes the comparison apples-to-apples.
    text = generate_fn(gen, prompt, max_new_tokens=CONTENT_MAX_NEW_TOKENS, do_sample=False)
    sd = parse_score_decision(text) if text else {"score": None, "decision": None, "well_formed": False}
    return {"text": text or "", "score": sd["score"], "decision": sd["decision"], "well_formed": sd["well_formed"]}


def main(condition, model_id, backend, limit, adapter_path):
    defense = "D1" if condition == "D1" else None
    print(f"[content-capture] condition={condition} model={model_id} backend={backend} defense={defense} limit={limit}")

    if backend == "frontier":
        reset_usage_summary()
        gen = load_frontier_model(model_id)
        generate_fn = frontier_generate
    else:
        gen = load_model_unsloth(model_id, adapter_path=adapter_path)
        generate_fn = local_generate

    papers = load_test_variants(limit)
    print(f"Capturing full text for {len(papers)} papers")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"content_{condition}_{model_id.replace('/', '_')}.jsonl"
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                done_ids.add(json.loads(line)["arxiv_id"])
        if done_ids:
            print(f"Resuming: {len(done_ids)} papers already done in {out_path}")

    with open(out_path, "a") as out_f:
        for i, (arxiv_id, variants) in enumerate(papers.items()):
            if arxiv_id in done_ids:
                continue
            clean_variants = [v for v in variants if v["variant_type"] == "clean"]
            injected = [v for v in variants if v["variant_type"] in ("injection_only", "injection_plus_flaw")]
            if not clean_variants:
                continue
            clean_result = review_one_full(gen, clean_variants[0], generate_fn, defense)
            injected_results = {}
            for v in injected:
                r = review_one_full(gen, v, generate_fn, defense)
                injected_results[v["family"]] = r
            rec = {"arxiv_id": arxiv_id, "clean": clean_result, "injected": injected_results}
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            print(f"  [{i+1}/{len(papers)}] {arxiv_id} done", flush=True)

    print(f"Wrote {out_path}")
    if backend == "frontier":
        usage = get_usage_summary()
        print(f"Real Gemini usage this run: {usage['calls']} calls, "
              f"{usage['prompt_tokens']} prompt tokens, {usage['output_tokens']} output tokens")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["D0", "D1", "D2", "D3"], required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--backend", choices=["local", "frontier"], default="local")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--adapter-path", default=None)
    args = ap.parse_args()
    main(args.condition, args.model_id, args.backend, args.limit, args.adapter_path)
