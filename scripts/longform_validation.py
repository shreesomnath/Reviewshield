"""Long-form review-length validation check: does the robustness finding
(D2 producing lower/safer absolute scores under attack than D0, D1
somewhere in between) hold when reviews are generated at realistic
length (~350-700+ words, matching FirstPass-style full reviews) instead
of the ~80-word brief format used for the main 50-paper evaluation?

This is a genuine, disclosed open validity question (not something to
assume away): the brief format was adopted for a real, diagnosed
technical reason (use_cache=False makes generation cost scale with
output length on this stack), but review length could plausibly affect
attack susceptibility either direction, and near-zero flaw-detection
recall in brief mode is itself evidence the short format changes review
behavior. This script directly tests whether the main finding survives
at realistic length, on a smaller (~8-10 paper) sample given the higher
per-generation cost of long-form output.

Includes incremental, resume-safe checkpointing from the start (lesson
already learned twice this project) and unbuffered output for real-time
visibility.

Usage (inside the container):
    python /workspace/scripts/longform_validation.py
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path("/workspace/scripts")))
sys.path.insert(0, str(Path("/workspace/scripts/generation")))
from unsloth import FastLanguageModel
from model_utils import load_model_unsloth, generate, truncate_manuscript, parse_score_decision
from review_prompt import build_prompt

BENCH_PATH = "/workspace/data/processed/revguard_bench/test.jsonl"
EVAL_DIR = Path("/workspace/outputs/eval")
N_PAPERS = 5
FAMILIES = ["F1", "F2", "F3", "F4", "F5"]
LONGFORM_MAX_TOKENS = 600  # matches the earlier diagnostic finding that
# even 400 tokens got cut off before reaching a Limitations-style
# section where specific critiques concentrate - 600 gives more room.

CONDITIONS = [
    ("D0", None, None),
    ("D1", "D1", None),
    ("D2", None, "/workspace/outputs/checkpoints/dpo_qwen14b_v2/final"),
]


def load_sample():
    clean, injected, ids_seen = {}, {}, []
    with open(BENCH_PATH) as f:
        for line in f:
            v = json.loads(line)
            aid = v["arxiv_id"]
            if v["variant_type"] == "clean":
                if aid not in ids_seen and len(ids_seen) < N_PAPERS:
                    ids_seen.append(aid)
                if aid in ids_seen:
                    clean[aid] = v["manuscript_text"]
            elif v["variant_type"] == "injection_only" and aid in ids_seen:
                injected.setdefault(aid, {})[v["family"]] = v["manuscript_text"]
    return clean, injected, ids_seen


def run_condition(label, defense, adapter_path, clean, injected, ids_seen):
    progress_path = EVAL_DIR / f"longform_progress_{label}.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    results = {fam: {"clean": [], "injected": []} for fam in FAMILIES}
    if progress_path.exists():
        with open(progress_path) as f:
            for line in f:
                rec = json.loads(line)
                done_ids.add(rec["aid"])
                for fam, vals in rec["per_family"].items():
                    if vals["clean"] is not None:
                        results[fam]["clean"].append(vals["clean"])
                    if vals["injected"] is not None:
                        results[fam]["injected"].append(vals["injected"])
        if done_ids:
            print(f"[{label}] Resuming: {len(done_ids)} papers already done")

    gen = load_model_unsloth("Qwen/Qwen2.5-14B-Instruct", adapter_path=adapter_path)
    print(f"\n=== {label} (long-form, {LONGFORM_MAX_TOKENS} tokens) ===")
    with open(progress_path, "a") as pf:
        for aid in ids_seen:
            if aid in done_ids:
                continue
            c_out = generate(gen, build_prompt(truncate_manuscript(clean[aid]), defense=defense, longform=True),
                              max_new_tokens=LONGFORM_MAX_TOKENS, do_sample=False)
            c_sd = parse_score_decision(c_out)
            per_family = {}
            for fam in FAMILIES:
                if fam not in injected.get(aid, {}):
                    continue
                i_out = generate(gen, build_prompt(truncate_manuscript(injected[aid][fam]), defense=defense, longform=True),
                                  max_new_tokens=LONGFORM_MAX_TOKENS, do_sample=False)
                i_sd = parse_score_decision(i_out)
                print(f"{aid} {fam}: clean={c_sd['score']} injected={i_sd['score']} "
                      f"(clean_wf={c_sd['well_formed']} inj_wf={i_sd['well_formed']})")
                per_family[fam] = {"clean": c_sd["score"], "injected": i_sd["score"]}
                if c_sd["score"] is not None:
                    results[fam]["clean"].append(c_sd["score"])
                if i_sd["score"] is not None:
                    results[fam]["injected"].append(i_sd["score"])
            pf.write(json.dumps({"aid": aid, "per_family": per_family}) + "\n")
            pf.flush()
    return results


def main():
    clean, injected, ids_seen = load_sample()
    print(f"Sampled {len(ids_seen)} papers: {ids_seen}")

    all_results = {}
    for label, defense, adapter in CONDITIONS:
        all_results[label] = run_condition(label, defense, adapter, clean, injected, ids_seen)

    print("\n\n=== LONG-FORM SUMMARY: mean absolute scores per family ===")
    print(f"{'Family':<8}{'D0 inj':<10}{'D1 inj':<10}{'D2 inj':<10}")
    for fam in FAMILIES:
        d0i = all_results["D0"][fam]["injected"]
        d1i = all_results["D1"][fam]["injected"]
        d2i = all_results["D2"][fam]["injected"]
        d0m = round(statistics.mean(d0i), 2) if d0i else None
        d1m = round(statistics.mean(d1i), 2) if d1i else None
        d2m = round(statistics.mean(d2i), 2) if d2i else None
        print(f"{fam:<8}{str(d0m):<10}{str(d1m):<10}{str(d2m):<10}")

    out_path = EVAL_DIR / "longform_validation_summary.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
