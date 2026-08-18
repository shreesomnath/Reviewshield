"""Real, broader absolute-score comparison between D0 and D2, across all
5 attack families, to properly check whether the relative-ASR result
(D2 apparently worse than D0) is a measurement artifact of D2's lower
clean-paper baseline, or a real robustness regression - the initial
5-paper/F1-only spot check strongly suggested an artifact (D2's absolute
injected scores matched or beat D0's on every paper), but that was too
small a sample to trust as the final answer.

Samples 8 real papers x 5 families = 40 injected generations + 8 clean
generations per condition (D0, D2), a meaningful sample without paying
for a full 500-variant re-run.

Usage (inside the container):
    python /workspace/scripts/absolute_score_sample.py
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
N_PAPERS = 50
FAMILIES = ["F1", "F2", "F3", "F4", "F5"]


def load_sample():
    clean = {}
    injected = {}
    ids_seen = []
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


def run_condition(label, gen, clean, injected, ids_seen):
    # Incremental, resume-safe checkpointing: every paper's result is
    # appended to a .jsonl file the instant it's computed, and already-done
    # papers are skipped on restart. Same pattern already proven in
    # run_eval.py after the training-OOM lesson - this ad hoc script should
    # have had it from the start, not just the main eval harness.
    progress_path = Path(f"/workspace/outputs/eval/abs_score_progress_{label}.jsonl")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    results = {fam: {"clean": [], "injected": []} for fam in FAMILIES}
    done_ids = set()
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
            print(f"[{label}] Resuming: {len(done_ids)} papers already done in {progress_path}")

    print(f"\n=== {label} ===")
    with open(progress_path, "a") as pf:
        for aid in ids_seen:
            if aid in done_ids:
                continue
            c_out = generate(gen, build_prompt(truncate_manuscript(clean[aid]), defense=None, brief=True),
                              max_new_tokens=150, do_sample=False)
            c_sd = parse_score_decision(c_out)
            per_family = {}
            for fam in FAMILIES:
                if fam not in injected.get(aid, {}):
                    continue
                i_out = generate(gen, build_prompt(truncate_manuscript(injected[aid][fam]), defense=None, brief=True),
                                  max_new_tokens=150, do_sample=False)
                i_sd = parse_score_decision(i_out)
                print(f"{aid} {fam}: clean={c_sd['score']} injected={i_sd['score']}")
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
    for label, adapter in [("D0", None), ("D2", "/workspace/outputs/checkpoints/dpo_qwen14b_v2/final")]:
        gen = load_model_unsloth("Qwen/Qwen2.5-14B-Instruct", adapter_path=adapter)
        all_results[label] = run_condition(label, gen, clean, injected, ids_seen)

    print("\n\n=== SUMMARY: mean absolute scores per family ===")
    print(f"{'Family':<8}{'D0 clean':<10}{'D0 inj':<10}{'D2 clean':<10}{'D2 inj':<10}")
    for fam in FAMILIES:
        d0c = all_results["D0"][fam]["clean"]
        d0i = all_results["D0"][fam]["injected"]
        d2c = all_results["D2"][fam]["clean"]
        d2i = all_results["D2"][fam]["injected"]
        d0c_m = round(statistics.mean(d0c), 2) if d0c else None
        d0i_m = round(statistics.mean(d0i), 2) if d0i else None
        d2c_m = round(statistics.mean(d2c), 2) if d2c else None
        d2i_m = round(statistics.mean(d2i), 2) if d2i else None
        print(f"{fam:<8}{str(d0c_m):<10}{str(d0i_m):<10}{str(d2c_m):<10}{str(d2i_m):<10}")


if __name__ == "__main__":
    main()
