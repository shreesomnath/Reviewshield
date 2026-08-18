"""D1-only absolute-score pass over the same 50-paper x 5-family sample
used in absolute_score_sample.py (which covers D0 and D2) - kept separate
so this can run in parallel on the other GPU without recomputing D0,
which is already being generated there.

Usage (inside the container):
    python /workspace/scripts/absolute_score_sample_d1.py
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


def main():
    clean, injected, ids_seen = load_sample()
    print(f"Sampled {len(ids_seen)} papers: {ids_seen}")

    # Incremental, resume-safe checkpointing - same pattern as run_eval.py
    # and absolute_score_sample.py, applied here too after losing ~23
    # papers of real progress to the server crash the first time this
    # script ran without it.
    progress_path = Path("/workspace/outputs/eval/abs_score_progress_D1.jsonl")
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
            print(f"Resuming: {len(done_ids)} papers already done in {progress_path}")

    gen = load_model_unsloth("Qwen/Qwen2.5-14B-Instruct")
    with open(progress_path, "a") as pf:
        for aid in ids_seen:
            if aid in done_ids:
                continue
            c_out = generate(gen, build_prompt(truncate_manuscript(clean[aid]), defense="D1", brief=True),
                              max_new_tokens=150, do_sample=False)
            c_sd = parse_score_decision(c_out)
            per_family = {}
            for fam in FAMILIES:
                if fam not in injected.get(aid, {}):
                    continue
                i_out = generate(gen, build_prompt(truncate_manuscript(injected[aid][fam]), defense="D1", brief=True),
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

    print("\n\n=== D1 SUMMARY: mean absolute scores per family ===")
    print(f"{'Family':<8}{'D1 clean':<10}{'D1 inj':<10}")
    for fam in FAMILIES:
        c = results[fam]["clean"]
        i = results[fam]["injected"]
        c_m = round(statistics.mean(c), 2) if c else None
        i_m = round(statistics.mean(i), 2) if i else None
        print(f"{fam:<8}{str(c_m):<10}{str(i_m):<10}")

    out_path = Path("/workspace/outputs/eval/absolute_score_sample_d1.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
