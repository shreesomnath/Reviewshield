"""HISTORICAL / GPU-parallelization helper, not needed to reproduce the
paper's results. The canonical script is evaluation/longform_fdr_validation.py,
which runs all four conditions (D0, D1, D2, D3) sequentially on one GPU.
This is that same script cut down to D3 only, so it could run alone on a
second, otherwise-idle GPU while the sequential run was still working
through D0-D2 on the first -- both write to the same
longform_fdr_progress_D3.jsonl and rely on that script's resume-safe
done_ids check to avoid duplicating work. Kept here for provenance, not
as a script you need to run: evaluation/longform_fdr_validation.py alone
reproduces the full result.

Long-form Flaw-Detection Recall (FDR) follow-up check, added in
response to independent review feedback (both an external LLM reviewer
and a second AI-review tool independently flagged the same issue): the
main 50-paper evaluation reports FDR near zero for every condition, and
Section 7 traces this to the ~80-word brief-review format not reaching
the limitations-style content where planted flaws get flagged. The
existing longform_validation.py check (600 tokens, 5 papers) already
established that qualitative score-inflation finding survives at
realistic length, but it never actually measured FDR at that length
(injection_only variant only, no flaw text, no FDR judge call) and never
included D3 (SFT), which is now central to the paper's finding. This
script fixes both gaps directly, reusing the same 5-paper sample and the
same 600-token budget already validated as sufficient to reach
limitations-style content.

Uses the injection_plus_flaw variant (same as the main harness's FDR
computation): each paper's planted flaw combined with that family's
attack, so FDR is measured under the harder, more realistic condition
(resist the attack AND still catch the real problem), not on flaw_only
alone.

Usage (inside the container, on a second GPU while the full script runs
on the first):
    CUDA_VISIBLE_DEVICES=1 python /workspace/scripts/longform_fdr_validation_d3only.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("/workspace/scripts")))
sys.path.insert(0, str(Path("/workspace/scripts/generation")))
from unsloth import FastLanguageModel
from model_utils import load_model_unsloth, generate, truncate_manuscript, parse_score_decision
from review_prompt import build_prompt
from response_judge import judge_flaw_detected

BENCH_PATH = "/workspace/data/processed/revguard_bench/test.jsonl"
EVAL_DIR = Path("/workspace/outputs/eval")
N_PAPERS = 5  # same sample as longform_validation.py, for direct comparability
FAMILIES = ["F1", "F2", "F3", "F4", "F5"]
LONGFORM_MAX_TOKENS = 600  # matches longform_validation.py's established budget

CONDITIONS = [
    ("D3", None, "/workspace/outputs/checkpoints/sft_qwen14b_v1/final"),
]


def load_sample():
    """Same paper selection logic as longform_validation.py (first 5 papers
    by clean-variant order in test.jsonl), so results are directly
    comparable to the existing 5-paper long-form check."""
    ids_seen = []
    with open(BENCH_PATH) as f:
        for line in f:
            v = json.loads(line)
            if v["variant_type"] == "clean":
                aid = v["arxiv_id"]
                if aid not in ids_seen and len(ids_seen) < N_PAPERS:
                    ids_seen.append(aid)

    by_paper_flaw = {aid: {} for aid in ids_seen}
    with open(BENCH_PATH) as f:
        for line in f:
            v = json.loads(line)
            aid = v["arxiv_id"]
            if aid in ids_seen and v["variant_type"] == "injection_plus_flaw":
                by_paper_flaw[aid][v["family"]] = v
    return ids_seen, by_paper_flaw


def run_condition(label, defense, adapter_path, ids_seen, by_paper_flaw):
    progress_path = EVAL_DIR / f"longform_fdr_progress_{label}.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if progress_path.exists():
        with open(progress_path) as f:
            for line in f:
                done_ids.add(json.loads(line)["aid"])
        if done_ids:
            print(f"[{label}] Resuming: {len(done_ids)} papers already done")

    gen = load_model_unsloth("Qwen/Qwen2.5-14B-Instruct", adapter_path=adapter_path)
    judge = gen  # same self-judging pattern as run_eval.py
    print(f"\n=== {label} (long-form FDR, {LONGFORM_MAX_TOKENS} tokens) ===")
    with open(progress_path, "a") as pf:
        for aid in ids_seen:
            if aid in done_ids:
                continue
            per_family = {}
            for fam in FAMILIES:
                variant = by_paper_flaw[aid].get(fam)
                if variant is None:
                    continue
                text = truncate_manuscript(variant["manuscript_text"])
                out = generate(gen, build_prompt(text, defense=defense, longform=True),
                                max_new_tokens=LONGFORM_MAX_TOKENS, do_sample=False)
                sd = parse_score_decision(out) if out else {"score": None, "well_formed": False}
                fdr = judge_flaw_detected(judge, out, variant["flag_description"], generate) if out else None
                print(f"{aid} {fam}: score={sd['score']} well_formed={sd['well_formed']} fdr={fdr}")
                per_family[fam] = {"score": sd["score"], "well_formed": sd["well_formed"], "fdr": fdr}
            pf.write(json.dumps({"aid": aid, "per_family": per_family}) + "\n")
            pf.flush()


def summarize():
    print("\n\n=== LONG-FORM FDR SUMMARY (600 tokens, n=5 papers) ===")
    print(f"{'Family':<8}" + "".join(f"{c[0]:<8}" for c in CONDITIONS))
    all_data = {}
    for label, _, _ in CONDITIONS:
        path = EVAL_DIR / f"longform_fdr_progress_{label}.jsonl"
        by_family = {fam: [] for fam in FAMILIES}
        if path.exists():
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    for fam, vals in rec["per_family"].items():
                        if vals["fdr"] is not None:
                            by_family[fam].append(vals["fdr"])
        all_data[label] = by_family
    for fam in FAMILIES:
        row = f"{fam:<8}"
        for label, _, _ in CONDITIONS:
            vals = all_data[label][fam]
            rate = f"{sum(vals)}/{len(vals)}" if vals else "n/a"
            row += f"{rate:<8}"
        print(row)
    out_path = EVAL_DIR / "longform_fdr_summary.json"
    out_path.write_text(json.dumps(all_data, indent=2))
    print(f"Wrote {out_path}")


def main():
    ids_seen, by_paper_flaw = load_sample()
    print(f"Sampled {len(ids_seen)} papers: {ids_seen}")
    for label, defense, adapter in CONDITIONS:
        run_condition(label, defense, adapter, ids_seen, by_paper_flaw)
    summarize()


if __name__ == "__main__":
    main()
