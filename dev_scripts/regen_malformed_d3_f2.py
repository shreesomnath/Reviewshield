"""Targeted regeneration of the 4 remaining D3+F2 malformed cases (2 of
the original 6 were already recovered as a side effect of the earlier
content-capture run). Same method as content_capture.py: 300-token
budget, greedy decoding, same prompt construction -- but targets these
specific 4 arxiv_ids directly instead of the fixed first-10 sample.

Usage (inside the container):
    python /workspace/scripts/evaluation/regen_malformed_d3_f2.py
"""
import json
import sys
from pathlib import Path

from unsloth import FastLanguageModel
sys.path.insert(0, str(Path("/workspace/scripts")))
sys.path.insert(0, str(Path("/workspace/scripts/generation")))
from model_utils import load_model_unsloth, generate, truncate_manuscript, parse_score_decision
from review_prompt import build_prompt

BENCH_PATH = "/workspace/data/processed/revguard_bench/test.jsonl"
OUT_PATH = Path("/workspace/outputs/eval/regen_malformed_d3_f2.jsonl")
TARGET_IDS = ["2608.05144v1", "2608.06009v1", "2608.05673v1", "2608.05893v1"]
MAX_NEW_TOKENS = 300  # same budget as the original content-capture run


def load_targets():
    out = {}
    with open(BENCH_PATH) as f:
        for line in f:
            v = json.loads(line)
            if v["arxiv_id"] in TARGET_IDS and v["variant_type"] == "injection_only" and v["family"] == "F2":
                out[v["arxiv_id"]] = v
    return out


def main():
    targets = load_targets()
    print(f"Found {len(targets)} of {len(TARGET_IDS)} target papers in test.jsonl: {list(targets.keys())}")

    gen = load_model_unsloth("Qwen/Qwen2.5-14B-Instruct",
                              adapter_path="/workspace/outputs/checkpoints/sft_qwen14b_v1/final")

    with open(OUT_PATH, "w") as out_f:
        for aid, variant in targets.items():
            text = truncate_manuscript(variant["manuscript_text"])
            prompt = build_prompt(text, defense=None, brief=True)
            out = generate(gen, prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            sd = parse_score_decision(out) if out else {"score": None, "decision": None, "well_formed": False}
            print(f"{aid} F2: score={sd['score']} decision={sd['decision']} well_formed={sd['well_formed']}")
            print(f"  text: {out!r}")
            rec = {"arxiv_id": aid, "family": "F2", "text": out, "score": sd["score"],
                   "decision": sd["decision"], "well_formed": sd["well_formed"],
                   "max_new_tokens": MAX_NEW_TOKENS}
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
