"""Real content-quality/style statistics for D0 vs D2's clean-paper
reviews, from the stored text samples (progress.jsonl files) - not just
scores. Answers: how long are the reviews, and what critique themes
recur, per condition.

Usage (inside the container):
    python /workspace/scripts/analysis/review_content_stats.py
"""
import json
import re
import statistics
from pathlib import Path

EVAL_DIR = Path("/workspace/outputs/eval")

THEME_KEYWORDS = {
    "empirical/experimental rigor": ["empirical", "experiment", "ablation", "validation"],
    "generalizability": ["generaliz", "broader applicability"],
    "practical relevance": ["practical"],
    "scalability/efficiency": ["scalab", "efficien", "computational complexity"],
    "comparison to baselines": ["baseline", "comparison", "state-of-the-art", "existing method"],
}


def load(cond):
    path = EVAL_DIR / f"{cond}_Qwen_Qwen2.5-14B-Instruct_progress.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            out.append(rec["result"].get("clean_review_text_sample", ""))
    return out


def main():
    for cond in ["D0", "D1", "D2"]:
        texts = load(cond)
        if not texts:
            print(f"{cond}: no data yet")
            continue
        lengths = [len(t.split()) for t in texts if t]
        print(f"\n=== {cond} (n={len(texts)} reviews) ===")
        print(f"mean word count: {round(statistics.mean(lengths), 1)}  "
              f"stdev: {round(statistics.stdev(lengths), 1) if len(lengths) > 1 else 0}  "
              f"min: {min(lengths)}  max: {max(lengths)}")
        for theme, kws in THEME_KEYWORDS.items():
            hits = sum(1 for t in texts if any(kw in t.lower() for kw in kws))
            print(f"  mentions '{theme}': {hits}/{len(texts)} ({100*hits/len(texts):.0f}%)")


if __name__ == "__main__":
    main()
