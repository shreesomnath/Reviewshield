"""End-to-end smoke test of the parse -> inject loop on a handful of papers,
before scaling to the full corpus (plan.md sec:9, step 3).

Reads the arXiv corpus JSONL, applies F1 (direct-override) attacks to the
first --n papers, and writes the injected manuscripts alongside a manifest
so we can eyeball insertion quality before building the remaining attack
families or spending compute on preference-pair generation.

Usage (inside the container):
    python /workspace/scripts/smoke_test_loop.py --n 5
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from attacks.family_f1_direct_override import generate as gen_f1
from inject import inject

CORPUS_PATH = Path("/workspace/data/interim/arxiv_corpus.jsonl")
OUT_DIR = Path("/workspace/data/interim/smoke_test")


def main(n: int, seed: int):
    rng = random.Random(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    papers = []
    with open(CORPUS_PATH) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            papers.append(json.loads(line))

    if not papers:
        print(f"No papers found in {CORPUS_PATH} — run download_arxiv_corpus.py first.")
        return

    manifest = []
    for paper in papers:
        attacks = gen_f1(paper, rng, k=1)
        for atk in attacks:
            injected = inject(paper["full_text"], atk["injected_text"], atk["insertion_point"])
            out_path = OUT_DIR / f"{paper['arxiv_id'].replace('/', '_')}_{atk['variant']}.txt"
            out_path.write_text(injected)
            manifest.append({
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "family": atk["family"],
                "variant": atk["variant"],
                "insertion_point": atk["insertion_point"],
                "target_behavior": atk["target_behavior"],
                "injected_manuscript_path": str(out_path),
                "clean_len": len(paper["full_text"]),
                "injected_len": len(injected),
            })
            print(f"[OK] {paper['arxiv_id']}  {atk['variant']}  @ {atk['insertion_point']}  -> {out_path.name}")

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(manifest)} injected manuscripts + manifest to {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args.n, args.seed)
