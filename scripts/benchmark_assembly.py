"""Assemble RevGuard-Bench from the arXiv corpus + attack families + flaws.

Two splits matter (plan.md sec:3, sec:6):
  1. Paper-level train/val/test - disjoint papers in each split so no
     preference-optimization run has seen a test paper's content, controlling
     for memorization (plan.md sec:8, "Contamination / memorization").
  2. Attack-family seen/held-out - F1, F2, F3 are used to build DPO/SFT
     training data; F4 (authority) and F5 (sycophancy) are held out entirely
     from training and used only at evaluation time, to test whether the
     defense generalizes to attack styles it never saw rather than
     memorizing surface patterns (RQ3).

For each paper x split we build four manuscript variants, matching plan.md
sec:4: clean; clean+injection; clean+flaw; injection+flaw. The clean+flaw
and injection+flaw variants use only the deterministic flaw types for now
(see flaws.py); the LLM-assisted "deleted baseline" flaw is added at the
generation stage once available.

Usage (inside the container):
    python /workspace/scripts/benchmark_assembly.py --n-papers 800
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from attacks.family_f1_direct_override import generate as gen_f1
from attacks.family_f2_stealth import generate as gen_f2
from attacks.family_f3_reviewer_impersonation import generate as gen_f3
from attacks.family_f4_authority import generate as gen_f4
from attacks.family_f5_sycophancy import generate as gen_f5
from flaws import generate as gen_flaws
from inject import inject

CORPUS_PATH = Path("/workspace/data/interim/arxiv_corpus.jsonl")
OUT_DIR = Path("/workspace/data/processed/revguard_bench")

TRAIN_FAMILIES = {"F1": gen_f1, "F2": gen_f2, "F3": gen_f3}
HELDOUT_FAMILIES = {"F4": gen_f4, "F5": gen_f5}
ALL_FAMILIES = {**TRAIN_FAMILIES, **HELDOUT_FAMILIES}

# Paper-level split fractions
SPLIT_FRACS = {"train": 0.70, "val": 0.10, "test": 0.20}


def load_corpus(path: Path) -> list[dict]:
    """Load the corpus and drop any paper that failed audit_corpus.py's
    completeness check (missing intro/references, or a suspiciously
    short/long extraction - see audit_corpus.py). Requires the audit to
    have already been run (`<corpus>_audit.json` alongside the corpus
    file); fails loudly rather than silently including unaudited or
    known-bad papers, since a manuscript with a truncated or garbled
    extraction would corrupt both the attack-injection and the
    flaw-detection utility signal for that paper."""
    papers = []
    with open(path) as f:
        for line in f:
            papers.append(json.loads(line))

    audit_path = path.parent / f"{path.stem}_audit.json"
    if not audit_path.exists():
        raise FileNotFoundError(
            f"{audit_path} not found - run audit_corpus.py on {path} before "
            f"assembling the benchmark, so known-incomplete papers are excluded."
        )
    audit = json.loads(audit_path.read_text())
    failed_ids = {r["arxiv_id"] for r in audit["per_paper"] if not r["passed"]}
    kept = [p for p in papers if p["arxiv_id"] not in failed_ids]
    print(f"Loaded {len(papers)} papers, dropped {len(failed_ids)} that failed "
          f"the content-completeness audit, kept {len(kept)}.")
    return kept


def paper_split(papers: list[dict], rng: random.Random) -> dict:
    ids = [p["arxiv_id"] for p in papers]
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * SPLIT_FRACS["train"])
    n_val = int(n * SPLIT_FRACS["val"])
    return {
        "train": set(ids[:n_train]),
        "val": set(ids[n_train:n_train + n_val]),
        "test": set(ids[n_train + n_val:]),
    }


def build_variants(paper: dict, split: str, rng: random.Random) -> list[dict]:
    """Return the manuscript variants for one paper: clean, +injection,
    +flaw, +injection+flaw. Train/val only use seen families (F1-F3); test
    uses all five so held-out-family generalization can be measured."""
    families = ALL_FAMILIES if split == "test" else TRAIN_FAMILIES
    variants = [{
        "arxiv_id": paper["arxiv_id"],
        "split": split,
        "variant_type": "clean",
        "manuscript_text": paper["full_text"],
        "family": None,
        "attack_variant": None,
        "target_behavior": None,
        "flaw_type": None,
        "flag_description": None,
    }]

    flaw = gen_flaws(paper, rng, k=1)[0]
    flawed_text = inject(paper["full_text"], flaw["flawed_text"], flaw["insertion_point"])
    variants.append({
        "arxiv_id": paper["arxiv_id"], "split": split, "variant_type": "flaw_only",
        "manuscript_text": flawed_text, "family": None, "attack_variant": None,
        "target_behavior": None, "flaw_type": flaw["flaw_type"],
        "flag_description": flaw["flag_description"],
    })

    for fam_name, gen_fn in families.items():
        for atk in gen_fn(paper, rng, k=1):
            injected_text = inject(paper["full_text"], atk["injected_text"], atk["insertion_point"])
            variants.append({
                "arxiv_id": paper["arxiv_id"], "split": split, "variant_type": "injection_only",
                "manuscript_text": injected_text, "family": fam_name,
                "attack_variant": atk["variant"], "target_behavior": atk["target_behavior"],
                "flaw_type": None, "flag_description": None,
            })
            # injection + flaw combined variant (harder case: must resist the
            # attack AND still catch the real problem)
            combined_text = inject(flawed_text, atk["injected_text"], atk["insertion_point"])
            variants.append({
                "arxiv_id": paper["arxiv_id"], "split": split, "variant_type": "injection_plus_flaw",
                "manuscript_text": combined_text, "family": fam_name,
                "attack_variant": atk["variant"], "target_behavior": atk["target_behavior"],
                "flaw_type": flaw["flaw_type"], "flag_description": flaw["flag_description"],
            })
    return variants


def main(n_papers: int, seed: int):
    rng = random.Random(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    papers = load_corpus(CORPUS_PATH)[:n_papers]
    if not papers:
        print(f"No papers in {CORPUS_PATH} — run download_arxiv_corpus.py first.")
        return

    splits = paper_split(papers, rng)
    by_id = {p["arxiv_id"]: p for p in papers}

    counts = {"train": 0, "val": 0, "test": 0}
    for split_name, id_set in splits.items():
        out_path = OUT_DIR / f"{split_name}.jsonl"
        with open(out_path, "w") as out_f:
            for aid in id_set:
                for v in build_variants(by_id[aid], split_name, rng):
                    out_f.write(json.dumps(v) + "\n")
                    counts[split_name] += 1
        print(f"{split_name}: {len(id_set)} papers -> {counts[split_name]} manuscript variants -> {out_path}")

    manifest = {
        "n_papers": len(papers),
        "papers_per_split": {k: len(v) for k, v in splits.items()},
        "variants_per_split": counts,
        "train_families": list(TRAIN_FAMILIES.keys()),
        "heldout_families": list(HELDOUT_FAMILIES.keys()),
        "seed": seed,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest to {OUT_DIR / 'manifest.json'}")
    print(f"Train/val use seen families only {list(TRAIN_FAMILIES)}; "
          f"test additionally includes held-out families {list(HELDOUT_FAMILIES)}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-papers", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args.n_papers, args.seed)
