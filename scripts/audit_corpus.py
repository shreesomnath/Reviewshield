"""Content-quality audit for the pulled arXiv corpus, run after every
download_arxiv_corpus.py call and before benchmark_assembly.py consumes it.

Checks completeness (not just "did the script run without error"): real
abstract content, real introduction content, and a real references/
bibliography section present ANYWHERE in the text (not just a narrow tail
window - modern ML papers often have long appendices/checklists after the
bibliography, which a naive tail-only search misses and falsely flags as
"no references"). Also flags papers that are suspiciously short (likely a
failed/partial PDF extraction) or suspiciously long (likely an extraction
artifact duplicating content).

Usage (inside the container):
    python /workspace/scripts/audit_corpus.py --in /workspace/data/interim/arxiv_corpus.jsonl
"""
import argparse
import json
import re
from pathlib import Path

ABSTRACT_RE = re.compile(r"\babstract\b", re.IGNORECASE)
INTRO_RE = re.compile(r"\bintroduction\b", re.IGNORECASE)
REFS_RE = re.compile(r"\breferences\b", re.IGNORECASE)

MIN_CHARS = 8000    # shorter than this suggests a truncated/failed extraction
MAX_CHARS = 400000  # longer than this suggests a duplication artifact


def audit_one(rec: dict) -> dict:
    t = rec["full_text"]
    # Abstract can appear unlabeled (no literal "Abstract" heading in some
    # templates), so its absence alone is not disqualifying - flagged as
    # informational, not a failure.
    has_abstract_label = bool(ABSTRACT_RE.search(t[:3000]))
    has_intro = bool(INTRO_RE.search(t[:12000]))
    has_refs_anywhere = bool(REFS_RE.search(t))
    length_ok = MIN_CHARS <= len(t) <= MAX_CHARS
    # The real completeness bar: intro present, references present
    # somewhere, and length in a sane range. Abstract-label presence is
    # reported but not required.
    passed = has_intro and has_refs_anywhere and length_ok
    return {
        "arxiv_id": rec["arxiv_id"],
        "chars": len(t),
        "n_pages": rec.get("n_pages"),
        "has_abstract_label": has_abstract_label,
        "has_intro": has_intro,
        "has_refs_anywhere": has_refs_anywhere,
        "length_ok": length_ok,
        "passed": passed,
    }


def main(in_path: Path):
    recs = [json.loads(l) for l in open(in_path)]
    results = [audit_one(r) for r in recs]

    n = len(results)
    n_passed = sum(r["passed"] for r in results)
    n_no_abstract_label = sum(not r["has_abstract_label"] for r in results)
    n_no_intro = sum(not r["has_intro"] for r in results)
    n_no_refs = sum(not r["has_refs_anywhere"] for r in results)
    n_bad_length = sum(not r["length_ok"] for r in results)

    print(f"Audited {n} papers from {in_path}")
    print(f"  passed (intro + refs + sane length): {n_passed}/{n} ({100*n_passed/n:.1f}%)")
    print(f"  no literal 'Abstract' label (informational, not a failure): {n_no_abstract_label}/{n}")
    print(f"  no 'Introduction' found in first 12000 chars: {n_no_intro}/{n}")
    print(f"  no 'References' found anywhere: {n_no_refs}/{n}")
    print(f"  length outside [{MIN_CHARS}, {MAX_CHARS}] chars: {n_bad_length}/{n}")

    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n{len(failures)} papers failed the audit and should be dropped before benchmark assembly:")
        for r in failures:
            print(f"  {r['arxiv_id']}: intro={r['has_intro']} refs={r['has_refs_anywhere']} chars={r['chars']}")

    out_path = in_path.parent / f"{in_path.stem}_audit.json"
    out_path.write_text(json.dumps({"summary": {
        "n": n, "n_passed": n_passed, "n_no_intro": n_no_intro,
        "n_no_refs": n_no_refs, "n_bad_length": n_bad_length,
    }, "per_paper": results}, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    args = ap.parse_args()
    main(Path(args.in_path))
