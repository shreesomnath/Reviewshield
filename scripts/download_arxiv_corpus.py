"""Pull the seed corpus of open-access CS/ML papers from arXiv.

Downloads PDFs for a target count of recent cs.LG / cs.CL / cs.AI papers,
extracts text with pymupdf, and writes one JSON record per paper with the
fields the rest of the pipeline needs (title, abstract, sections, raw PDF
path). No API key required: arXiv's API and PDF mirrors are open.

Usage (inside the container):
    python /workspace/scripts/download_arxiv_corpus.py --n 800
"""
import argparse
import json
import time
from pathlib import Path

import arxiv
import fitz  # pymupdf
from loguru import logger

RAW_DIR = Path("/workspace/data/raw/arxiv_pdfs")
OUT_PATH = Path("/workspace/data/interim/arxiv_corpus.jsonl")
CATEGORIES = ["cs.LG", "cs.CL", "cs.AI", "cs.CV", "stat.ML"]


def extract_sections(pdf_path: Path) -> dict:
    """Pull raw text and a best-effort split into title/abstract/body."""
    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    n_pages = len(doc)
    doc.close()
    return {"full_text": full_text, "n_pages": n_pages}


def main(n_target: int, out_path: Path, raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)
    query = " OR ".join(f"cat:{c}" for c in CATEGORIES)
    search = arxiv.Search(
        query=query,
        max_results=n_target,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    n_written = 0
    with open(out_path, "w") as out_f:
        for result in client.results(search):
            arxiv_id = result.get_short_id()
            pdf_path = raw_dir / f"{arxiv_id.replace('/', '_')}.pdf"
            try:
                if not pdf_path.exists():
                    result.download_pdf(dirpath=str(raw_dir), filename=pdf_path.name)
                    time.sleep(1)  # be polite to arXiv's mirror
                parsed = extract_sections(pdf_path)
            except Exception as e:
                logger.warning(f"skip {arxiv_id}: {e}")
                continue

            record = {
                "arxiv_id": arxiv_id,
                "title": result.title.strip(),
                "abstract": result.summary.strip().replace("\n", " "),
                "authors": [a.name for a in result.authors],
                "categories": result.categories,
                "published": result.published.isoformat(),
                "pdf_path": str(pdf_path),
                "full_text": parsed["full_text"],
                "n_pages": parsed["n_pages"],
            }
            out_f.write(json.dumps(record) + "\n")
            n_written += 1
            if n_written % 25 == 0:
                logger.info(f"{n_written}/{n_target} papers pulled")

    logger.info(f"Done. Wrote {n_written} papers to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800, help="target number of papers")
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    ap.add_argument("--raw-dir", type=str, default=str(RAW_DIR))
    args = ap.parse_args()
    main(args.n, Path(args.out), Path(args.raw_dir))
