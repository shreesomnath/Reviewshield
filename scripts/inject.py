"""Insert an attack's injected_text into a paper's extracted full_text at a
heuristic insertion point.

PDF text extraction rarely gives clean section boundaries, so insertion
points are located with simple keyword heuristics. This is intentionally
approximate for now; sufficient to validate the parse -> inject -> generate
loop end to end (plan.md sec:9, immediate next step 3). Can be swapped for a
layout-aware splitter later without touching callers.

Every computed position is snapped to the nearest paragraph break (a blank
line) before splicing, so the injected text always starts its own paragraph
rather than landing mid-sentence - confirmed necessary by inspecting actual
smoke-test output, where the raw character-offset approach produced
mid-sentence insertions.
"""
import re

SECTION_KEYWORDS = {
    "introduction": [r"\n\s*1\.?\s+Introduction", r"\n\s*Introduction\s*\n"],
    "conclusion": [r"\n\s*\d+\.?\s+Conclusion", r"\n\s*Conclusion(s)?\s*\n",
                   r"\n\s*\d+\.?\s+Discussion"],
}

_PARA_BREAK = re.compile(r"\n\s*\n")


def _find_first(text: str, patterns: list[str]) -> int | None:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.start()
    return None


def _snap_to_paragraph_break(text: str, pos: int) -> int:
    """Move pos to the nearest paragraph boundary (blank line) so the
    injected block never lands mid-sentence. Searches forward first (keeps
    insertions from drifting earlier than intended, e.g. before conclusion);
    falls back to the nearest break in either direction if none is found
    within a reasonable window."""
    window = 800
    forward = _PARA_BREAK.search(text, pos, min(len(text), pos + window))
    if forward:
        return forward.end()
    backward_matches = list(_PARA_BREAK.finditer(text, max(0, pos - window), pos))
    if backward_matches:
        return backward_matches[-1].end()
    return pos  # no paragraph break nearby (short/oddly-formatted text); leave as-is


def inject(full_text: str, injected_text: str, insertion_point: str) -> str:
    """Return full_text with injected_text spliced in at insertion_point."""
    n = len(full_text)

    if insertion_point == "end_of_abstract":
        idx = _find_first(full_text, SECTION_KEYWORDS["introduction"])
        pos = idx if idx is not None else min(1800, n)
    elif insertion_point == "end_of_introduction":
        idx = _find_first(full_text, SECTION_KEYWORDS["introduction"])
        if idx is not None:
            # insert ~2000 chars after the Introduction heading, or at the
            # next section heading if it comes sooner
            tail_search = full_text[idx + 20:]
            next_idx = _find_first(tail_search, SECTION_KEYWORDS["conclusion"])
            pos = idx + 20 + (min(2000, next_idx) if next_idx else 2000)
        else:
            pos = min(4000, n)
    elif insertion_point == "before_conclusion":
        idx = _find_first(full_text, SECTION_KEYWORDS["conclusion"])
        pos = idx if idx is not None else max(0, n - 1500)
    else:
        raise ValueError(f"unknown insertion_point {insertion_point!r}")

    pos = max(0, min(pos, n))
    pos = _snap_to_paragraph_break(full_text, pos)
    return full_text[:pos] + f"\n\n{injected_text}\n\n" + full_text[pos:]
