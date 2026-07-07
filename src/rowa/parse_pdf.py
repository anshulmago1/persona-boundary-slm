"""Turn cached AP Central PDFs into labeled Row A thesis records.

Two tiers of gold, both officially labeled:

- **Tier 1 (rubric examples):** the Row A scoring-guideline page lists example
  theses under "Examples that earn / do not earn this point" (with sub-labels like
  "restatement" / "minimally acceptable"). Typed, two-column; separated by x-position.
- **Tier 2 (student samples):** the typed scoring commentary gives, per sample
  (2A/2B/2C), the official ``Thesis Score`` and usually **quotes the thesis verbatim**.
  When it doesn't (~10%), we fall back to gpt-4o **vision** on the handwritten scan.

Output (``data/rowa/gold_scraped_raw.jsonl``, gitignored): one record per thesis with
``{source_pdf, year, set, qtype, sample_id, prompt, thesis, label, official_reason,
tier, needs_vision}``.

    python -m src.rowa.parse_pdf                 # typed pass over all cached PDFs
    python -m src.rowa.parse_pdf --vision        # + gpt-4o vision for missing theses
    python -m src.rowa.parse_pdf --vision --dry-run   # wiring only, no API
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import pdfplumber

PDF_DIR = Path("data/rowa/pdfs")
OUT_RAW = Path("data/rowa/gold_scraped_raw.jsonl")
IMG_DIR = Path("data/rowa/page_images")

_QUOTE = re.compile(r"[“\"]([^“”\"]{12,}?)[”\"]", re.S)
_SAMPLE_SPLIT = re.compile(r"\bSample:\s*([0-9]?[A-C])\b")
_THESIS_DEC = re.compile(
    r"Thesis/Claim\s*\(0[–-]1\s*points?\):\s*([01])(.*?)"
    r"(?=(?:[A-E]\.\s*)?Contextualization\s*\(0[–-]1|\Z)",
    re.S,
)
# Summary line: "Thesis Score: 1" (2024/25) or "Thesis/Claim: 1" (2023). The decision
# line "Thesis/Claim (0–1 points): 1" is excluded by requiring ":" right after the word.
_THESIS_SCORE = re.compile(r"(?:Thesis Score|Thesis/Claim):\s*([01])\b")
_PROSE_EARN = re.compile(r"earned (?:1 point|the point) for thesis", re.I)
_PROSE_DENY = re.compile(r"did not earn (?:the point|1 point|the thesis)", re.I)
_IMG_PAGE = re.compile(r"^\s*\d+\s*of\s*\d+\s*([0-9]?[A-C])\s*$")
_FOOTER = re.compile(r"^(©|Visit College Board|AP Central|AP® World History)", re.I)


@dataclass
class ThesisRecord:
    source_pdf: str
    year: int
    set_no: int
    qtype: str
    sample_id: Optional[str]  # e.g. "2A" for tier-2; None for tier-1 examples
    prompt: str
    thesis: Optional[str]
    label: int
    official_reason: str
    tier: int  # 1 = rubric example, 2 = student sample
    subcat: Optional[str] = None  # tier-1 sub-heading, e.g. "restatement"
    needs_vision: bool = False
    contradictory: bool = False  # tier-2: decision digit disagrees with the prose


# --- shared helpers ---------------------------------------------------------


def _clean(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not _FOOTER.match(ln.strip())]
    joined = " ".join(lines).replace("•", " ")  # drop stray column bullets
    return re.sub(r"\s+", " ", joined).strip()


def _meta_from_name(name: str):
    m = re.match(r"ap(\d\d)-.*?-(dbq|leq\d)-set-(\d)", name)
    year = 2000 + int(m.group(1))
    return year, m.group(2), int(m.group(3))


def _full_text(pdf) -> str:
    return "\n".join((p.extract_text() or "") for p in pdf.pages)


def extract_prompt(full: str) -> str:
    """The '[context]. Develop an argument ... during this period.' prompt."""
    dev = re.search(r"Develop an argument.*?\.", full, re.S)
    if not dev:
        return ""
    pre = re.search(r"(In the period[^.]*\.)\s*Develop an argument", full, re.S)
    parts = [pre.group(1)] if pre else []
    parts.append(dev.group(0))
    return _clean(" ".join(parts))


# --- Tier 2: student samples via scoring commentary -------------------------


def parse_commentary(full: str) -> List[dict]:
    """One dict per sample: {sample_id, label, thesis, reason, needs_vision}."""
    out = []
    parts = _SAMPLE_SPLIT.split(full)
    # parts = [pre, id1, block1, id2, block2, ...]
    for i in range(1, len(parts) - 1, 2):
        sid, block = parts[i], parts[i + 1]
        dec = _THESIS_DEC.search(block)
        if not dec:
            continue
        score = _THESIS_SCORE.search(block)
        label = int(score.group(1)) if score else int(dec.group(1))
        rationale = dec.group(2)
        # Third, independent signal: the rationale prose. Disagreement flags a
        # source inconsistency (e.g. 2023 LEQ2 2C: digit=0 but prose "earned 1 point").
        prose = None
        if _PROSE_DENY.search(rationale):
            prose = 0
        elif _PROSE_EARN.search(rationale):
            prose = 1
        contradictory = prose is not None and prose != label
        quotes = _QUOTE.findall(rationale)
        thesis = _clean(max(quotes, key=len)) if quotes else None
        out.append(
            {
                "sample_id": sid,
                "label": label,
                "thesis": thesis,
                "reason": _clean(rationale)[:600],
                "needs_vision": thesis is None,
                "contradictory": contradictory,
            }
        )
    return out


# --- Tier 1: rubric example theses (two-column Row A page) -------------------


def _rowa_page(pdf):
    for p in pdf.pages:
        t = p.extract_text() or ""
        if "Thesis/Claim" in t and "Examples that" in t and "earn this point" in t:
            return p
    return None


def parse_rowa_examples(pdf) -> List[dict]:
    """Split the Row A page into left (not-earn=0) / right (earn=1) columns by x,
    then pull each quoted example plus the nearest preceding sub-heading."""
    page = _rowa_page(pdf)
    if not page:
        return []
    mid = page.width / 2
    out = []
    for label, crop in ((0, (0, 0, mid, page.height)), (1, (mid, 0, page.width, page.height))):
        text = page.crop(crop).extract_text() or ""
        # keep only the "Examples that ..." region (below the criteria bullets)
        idx = text.find("Examples that")
        region = text[idx:] if idx >= 0 else text
        for q in _QUOTE.findall(region):  # multi-line quotes (re.S) => the example theses
            thesis = _clean(q)
            if len(thesis) > 12:
                out.append({"label": label, "thesis": thesis, "subcat": None})
    return out


# --- vision fallback for un-quoted theses -----------------------------------

_VISION_SYS = (
    "You transcribe handwritten AP World History exam essays from scanned images, "
    "then extract the thesis. Transcribe faithfully, preserving the student's spelling "
    "and grammar. The thesis is the sentence(s) stating a defensible claim, in the "
    "introduction or conclusion. Return JSON: "
    '{"essay": "...full transcription...", "thesis": "...the thesis sentence(s)..."}'
)


def _sample_pages(pdf, sample_id: str):
    """(page_index, page_text) for a sample's response pages, across all layouts:
    '1 of 3 2A' (2024 scan), 'Page 1 of 3 1A' (2023 scan), 'Sample 2A Page 1 of 2 ...'
    (2025 typed transcription)."""
    sid = re.escape(sample_id)
    hdr_scan = re.compile(rf"^(?:Page\s+)?\d+\s*of\s*\d+\s+{sid}\b")
    hdr_typed = re.compile(rf"^Sample\s+{sid}\s+Page")
    pages = []
    for i, p in enumerate(pdf.pages):
        t = p.extract_text() or ""
        head = t.strip()[:40]
        if hdr_scan.match(head) or hdr_typed.match(head):
            pages.append((i, t))
    return pages


def _render(pdf_path: Path, page_idxs: List[int]) -> List[str]:
    import fitz  # PyMuPDF

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    paths = []
    for i in page_idxs:
        pix = doc[i].get_pixmap(dpi=150)
        out = IMG_DIR / f"{pdf_path.stem}_p{i}.png"
        pix.save(out)
        paths.append(str(out))
    doc.close()
    return paths


_TEXT_THESIS_SYS = (
    "You extract the thesis from a transcribed AP World History essay. The thesis is "
    "the sentence(s) making a historically defensible claim that answers the prompt, "
    "located in the introduction or conclusion. Copy it verbatim from the text. "
    'Return JSON: {"thesis": "..."}'
)


def fill_missing(records: List[ThesisRecord], teacher) -> None:
    """Recover theses the commentary didn't quote: from typed page text when the
    essay is transcribed (2025), else via gpt-4o vision on the handwritten scan."""
    from src.teacher import extract_json

    need = [r for r in records if r.needs_vision and r.sample_id]
    for r in need:
        path = PDF_DIR / r.source_pdf
        with pdfplumber.open(path) as pdf:
            pages = _sample_pages(pdf, r.sample_id)
        if not pages:
            continue
        body = _clean("\n".join(t for _, t in pages))
        typed = len(body) > 250  # transcribed essay text present (2025 layout)
        try:
            if typed:
                raw = teacher.chat(
                    [{"role": "system", "content": _TEXT_THESIS_SYS},
                     {"role": "user", "content": f"PROMPT: {r.prompt}\n\nESSAY:\n{body}"}],
                    temperature=0.0, json_mode=True, max_tokens=300,
                )
            else:
                imgs = _render(path, [i for i, _ in pages])
                raw = teacher.vision(_VISION_SYS, "Transcribe and extract the thesis.", imgs)
            thesis = _clean(extract_json(raw).get("thesis") or "")
            if thesis:
                r.thesis = thesis
                r.needs_vision = False
        except Exception:
            pass


# --- driver -----------------------------------------------------------------


def parse_pdf(path: Path) -> List[ThesisRecord]:
    year, qtype, set_no = _meta_from_name(path.name)
    records: List[ThesisRecord] = []
    with pdfplumber.open(path) as pdf:
        full = _full_text(pdf)
        prompt = extract_prompt(full)
        for ex in parse_rowa_examples(pdf):
            records.append(
                ThesisRecord(path.name, year, set_no, qtype, None, prompt,
                             ex["thesis"], ex["label"], "rubric example", 1,
                             subcat=ex.get("subcat"))
            )
        for s in parse_commentary(full):
            records.append(
                ThesisRecord(path.name, year, set_no, qtype, s["sample_id"], prompt,
                             s["thesis"], s["label"], s["reason"], 2,
                             needs_vision=s["needs_vision"],
                             contradictory=s["contradictory"])
            )
    return records


def run(use_vision: bool, dry_run: bool) -> None:
    pdfs = sorted(p for p in PDF_DIR.glob("*apc*.pdf"))
    all_recs: List[ThesisRecord] = []
    for p in pdfs:
        recs = parse_pdf(p)
        all_recs.extend(recs)
    t1 = [r for r in all_recs if r.tier == 1]
    t2 = [r for r in all_recs if r.tier == 2]
    missing = [r for r in t2 if r.needs_vision]
    contra = [r for r in t2 if r.contradictory]
    print(f"parsed {len(pdfs)} sample PDFs -> tier1={len(t1)} tier2={len(t2)} "
          f"(tier2 missing thesis={len(missing)}, source-contradictory={len(contra)})")
    for r in contra:
        print(f"  CONTRADICTORY: {r.source_pdf} {r.sample_id} digit={r.label} vs prose")
    if use_vision and missing:
        from src.teacher import Teacher
        teacher = Teacher(dry_run=dry_run)
        fill_missing(all_recs, teacher)
        still = sum(1 for r in all_recs if r.tier == 2 and r.needs_vision)
        print(f"vision fill done; tier2 still missing={still}")
    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    with OUT_RAW.open("w") as f:
        for r in all_recs:
            f.write(json.dumps(asdict(r)) + "\n")
    print(f"wrote {len(all_recs)} records -> {OUT_RAW}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vision", action="store_true", help="gpt-4o vision for missing theses")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(use_vision=args.vision, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
