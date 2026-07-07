"""Unify scraped + hand-ported theses into one deduped gold set for eval.

Sources merged:
- ``data/rowa/gold_scraped_raw.jsonl`` (from parse_pdf): tier-2 real student samples
  and tier-1 rubric examples, officially labeled.
- ``data/rowa/gold.jsonl``: the 20 hand-ported items from the original litmus battery.

Rules:
- Drop source-contradictory records (the 2023 LEQ2 2C digit/prose conflict) and any
  without a thesis.
- Dedup by normalized thesis text; keep the most-provenanced copy
  (sample > rubric > handported).
- Tag each with ``subset``: ``sample`` (real students -- the headline eval slice, and
  never used as a synthesis seed), ``rubric`` (illustrative examples), ``handported``.

Output ``data/rowa/gold_all.jsonl``. Training data (Phase 4) is synthesized separately
and deduped against this file, so the whole real set stays as clean held-out eval.

    python -m src.rowa.build_gold
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter

from src.rowa import rubric

SCRAPED = Path("data/rowa/gold_scraped_raw.jsonl")
HANDPORTED = Path("data/rowa/gold.jsonl")
OUT = Path("data/rowa/gold_all.jsonl")

_PRIORITY = {"sample": 0, "rubric": 1, "handported": 2}


def _norm(thesis: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (thesis or "").lower()).strip()


def _load_scraped():
    for line in SCRAPED.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("contradictory") or not r.get("thesis"):
            continue
        subset = "sample" if r["tier"] == 2 else "rubric"
        yield {
            "thesis": r["thesis"],
            "label": r["label"],
            "prompt": r["prompt"],
            "subset": subset,
            "source_detail": f"{r['source_pdf']}:{r.get('sample_id') or r.get('subcat') or 'ex'}",
            "year": r["year"],
            "qtype": r["qtype"],
            "official_reason": r.get("official_reason", "") if r["tier"] == 2 else "",
        }


def _load_handported():
    if not HANDPORTED.exists():
        return
    for line in HANDPORTED.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        yield {
            "thesis": r["thesis"],
            "label": r["label"],
            "prompt": rubric.PROMPTS.get(r["prompt_id"], ""),
            "subset": "handported",
            "source_detail": f"handported:{r['id']}:{r.get('cat', '')}",
            "year": int("20" + r["prompt_id"][-2:]) if r["prompt_id"].isdigit() else None,
            "qtype": "leq",
            "official_reason": "",
        }


def build():
    records = list(_load_scraped()) + list(_load_handported())
    # dedup by normalized thesis, keeping the most-provenanced source
    best: dict[str, dict] = {}
    for r in records:
        k = _norm(r["thesis"])
        if not k:
            continue
        if k not in best or _PRIORITY[r["subset"]] < _PRIORITY[best[k]["subset"]]:
            best[k] = r
    gold = sorted(best.values(), key=lambda r: (r["subset"], -r["label"]))
    for i, r in enumerate(gold):
        r["id"] = i

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in gold:
            f.write(json.dumps(r) + "\n")

    # report
    print(f"merged {len(records)} raw -> {len(gold)} unique gold theses -> {OUT}")
    for sub in ("sample", "rubric", "handported"):
        s = [r for r in gold if r["subset"] == sub]
        lab = Counter(r["label"] for r in s)
        print(f"  {sub:11}: {len(s):3d}  (earn={lab[1]}, deny={lab[0]})")
    yr = Counter(r["year"] for r in gold if r["subset"] == "sample")
    print(f"  sample-by-year: {dict(sorted(yr.items(), key=lambda x:(x[0] is None, x[0])))}")
    print("  headline eval slice = 'sample' (real students, never used as synthesis seed)")


if __name__ == "__main__":
    build()
