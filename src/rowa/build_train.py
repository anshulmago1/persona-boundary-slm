"""Turn synthetic theses + official rubric examples into SFT (+ DPO) training data.

Pipeline:
1. Candidates = synthetic theses (band-labeled) + tier-1 rubric examples (official labels).
2. Verify each with the decomposed rubric judge (``judge_rowa``). Keep a synthetic item
   only when the judge's label matches its band label (drops generation drift). Rubric
   examples keep their official label; the judge just supplies a rubric-grounded reason.
3. Dedup synthetic theses against the **real-student eval slice** (subset=='sample') so
   nothing the specialist is tested on leaks into training.
4. Emit chat-format SFT rows (system = hardened grader prompt, assistant = {point,reason})
   and DPO pairs that pit the correct call against the exact failure modes: minimal-earn
   theses vs. a Row-D-substitution denial; eloquent-empty theses vs. a leniency award.

    python -m src.rowa.build_train                 # verify + build (uses API)
    python -m src.rowa.build_train --no-verify     # trust band labels, no API
    python -m src.rowa.build_train --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List

from src.rowa import rubric
from src.rowa.judge_rowa import decomposed_grade
from src.teacher import Teacher

SYNTH = Path("data/rowa/synth_raw.jsonl")
GOLD = Path("data/rowa/gold_all.jsonl")
OUT_SFT = Path("data/rowa/train.jsonl")
OUT_DPO = Path("data/rowa/dpo.jsonl")

# The specialist is trained + evaluated with the compact prompt (short => cheap; and
# Kucia et al. find concise prompts beat full rubric-text on analytic scoring).
_SYSTEM = rubric.grader_system("compact")

# The two failure modes, as rejected DPO responses.
_ROWD_DENIAL = json.dumps({"point": 0, "reason": "The thesis does not evaluate the extent "
    "of change or establish a broader analytical framework, and lacks analytical complexity."})
_LENIENT_AWARD = json.dumps({"point": 1, "reason": "The response is sophisticated, eloquent, "
    "and clearly reflects strong historical understanding."})


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _load_candidates():
    cands = []
    for line in SYNTH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            cands.append({"prompt": r["prompt"], "thesis": r["thesis"],
                          "label": r["label"], "band": r["band"], "src": "synth"})
    for line in GOLD.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("subset") == "rubric":
                cands.append({"prompt": r["prompt"], "thesis": r["thesis"],
                              "label": r["label"], "band": "rubric_example", "src": "rubric"})
    return cands


def _eval_norms():
    norms = set()
    for line in GOLD.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("subset") == "sample":
                norms.add(_norm(r["thesis"]))
    return norms


def _assistant(label: int, reason: str) -> str:
    return json.dumps({"point": int(label), "reason": reason})


def _sft_row(prompt: str, thesis: str, label: int, reason: str) -> dict:
    return {"messages": [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": rubric.grader_user(prompt, thesis)},
        {"role": "assistant", "content": _assistant(label, reason)},
    ]}


def build(verify: bool, dry_run: bool):
    cands = _load_candidates()
    eval_norms = _eval_norms()
    teacher = Teacher(dry_run=dry_run) if (verify or dry_run) else None

    # dedup synthetic vs eval slice + drop internal duplicate theses
    seen = set()
    filtered = []
    for c in cands:
        n = _norm(c["thesis"])
        if not n or n in seen:
            continue
        if c["src"] == "synth" and n in eval_norms:
            continue
        seen.add(n)
        filtered.append(c)

    # verify labels with the decomposed judge (parallel)
    kept, dropped = [], 0
    if verify:
        def _v(c):
            try:
                d = decomposed_grade(teacher, c["prompt"], c["thesis"])
                return c, d
            except Exception:
                return c, None
        for c, d in teacher.map(_v, filtered):
            if d is None:
                continue
            if c["src"] == "synth" and d.label != c["label"]:
                dropped += 1
                continue  # generation drift: judge disagrees with intended band
            reason = d.reason or ("Earns Row A: defensible claim with a line of reasoning."
                                  if c["label"] == 1 else "Does not earn Row A.")
            kept.append({**c, "reason": reason})
    else:
        for c in filtered:
            kept.append({**c, "reason": ("Earns Row A: makes a historically defensible claim "
                "that establishes a line of reasoning." if c["label"] == 1
                else "Does not earn Row A under the Row A decision rules.")})

    # SFT rows
    OUT_SFT.parent.mkdir(parents=True, exist_ok=True)
    with OUT_SFT.open("w") as f:
        for c in kept:
            f.write(json.dumps(_sft_row(c["prompt"], c["thesis"], c["label"], c["reason"])) + "\n")

    # DPO pairs targeting the two failure modes
    dpo = []
    for c in kept:
        prompt_msgs = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": rubric.grader_user(c["prompt"], c["thesis"])},
        ]
        chosen = _assistant(c["label"], c["reason"])
        if c["label"] == 1:  # minimal/earn theses vs the Row-D-substitution denial
            dpo.append({"prompt": prompt_msgs, "chosen": chosen, "rejected": _ROWD_DENIAL})
        elif c["band"] in ("eloquent_empty", "restatement", "overgeneralized"):
            dpo.append({"prompt": prompt_msgs, "chosen": chosen, "rejected": _LENIENT_AWARD})
    with OUT_DPO.open("w") as f:
        for d in dpo:
            f.write(json.dumps(d) + "\n")

    import collections
    print(f"candidates={len(cands)} -> deduped={len(filtered)} -> kept={len(kept)} "
          f"(verify-dropped={dropped})")
    print("SFT label balance:", dict(collections.Counter(c["label"] for c in kept)))
    print("SFT by source:", dict(collections.Counter(c["src"] for c in kept)))
    print(f"wrote {len(kept)} SFT -> {OUT_SFT}")
    print(f"wrote {len(dpo)} DPO pairs -> {OUT_DPO}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-verify", action="store_true", help="trust band labels, skip judge")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    build(verify=not args.no_verify, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
