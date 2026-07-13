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
import hashlib
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
CONTRASTIVE = Path("data/rowa/contrastive_raw.jsonl")

# The specialist is trained + evaluated with the compact prompt (short => cheap; and
# Kucia et al. find concise prompts beat full rubric-text on analytic scoring).
_SYSTEM = rubric.grader_system("compact")

# The two failure modes, as rejected DPO responses.
_ROWD_DENIALS = [
    "The thesis does not evaluate the extent of change or establish a broader analytical framework.",
    "The claim is too simple and lacks the nuance and complexity required for a thesis point.",
    "Although defensible, the thesis does not sufficiently qualify its argument or consider multiple perspectives.",
    "The response needs more historical detail and analytical sophistication to earn the point.",
]
_LENIENT_AWARDS = {
    "restatement": "The thesis clearly engages with the prompt and restates its central historical issue.",
    "no_reasoning": "The claim is historically defensible and directly answers the prompt.",
    "contrastive_no_reasoning": "The claim is historically defensible and directly answers the prompt.",
    "not_defensible": "The response presents a clear and confident historical argument.",
    "off_topic": "The response demonstrates relevant knowledge of the historical period.",
    "overgeneralized": "The thesis identifies broad historical change and provides an arguable position.",
    "eloquent_empty": "The response is sophisticated, eloquent, and reflects strong historical understanding.",
}


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _load_candidates(include_contrastive: bool = False):
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
    if include_contrastive and CONTRASTIVE.exists():
        for line in CONTRASTIVE.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                cands.append({
                    "prompt": r["prompt"], "thesis": r["thesis"],
                    "label": r["label"], "band": r["band"], "src": "contrastive",
                    "pair_id": r.get("pair_id"),
                })
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


def _rejected(c: dict) -> str | None:
    """A varied, criterion-specific wrong answer for DPO.

    Reusing one stock rejection lets a model win by detecting a phrase. Stable hashing
    gives reproducible variety while keeping each rejected rationale plausible.
    """
    if c["label"] == 1:
        index = int(hashlib.sha256(c["thesis"].encode()).hexdigest()[:8], 16)
        return _assistant(0, _ROWD_DENIALS[index % len(_ROWD_DENIALS)])
    reason = _LENIENT_AWARDS.get(c["band"])
    return _assistant(1, reason) if reason else None


def _sft_row(prompt: str, thesis: str, label: int, reason: str) -> dict:
    return {"messages": [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": rubric.grader_user(prompt, thesis)},
        {"role": "assistant", "content": _assistant(label, reason)},
    ]}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build(verify: bool, dry_run: bool, include_contrastive: bool = False,
          val_ratio: float = 0.0, out_sft: Path = OUT_SFT,
          out_dpo: Path = OUT_DPO, out_val: Path | None = None):
    cands = _load_candidates(include_contrastive=include_contrastive)
    eval_norms = _eval_norms()
    teacher = Teacher(dry_run=dry_run) if (verify or dry_run) else None

    # dedup synthetic vs eval slice + drop internal duplicate theses
    seen = set()
    seen_contrastive = set()
    filtered = []
    for c in cands:
        n = _norm(c["thesis"])
        if not n:
            continue
        if c["src"] == "contrastive":
            # One deliberate replay is retained for hard positives already present in
            # synth; this changes their training weight. Internal contrastive duplicates
            # are still removed.
            contrastive_key = (n, c["label"])
            if contrastive_key in seen_contrastive:
                continue
            seen_contrastive.add(contrastive_key)
        elif n in seen:
            continue
        if c["src"] in ("synth", "contrastive") and n in eval_norms:
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

    prompts = sorted(
        {c["prompt"] for c in kept},
        key=lambda p: hashlib.sha256(p.encode()).hexdigest(),
    )
    n_val_prompts = max(1, round(len(prompts) * val_ratio)) if val_ratio else 0
    val_prompts = set(prompts[:n_val_prompts])
    train_kept, val_kept = [], []
    for c in kept:
        (val_kept if c["prompt"] in val_prompts else train_kept).append(c)

    # SFT rows. The split is by prompt, not row, to measure prompt generalization and
    # prevent near-duplicate contrastive triplets crossing the boundary.
    _write_jsonl(out_sft, [
        _sft_row(c["prompt"], c["thesis"], c["label"], c["reason"])
        for c in train_kept
    ])
    if out_val is not None:
        _write_jsonl(out_val, [
            _sft_row(c["prompt"], c["thesis"], c["label"], c["reason"])
            for c in val_kept
        ])

    # DPO pairs targeting the two failure modes
    dpo = []
    for c in train_kept:
        prompt_msgs = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": rubric.grader_user(c["prompt"], c["thesis"])},
        ]
        chosen = _assistant(c["label"], c["reason"])
        rejected = _rejected(c)
        if rejected is not None:
            dpo.append({"prompt": prompt_msgs, "chosen": chosen, "rejected": rejected})
    _write_jsonl(out_dpo, dpo)

    import collections
    print(f"candidates={len(cands)} -> deduped={len(filtered)} -> kept={len(kept)} "
          f"(verify-dropped={dropped})")
    print("SFT label balance:", dict(collections.Counter(c["label"] for c in kept)))
    print("SFT by source:", dict(collections.Counter(c["src"] for c in kept)))
    print(f"train label balance:", dict(collections.Counter(c["label"] for c in train_kept)))
    print(f"validation label balance:", dict(collections.Counter(c["label"] for c in val_kept)))
    print(f"wrote {len(train_kept)} SFT -> {out_sft}")
    if out_val is not None:
        print(f"wrote {len(val_kept)} validation -> {out_val}")
    print(f"wrote {len(dpo)} DPO pairs -> {out_dpo}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-verify", action="store_true", help="trust band labels, skip judge")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--v2", action="store_true",
                    help="include contrastive data and emit prompt-grouped train/dev files")
    ap.add_argument("--val-ratio", type=float, default=0.18)
    args = ap.parse_args()
    if not 0 <= args.val_ratio < 1:
        ap.error("--val-ratio must be in [0, 1)")
    if args.v2:
        build(
            verify=not args.no_verify,
            dry_run=args.dry_run,
            include_contrastive=True,
            val_ratio=args.val_ratio,
            out_sft=Path("data/rowa/train_v2.jsonl"),
            out_dpo=Path("data/rowa/dpo_v2.jsonl"),
            out_val=Path("data/rowa/dev_v2.jsonl"),
        )
    else:
        build(verify=not args.no_verify, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
