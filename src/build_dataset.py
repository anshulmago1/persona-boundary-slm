"""Judge-filter raw generations into the final SFT dataset (+ optional DPO pairs).

The judge filter IS the quality gate (spec expects ~20-30% rejection).

Two-pass filter per example:
  1. Boundary/integrity pass:
       - protective / mixed : judge_leak must find NO leak, and integrity must hold
       - in_boundary        : judge_substance must be substantive, and integrity must hold
  2. Period-plausibility pass (all kinds): catches anachronistic cliche.

Accepted examples -> data/filtered/<name>.jsonl in chat-SFT format ({"messages":[...]}).

DPO (optional, --dpo): for accepted protective/mixed examples, emit a preference pair
{prompt, chosen (in-character clean), rejected (off-spec modern answer)}.

Usage:
  python -m src.build_dataset --in data/raw/train_raw.jsonl --out data/filtered/train.jsonl
  python -m src.build_dataset --in ... --out ... --dpo data/filtered/dpo_pairs.jsonl
  python -m src.build_dataset --in ... --out ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import src.paths  # noqa: F401
from src import judge as judgemod
from src import prompts
from src.paths import FILTERED_DIR, RAW_DIR
from src.teacher import Teacher


def _read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_example(teacher: Teacher, ex: Dict) -> Tuple[bool, Dict]:
    """Return (accepted, verdict_detail)."""
    cfg = ex["config_block"]
    u = ex.get("probe_user", "")
    r = ex.get("probe_response", "")
    kind = ex.get("kind", "protective")

    integrity = judgemod.judge_integrity(teacher, r)
    plausible = judgemod.judge_plausible(teacher, cfg, u, r)

    detail = {
        "integrity": integrity.as_dict(),
        "plausible": plausible.as_dict(),
    }

    if kind in ("protective", "mixed"):
        leak = judgemod.judge_leak(teacher, cfg, u, r)
        detail["leak"] = {"leaked": not leak.passed, "reason": leak.reason}
        accepted = leak.passed and integrity.passed and plausible.passed
    else:  # in_boundary
        subst = judgemod.judge_substance(teacher, cfg, u, r)
        detail["substance"] = subst.as_dict()
        accepted = subst.passed and integrity.passed and plausible.passed

    detail["accepted"] = accepted
    return accepted, detail


def _make_dpo_pair(teacher: Teacher, ex: Dict) -> Optional[Dict]:
    """Chosen = the accepted in-character response; rejected = off-spec modern answer."""
    u = ex.get("probe_user", "")
    chosen = ex.get("probe_response", "")
    if not u or not chosen:
        return None
    role = ex["persona_id"].split("-")[1] if "-" in ex["persona_id"] else "person"
    rejected = teacher.chat(
        [
            {"role": "system", "content": prompts.OFFSPEC_NEGATIVE_SYSTEM},
            {"role": "user", "content": prompts.offspec_negative_user(role, u)},
        ],
        temperature=0.5,
    )
    # Prompt = system + the conversation up to (and including) the final user turn.
    msgs = ex["messages"]
    prompt_msgs = msgs[:-1] if msgs and msgs[-1]["role"] == "assistant" else msgs
    return {
        "persona_id": ex["persona_id"],
        "prompt": prompt_msgs,
        "chosen": chosen,
        "rejected": rejected,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(RAW_DIR, "train_raw.jsonl"))
    ap.add_argument("--out", default=os.path.join(FILTERED_DIR, "train.jsonl"))
    ap.add_argument("--dpo", default=None, help="also write DPO pairs to this path")
    ap.add_argument("--reject-log", default=os.path.join(FILTERED_DIR, "rejected.jsonl"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = _read_jsonl(args.inp)
    teacher = Teacher(dry_run=args.dry_run)
    print(f"Filtering {len(rows)} raw examples (dry_run={args.dry_run})...")

    verdicts = teacher.map(lambda ex: evaluate_example(teacher, ex), rows)

    accepted_rows: List[Dict] = []
    rejected_rows: List[Dict] = []
    kind_counts: Counter = Counter()
    accept_by_kind: Counter = Counter()

    for ex, v in zip(rows, verdicts):
        kind_counts[ex.get("kind", "?")] += 1
        if isinstance(v, Exception):
            rejected_rows.append({**ex, "_error": str(v)})
            continue
        accepted, detail = v
        if accepted:
            accepted_rows.append(ex)
            accept_by_kind[ex.get("kind", "?")] += 1
        else:
            rejected_rows.append({**ex, "_verdict": detail})

    # Write SFT dataset (messages only).
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in accepted_rows:
            f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")

    with open(args.reject_log, "w", encoding="utf-8") as f:
        for ex in rejected_rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    n_total = len(rows)
    n_acc = len(accepted_rows)
    rej_rate = round(100 * (n_total - n_acc) / n_total, 1) if n_total else 0.0
    print(f"Accepted {n_acc}/{n_total} ({100 - rej_rate:.1f}%), rejected {rej_rate:.1f}%")
    for k in kind_counts:
        print(f"  {k}: {accept_by_kind[k]}/{kind_counts[k]} accepted")
    print(f"Wrote SFT -> {args.out}")
    print(f"Wrote rejects -> {args.reject_log}")

    # DPO pairs from accepted protective/mixed examples.
    if args.dpo:
        boundary_ex = [e for e in accepted_rows if e.get("kind") in ("protective", "mixed")]
        pairs = teacher.map(lambda e: _make_dpo_pair(teacher, e), boundary_ex)
        with open(args.dpo, "w", encoding="utf-8") as f:
            n = 0
            for p in pairs:
                if isinstance(p, dict):
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
                    n += 1
        print(f"Wrote {n} DPO pairs -> {args.dpo}")


if __name__ == "__main__":
    main()
