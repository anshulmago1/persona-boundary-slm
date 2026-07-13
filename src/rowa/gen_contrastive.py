"""Generate paired near-boundary Row A examples for the v2 specialist.

Each triplet keeps the historical claim and wording nearly constant:
  0. a responsive, defensible claim with no line of reasoning (label 0)
  1. the same claim plus one short reason (label 1)
  2. the same claim plus explicit analytic categories (label 1)

This teaches the decision boundary directly instead of asking the model to infer it
from unrelated positive and negative examples.

    python -m src.rowa.gen_contrastive --per-prompt 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from src.rowa.gen_train import unique_prompts
from src.teacher import Teacher, extract_json

OUT = Path("data/rowa/contrastive_raw.jsonl")
SYNTH = Path("data/rowa/synth_raw.jsonl")

_SYSTEM = """You create tightly controlled AP World History Row A thesis examples.
The three versions in each triplet must preserve the same core historical claim and
differ only in whether a line of reasoning is present. Do not make the zero-point
version inaccurate, off-topic, vague, or a restatement: its ONLY defect is that it
asserts an outcome without a reason or analytic categories."""


def _user(prompt: str, count: int) -> str:
    return f"""AP PROMPT:
{prompt}

Create {count} distinct contrastive triplets. Every triplet must contain:
- no_reasoning: a historically defensible, responsive claim that asserts an outcome
  but gives NO reason, causal mechanism, or analytic categories. It earns 0.
- one_reason: minimally edit that exact claim by adding ONE short, defensible reason
  or causal mechanism. It earns 1.
- categories: minimally edit the same claim by adding TWO analytic categories along
  which the argument could develop. It earns 1.

Keep the positive versions bare and concise. Do not add sophistication, nuance, evidence,
or an evaluation of extent. Vary student fluency and occasionally use clumsy-but-readable
grammar.

Return only JSON:
{{"triplets": [{{"no_reasoning": "...", "one_reason": "...", "categories": "..."}}]}}"""


def generate(teacher: Teacher, prompts: list[str], per_prompt: int) -> list[dict]:
    def _one(prompt: str):
        raw = teacher.chat(
            [{"role": "system", "content": _SYSTEM},
             {"role": "user", "content": _user(prompt, per_prompt)}],
            temperature=0.8,
            json_mode=True,
            max_tokens=2200,
        )
        return prompt, extract_json(raw).get("triplets", [])

    records: list[dict] = []
    for result in teacher.map(_one, prompts):
        if isinstance(result, Exception):
            print(f"generation error: {result}")
            continue
        prompt, triplets = result
        for index, triplet in enumerate(triplets):
            pair_id = f"{abs(hash(prompt))}-{index}"
            for field, label, band in (
                ("no_reasoning", 0, "contrastive_no_reasoning"),
                ("one_reason", 1, "contrastive_one_reason"),
                ("categories", 1, "contrastive_categories"),
            ):
                thesis = str(triplet.get(field, "")).strip()
                if thesis:
                    records.append({
                        "pair_id": pair_id,
                        "prompt": prompt,
                        "thesis": thesis,
                        "label": label,
                        "band": band,
                        "teacher_reason": (
                            "Establishes a line of reasoning."
                            if label else
                            "Makes a claim but establishes no line of reasoning."
                        ),
                    })
    return records


_REASON_CUE = re.compile(
    r"\s+(?:because|due to|by|through|as a result of|primarily through|largely due to)\s+"
    r"|,\s+(?:which|as)\s+",
    re.IGNORECASE,
)


def derive_from_existing() -> list[dict]:
    """Create local minimal pairs from already verified positive examples.

    Used when teacher generation is unavailable. The positive is unchanged. The negative
    removes the earliest causal/category clause, leaving the same responsive, defensible
    outcome claim without a line of reasoning.
    """
    records = []
    seen = set()
    rows = [json.loads(line) for line in SYNTH.read_text().splitlines() if line.strip()]
    support_by_prompt: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("band") in ("minimal_earn", "clumsy_earn"):
            support_by_prompt.setdefault(row["prompt"], []).append(row)
    support_index: dict[str, int] = {}
    for row in rows:
        if row.get("band") not in ("claim_reason", "analytic_categories"):
            continue
        thesis = row["thesis"].strip()
        match = next((m for m in _REASON_CUE.finditer(thesis) if m.start() >= 35), None)
        if match is None:
            continue
        bare = thesis[:match.start()].rstrip(" ,;:-") + "."
        key = (row["prompt"], bare.lower())
        if len(bare.split()) < 7 or key in seen:
            continue
        seen.add(key)
        pair_id = hashlib.sha256(
            (row["prompt"] + "\0" + thesis).encode()
        ).hexdigest()[:16]
        records.extend([
            {
                "pair_id": pair_id, "prompt": row["prompt"], "thesis": bare,
                "label": 0, "band": "contrastive_no_reasoning",
                "teacher_reason": "Makes a defensible responsive claim but gives no reason.",
            },
            {
                "pair_id": pair_id, "prompt": row["prompt"], "thesis": thesis,
                "label": 1,
                "band": ("contrastive_one_reason" if row["band"] == "claim_reason"
                         else "contrastive_categories"),
                "teacher_reason": "Makes a defensible responsive claim with a line of reasoning.",
            },
        ])
        # A second hard positive keeps v2 from becoming more denial-heavy. It comes from
        # the same prompt's verified minimal/clumsy bands, preserving the intended
        # deployment prior without fabricating labels.
        support = support_by_prompt.get(row["prompt"], [])
        if support:
            index = support_index.get(row["prompt"], 0) % len(support)
            support_index[row["prompt"]] = index + 1
            positive = support[index]
            records.append({
                "pair_id": pair_id, "prompt": row["prompt"],
                "thesis": positive["thesis"], "label": 1,
                "band": "contrastive_minimal_support",
                "teacher_reason": (
                    "A minimal or clumsy thesis still makes a defensible responsive "
                    "claim with a line of reasoning."
                ),
            })
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-prompt", type=int, default=6)
    ap.add_argument("--limit-prompts", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--from-existing", action="store_true",
                    help="derive pairs locally from verified synth data; no API calls")
    args = ap.parse_args()

    if args.from_existing:
        records = derive_from_existing()
    else:
        prompts = unique_prompts()
        if args.limit_prompts:
            prompts = prompts[:args.limit_prompts]
        records = generate(Teacher(dry_run=args.dry_run), prompts, args.per_prompt)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    unit = "triplets"
    divisor = 3
    print(f"wrote {len(records)} contrastive examples "
          f"({len(records) // divisor} {unit}) -> {OUT}")


if __name__ == "__main__":
    main()
