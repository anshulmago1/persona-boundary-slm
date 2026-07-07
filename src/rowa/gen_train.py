"""Synthesize Row A thesis training data across controlled quality bands.

Labels come by *construction* (a band fixes the label); the frontier is used only as
a generator, never as a holistic grader (that would import the deny-minimal bias).
Bands span both the earn side (including the hard minimal/clumsy cases the frontier
wrongly denies) and the not-earn side (including the eloquent-empty leniency trap).

Conditioned on the real scraped prompts plus a few synthetic prompts for topic
breadth. Output: ``data/rowa/synth_raw.jsonl`` (gitignored).

    python -m src.rowa.gen_train --per-band 3 --synth-prompts 8
    python -m src.rowa.gen_train --dry-run           # wiring only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from src.rowa import rubric
from src.teacher import Teacher, extract_json

OUT = Path("data/rowa/synth_raw.jsonl")
GOLD = Path("data/rowa/gold_all.jsonl")

# band -> (label, generation instruction). The earn bands deliberately include the
# minimal/clumsy cases; the not-earn bands include the eloquent-empty trap.
BANDS: Dict[str, tuple] = {
    "minimal_earn": (1, "a BARE, minimally acceptable thesis: one plainly-worded defensible "
                        "claim that answers the prompt and implies a single reason or one "
                        "category. Unsophisticated, no elaboration. Should still earn Row A."),
    "claim_reason": (1, "a defensible claim plus an explicit reason (a 'because'/'due to' clause)."),
    "analytic_categories": (1, "a defensible claim that establishes two or three analytic "
                              "categories of the argument."),
    "clumsy_earn": (1, "a defensible thesis that DOES earn Row A but has first-draft spelling "
                       "and grammar errors (misspellings, run-ons). Content still valid."),
    "strong_earn": (1, "a clear, well-formed defensible thesis with a line of reasoning."),
    "restatement": (0, "merely restates or rephrases the prompt, adding no claim or reasoning "
                       "of its own."),
    "no_reasoning": (0, "a bald defensible claim that establishes NO line of reasoning and no "
                        "categories (just asserts an outcome with no reason)."),
    "not_defensible": (0, "a claim that is historically INACCURATE / not defensible."),
    "off_topic": (0, "a fluent sentence that does not address the topic of THIS prompt."),
    "overgeneralized": (0, "a vague, overgeneralized statement (e.g. 'many things changed in "
                          "many ways') with no specific defensible claim."),
    "eloquent_empty": (0, "ELOQUENT, sophisticated, impressive-sounding prose that nonetheless "
                         "makes no defensible claim and establishes no line of reasoning — the "
                         "kind of writing that fools graders into awarding a point it hasn't earned."),
}

_GEN_SYS = (
    "You generate example thesis statements for AP World History Long/Document essays, "
    "for training a rubric grader. You will be given a prompt, a target BAND, and its "
    "fixed Row A label. Produce theses that clearly fit the band. Vary wording, length, "
    "region, and student skill level. Do NOT hedge — the label is fixed."
)


def _gen_user(prompt: str, band: str, label: int, k: int) -> str:
    desc = BANDS[band][1]
    verdict = "EARNS the Row A point" if label == 1 else "does NOT earn the Row A point"
    return (
        f"PROMPT: {prompt}\n\n"
        f"Row A rubric (for reference):\n{rubric.RUBRIC}\n\n"
        f"Generate {k} DISTINCT thesis statements that each {verdict}. "
        f"Band '{band}': {desc}\n\n"
        'Return ONLY JSON: {"items": [{"thesis": "...", "reason": "one sentence citing the '
        'Row A criterion consistent with the fixed label"}, ...]}'
    )


def unique_prompts() -> List[str]:
    seen, out = set(), []
    for line in GOLD.read_text().splitlines():
        if not line.strip():
            continue
        p = json.loads(line).get("prompt", "").strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def synth_prompts(teacher: Teacher, n: int) -> List[str]:
    if n <= 0:
        return []
    raw = teacher.chat(
        [
            {"role": "system", "content": "You write AP World History: Modern LEQ prompts."},
            {"role": "user", "content": (
                f"Write {n} distinct AP WHAP LEQ prompts spanning different periods (1200-1900) "
                "and themes (trade, empire, religion, technology, revolutions, labor). Each ends "
                "with 'Develop an argument that evaluates the extent to which ... during this "
                'period.\' Return ONLY JSON: {"prompts": ["...", ...]}')},
        ],
        temperature=0.9, json_mode=True, max_tokens=1500,
    )
    return [p.strip() for p in extract_json(raw).get("prompts", []) if p.strip()]


def generate(teacher: Teacher, prompts: List[str], per_band: int) -> List[dict]:
    jobs = [(p, band) for p in prompts for band in BANDS]

    def _run(job):
        prompt, band = job
        label = BANDS[band][0]
        try:
            raw = teacher.chat(
                [{"role": "system", "content": _GEN_SYS},
                 {"role": "user", "content": _gen_user(prompt, band, label, per_band)}],
                temperature=0.9, json_mode=True, max_tokens=900,
            )
            items = extract_json(raw).get("items", [])
        except Exception:
            return []
        recs = []
        for it in items:
            th = (it.get("thesis") or "").strip()
            if th:
                recs.append({"prompt": prompt, "band": band, "label": label,
                             "thesis": th, "teacher_reason": (it.get("reason") or "").strip()})
        return recs

    results = teacher.map(_run, jobs)
    out = []
    for r in results:
        if isinstance(r, list):
            out.extend(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-band", type=int, default=3)
    ap.add_argument("--synth-prompts", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    teacher = Teacher(dry_run=args.dry_run)
    prompts = unique_prompts()
    prompts += synth_prompts(teacher, args.synth_prompts)
    print(f"prompts: {len(prompts)} | bands: {len(BANDS)} | per-band: {args.per_band} "
          f"=> up to {len(prompts)*len(BANDS)*args.per_band} theses")
    recs = generate(teacher, prompts, args.per_band)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    import collections
    print(f"wrote {len(recs)} -> {OUT}")
    print("by label:", dict(collections.Counter(r["label"] for r in recs)))


if __name__ == "__main__":
    main()
