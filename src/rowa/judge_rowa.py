"""Decomposed, bias-resistant Row A verifier.

A holistic "would you award the point?" call to a frontier model imports the
essay-quality bias the whole project is about (it denies minimal-but-defensible
theses). So instead we ask only the rubric's *objective sub-questions* and compute
the Row A decision deterministically from them:

    earns Row A  <=>  defensible AND responsive AND has_reasoning AND NOT restatement

This mirrors the College Board decision rules directly and sidesteps the quality
substitution. It is used to (a) verify synthetic training labels and (b) serve as an
alternative grader in eval.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.rowa import rubric
from src.teacher import Teacher, extract_json

_SYS = f"""You verify one AP World History Row A (thesis) decision by answering only objective, factual sub-questions. You are NOT judging essay quality, sophistication, eloquence, or whether the thesis "evaluates the extent" or has a "broader analytical framework" — those are IRRELEVANT to Row A.

Here is the Row A rubric for reference:
{rubric.RUBRIC}

Answer these sub-questions about the thesis, then nothing else:
- defensible: Is the claim historically defensible/accurate (not factually wrong)?
- responsive: Does it address the topic of THIS prompt?
- has_reasoning: Does it state at least one reason for its claim OR establish analytic categories of an argument? (A bare defensible claim with a single "because"/reason, or 2+ named categories, counts as YES.)
- restatement: Is it merely a restatement/rephrasing of the prompt with no claim of its own?

Return ONLY JSON:
{{"defensible": bool, "responsive": bool, "has_reasoning": bool, "restatement": bool, "reason": "one short sentence"}}"""


@dataclass
class Decomposed:
    label: int
    defensible: bool
    responsive: bool
    has_reasoning: bool
    restatement: bool
    reason: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _label(d: dict) -> int:
    return int(
        bool(d.get("defensible"))
        and bool(d.get("responsive"))
        and bool(d.get("has_reasoning"))
        and not bool(d.get("restatement"))
    )


def decomposed_grade(teacher: Teacher, prompt: str, thesis: str) -> Decomposed:
    user = rubric.grader_user(prompt, thesis) + "\nAnswer the sub-questions. JSON only."
    raw = teacher.chat(
        [{"role": "system", "content": _SYS}, {"role": "user", "content": user}],
        temperature=0.0, json_mode=True, max_tokens=250,
    )
    d = extract_json(raw)
    return Decomposed(
        label=_label(d),
        defensible=bool(d.get("defensible")),
        responsive=bool(d.get("responsive")),
        has_reasoning=bool(d.get("has_reasoning")),
        restatement=bool(d.get("restatement")),
        reason=str(d.get("reason", "")),
    )
