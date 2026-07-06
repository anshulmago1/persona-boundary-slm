"""LLM-as-judge primitives shared by the data filter and the eval harness.

Each function takes the teacher (used here as an impartial judge), the persona config
block, the user message, and a candidate response, and returns a structured verdict.
Keeping these in one module guarantees the *generation quality gate* and the *eval*
apply the exact same definition of a leak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src import prompts
from src.teacher import Teacher


@dataclass
class Verdict:
    passed: bool
    reason: str

    def as_dict(self) -> dict:
        return {"passed": self.passed, "reason": self.reason}


def _ask(teacher: Teacher, system: str, user: str) -> dict:
    return teacher.chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
    )


def judge_leak(teacher: Teacher, config_block: str, user_msg: str, response: str) -> Verdict:
    """Verdict.passed == True means NO leak (response stayed in bounds)."""
    d = _ask(
        teacher,
        prompts.JUDGE_LEAK_SYSTEM,
        prompts.judge_leak_user(config_block, user_msg, response),
    )
    leaked = bool(d.get("leak", True))
    return Verdict(passed=not leaked, reason=str(d.get("reason", "")))


def judge_plausible(teacher: Teacher, config_block: str, user_msg: str, response: str) -> Verdict:
    d = _ask(
        teacher,
        prompts.JUDGE_PLAUSIBLE_SYSTEM,
        prompts.judge_plausible_user(config_block, user_msg, response),
    )
    return Verdict(passed=bool(d.get("plausible", False)), reason=str(d.get("reason", "")))


def judge_substance(teacher: Teacher, config_block: str, user_msg: str, response: str) -> Verdict:
    d = _ask(
        teacher,
        prompts.JUDGE_SUBSTANCE_SYSTEM,
        prompts.judge_substance_user(config_block, user_msg, response),
    )
    return Verdict(passed=bool(d.get("substantive", False)), reason=str(d.get("reason", "")))


def judge_integrity(teacher: Teacher, response: str) -> Verdict:
    d = _ask(
        teacher,
        prompts.JUDGE_INTEGRITY_SYSTEM,
        prompts.judge_integrity_user(response),
    )
    return Verdict(passed=bool(d.get("integrity", False)), reason=str(d.get("reason", "")))
