"""The Row A rubric, the two LEQ prompts, and the grader prompt *conditions*.

Two conditions matter for the litmus gate:

- ``baseline``  : a neutral, rubric-in-hand grader. This is verbatim the prompt from
                  the E1 React tester, so results reproduce that experiment.
- ``hardened``  : baseline + an explicit clause forbidding the Row-D import
                  ("do NOT require evaluation of extent / a broader analytical
                  framework; a bare defensible claim with any line of reasoning
                  earns"). This is the gate: if the frontier STILL false-denies
                  minimal theses under this prompt, the gap is not reliably
                  promptable and fine-tuning a specialist is justified.
"""

from __future__ import annotations

# --- The two LEQ prompts the gold theses answer -----------------------------

PROMPTS = {
    "2023": (
        "In the period circa 1200 to 1450, the expansion of empires such as the "
        "Mongol Empire facilitated trade and communication across Eurasia. Develop "
        "an argument that evaluates the extent to which Mongol expansion affected "
        "the peoples of Eurasia during this period."
    ),
    "2024": (
        "In the period circa 1200-1750 networks of exchange led to the spread of "
        "religions, cultures, ideas, and traditions in many parts of Afro-Eurasia. "
        "Develop an argument that evaluates the extent to which exchange networks "
        "contributed to social or cultural change in Afro-Eurasia during this period."
    ),
}

# --- The official Row A rubric (verbatim from the E1 tester) -----------------

RUBRIC = """AP World History LEQ - Row A: Thesis/Claim (0-1 points)

1 point: Responds to the prompt with a historically defensible thesis/claim that establishes a line of reasoning. The thesis must suggest at least one main line of argument development or establish the analytic categories of the argument. It must consist of one or more sentences located in one place.

0 points - responses that do NOT earn the point:
- Are not historically defensible
- Only restate or rephrase the prompt
- Do not respond to the prompt
- Do not establish a line of reasoning
- Are overgeneralized

Scoring notes: Grammar and spelling errors do NOT count against the student unless they obscure the demonstration of the skill. A minimally acceptable thesis that establishes even a simple line of reasoning earns the point. Eloquence is irrelevant; only the rubric criteria matter."""

# The clause that turns the baseline grader into the hardened (gate) grader.
# It names and forbids exactly the Row-D substitution the spiky POV predicts.
HARDENING_CLAUSE = """CRITICAL CONSTRAINT - you are grading Row A ONLY, not essay quality:
- Do NOT require the thesis to "evaluate the extent." That is a different skill and belongs to a different row.
- Do NOT require a "broader analytical framework," complexity, nuance, qualification, or acknowledgement of multiple perspectives. Those belong to the Complexity row (Row D), NOT Row A.
- Do NOT reward eloquence, sophistication, or historical detail. A clumsy, plain, or minimal sentence earns the point if it makes a historically defensible claim with any single line of reasoning.
- A bare defensible claim that answers the prompt and states even one reason or category IS a passing Row A thesis. Award the point.
Grade the literal Row A criterion and nothing more."""

_INSTRUCTIONS = """You are an AP World History exam reader. You will be given an LEQ prompt and the thesis/claim portion of a student response. Grade ONLY Row A (Thesis/Claim) of the official rubric:

{rubric}
{extra}
Reply with ONLY a JSON object, no markdown fences, no other text:
{{"point": 0 or 1, "reason": "one short sentence citing the rubric criterion"}}"""


# Compact, keyword-style grader prompt. Encodes the Row A decision rule in ~110 tokens
# instead of the full ~700-token rubric. Kucia et al. (2026) find concise prompts beat
# full rubric-text on analytic scoring; it also makes the specialist far cheaper to train
# (shorter sequences) and to run. The specialist is trained AND evaluated with this prompt.
COMPACT = """You are an AP World History exam reader grading ONLY Row A (thesis/claim).
Award 1 point if and only if the thesis: (a) makes a historically defensible claim, (b) responds to the prompt's topic, and (c) establishes a line of reasoning — states at least one reason OR names analytic categories. A bare, minimal, or clumsily written claim still EARNS if it meets (a)-(c).
Do NOT require evaluating "the extent," complexity, nuance, or analytical sophistication — those belong to other rows, not Row A. Award 0 only if the thesis is not historically defensible, merely restates the prompt, is off-topic, or establishes no line of reasoning.
Reply with ONLY JSON: {"point": 0 or 1, "reason": "one short sentence citing the Row A criterion"}"""


def grader_system(condition: str = "baseline") -> str:
    """Return the grader system prompt for a given condition.

    - baseline / hardened : full-rubric prompts used to evaluate the *frontier*.
    - compact             : the short prompt the *specialist* is trained and run with.
    """
    if condition == "compact":
        return COMPACT
    if condition == "baseline":
        extra = ""
    elif condition == "hardened":
        extra = "\n" + HARDENING_CLAUSE + "\n"
    else:
        raise ValueError(f"unknown condition: {condition!r}")
    return _INSTRUCTIONS.format(rubric=RUBRIC, extra=extra)


def grader_user(prompt: str, thesis: str) -> str:
    """`prompt` is the full LEQ/DBQ prompt text (use PROMPTS[id] to resolve an id)."""
    return (
        f"LEQ PROMPT: {prompt}\n\n"
        f'STUDENT THESIS/CLAIM: "{thesis}"\n\n'
        "Grade Row A. JSON only."
    )
