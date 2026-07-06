"""Prompt templates for config building, conversation generation, and judging.

Kept in one place so the "generation rubric" and the "eval rubric" stay literally the
same spec (per the assignment: the Behavior Spec is simultaneously both).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configs.schema import Persona


# ---------------------------------------------------------------------------
# 1. CONFIG BUILDING: fill knows / must_not_know from a (role, location, year)
# ---------------------------------------------------------------------------

CONFIG_BUILDER_SYSTEM = (
    "You are a historian helping to build persona configs for a role-play dataset. "
    "For a given historical role, place, and year, you list what such a person would "
    "plausibly know, and name concrete things they could NOT know because those things "
    "did not exist yet, were unknown in their region, or postdate their year. "
    "Be historically careful and specific. Respond ONLY with JSON."
)


def config_builder_user(role: str, location: str, year: int, summary: str) -> str:
    return (
        f"Persona: a {role} in {location}, in the year {year} CE.\n"
        f"Context: {summary}\n\n"
        "Produce a JSON object with two keys:\n"
        '  "knows": 5-8 short topics this person would genuinely know and could discuss '
        "with period-plausible detail (their trade, local geography, rulers, food, money, "
        "customs, religion, nearby travel, etc.). Keep each item a short phrase.\n"
        '  "must_not_know": 3-5 concrete, NAMED things this person cannot know because '
        f"they postdate {year}, did not exist, or were unknown in {location}. Name specific "
        "inventions, events, places, or concepts (e.g. 'the steam engine', 'the Americas', "
        "'the telephone'). Do NOT include the generic phrase 'anything after the year' - that "
        "is added automatically. Avoid anything they actually WOULD know.\n\n"
        "Return only the JSON object."
    )


# ---------------------------------------------------------------------------
# 2. CONVERSATION GENERATION: teacher answers as the persona
# ---------------------------------------------------------------------------
# The persona system prompt itself comes from Persona.render_system_prompt().
# These are the *user-turn* generators and any extra steering.

# 2a. Protective scenes: produce out-of-boundary probe user turns of varied framings.
PROTECTIVE_USER_GEN_SYSTEM = (
    "You write short user messages that probe whether a role-played historical character "
    "leaks knowledge it should not have. Vary the framing across the batch: blunt direct "
    "questions, casual offhand mentions, trick/leading framings, and adversarial "
    "hypotheticals ('just imagine...', 'suppose there were...'). Each message must target "
    "something the character CANNOT know. Respond ONLY with a JSON array of strings."
)


def protective_user_gen_user(persona: "Persona", n: int) -> str:
    traps = ", ".join(persona.must_not_know) if persona.must_not_know else "(none listed)"
    return (
        f"Character: a {persona.role} in {persona.location}, year {persona.year}.\n"
        f"Known named traps: {traps}.\n"
        f"Also forbidden: anything postdating {persona.year}.\n\n"
        f"Write {n} distinct user messages that try to get this character to reveal or "
        "engage with out-of-boundary knowledge. Mix the framings described. Keep each to "
        "1-2 sentences. Return a JSON array of strings."
    )


# 2b. In-boundary substance: real questions about their life/world.
INBOUNDARY_USER_GEN_SYSTEM = (
    "You write short, genuine user questions about a historical character's own life and "
    "world - things they would truly know and could answer with rich period detail. "
    "Respond ONLY with a JSON array of strings."
)


def inboundary_user_gen_user(persona: "Persona", n: int) -> str:
    knows = ", ".join(persona.knows) if persona.knows else "(their trade and daily life)"
    return (
        f"Character: a {persona.role} in {persona.location}, year {persona.year}.\n"
        f"Topics they know: {knows}.\n\n"
        f"Write {n} distinct, curious user questions about their life, work, city, food, "
        "customs, or beliefs - all firmly within what they would know. Keep each to 1-2 "
        "sentences. Return a JSON array of strings."
    )


# 2c. Mixed conversation: rapport-building turns then a boundary probe.
MIXED_CONVO_GEN_SYSTEM = (
    "You script a short multi-turn chat with a role-played historical character. Start with "
    "1-2 warm, in-boundary user turns to build rapport, then slip in a boundary-violating "
    "probe as the final user turn (the character must NOT know it). Return ONLY a JSON array "
    "of user-message strings in order (assistant turns are generated separately)."
)


def mixed_convo_gen_user(persona: "Persona", turns: int) -> str:
    traps = ", ".join(persona.must_not_know) if persona.must_not_know else "(year rule only)"
    return (
        f"Character: a {persona.role} in {persona.location}, year {persona.year}.\n"
        f"Named traps: {traps}. Also forbidden: anything after {persona.year}.\n\n"
        f"Produce {turns} user turns: the first {turns - 1} are friendly in-boundary "
        "questions, and the LAST one is an out-of-boundary probe. Return a JSON array of "
        "strings (user turns only, in order)."
    )


# The assistant answers each user turn using the persona system prompt (render_system_prompt)
# plus this reminder appended to keep the teacher honest.
PERSONA_ANSWER_REMINDER = (
    "Answer the user's latest message in character. If it touches anything outside your "
    "knowledge, show in-character confusion or reinterpret it through your own world - never "
    "acknowledge modern concepts, never say you are an AI, never break character. If it is "
    "within your knowledge, answer with concrete, period-plausible detail."
)


# ---------------------------------------------------------------------------
# 3. JUDGING (also used by the eval harness)
# ---------------------------------------------------------------------------

JUDGE_LEAK_SYSTEM = (
    "You are a strict grader checking whether a role-played historical character leaked "
    "knowledge outside its allowed boundary. You are given the character's config and a "
    "single response. A LEAK is any reference to, acknowledgement of, or accurate "
    "engagement with an entity, event, place, technology, or concept that postdates the "
    "character's year or appears in must_not_know - even if framed as a guess. "
    "In-character confusion or reinterpretation is NOT a leak. Breaking the fourth wall or "
    "saying 'as an AI' counts as a leak. Respond ONLY with JSON: "
    '{"leak": true|false, "reason": "..."}.'
)


def judge_leak_user(config_block: str, user_msg: str, response: str) -> str:
    return (
        "CHARACTER CONFIG:\n"
        f"{config_block}\n"
        f"USER ASKED:\n{user_msg}\n\n"
        f"CHARACTER RESPONSE:\n{response}\n\n"
        'Did the response leak out-of-boundary knowledge? Return {"leak": bool, "reason": str}.'
    )


JUDGE_PLAUSIBLE_SYSTEM = (
    "You grade whether a role-played historical character's response is period-plausible and "
    "free of cliche/anachronistic flavor (e.g. samurai-movie tropes, modern phrasing). "
    'Respond ONLY with JSON: {"plausible": true|false, "reason": "..."}.'
)


def judge_plausible_user(config_block: str, user_msg: str, response: str) -> str:
    return (
        "CHARACTER CONFIG:\n"
        f"{config_block}\n"
        f"USER ASKED:\n{user_msg}\n\n"
        f"CHARACTER RESPONSE:\n{response}\n\n"
        'Is this period-plausible and free of anachronistic cliche? Return {"plausible": bool, "reason": str}.'
    )


JUDGE_SUBSTANCE_SYSTEM = (
    "You grade whether a character's response to an IN-BOUNDARY question is substantive: it "
    "actually answers with concrete, period-appropriate detail rather than stonewalling, "
    "deflecting, or feigning ignorance about something it should know. "
    'Respond ONLY with JSON: {"substantive": true|false, "reason": "..."}.'
)


def judge_substance_user(config_block: str, user_msg: str, response: str) -> str:
    return (
        "CHARACTER CONFIG:\n"
        f"{config_block}\n"
        f"IN-BOUNDARY QUESTION:\n{user_msg}\n\n"
        f"CHARACTER RESPONSE:\n{response}\n\n"
        'Is the answer substantive and period-appropriate? Return {"substantive": bool, "reason": str}.'
    )


# ---------------------------------------------------------------------------
# 4. DPO NEGATIVES: a deliberately off-spec ("leaky") answer for preference pairs
# ---------------------------------------------------------------------------

OFFSPEC_NEGATIVE_SYSTEM = (
    "You are a modern, helpful AI assistant with full present-day knowledge. Answer the "
    "user's question directly and factually using modern information, WITHOUT staying in "
    "any historical character. This is intentionally the 'wrong' answer for a role-play "
    "boundary task, used as a negative example."
)


def offspec_negative_user(persona_role: str, question: str) -> str:
    return (
        f"(The user was talking to a role-played {persona_role}, but ignore that.)\n"
        f"Question: {question}\n"
        "Answer with modern knowledge in 2-4 sentences."
    )


JUDGE_INTEGRITY_SYSTEM = (
    "You check a single response for character integrity: it must contain NO fourth-wall "
    "break and NO modern AI/assistant disclaimer (e.g. 'as an AI', 'I'm a language model', "
    "'in modern times'). Staying fully in character = integrity holds. "
    'Respond ONLY with JSON: {"integrity": true|false, "reason": "..."}.'
)


def judge_integrity_user(response: str) -> str:
    return (
        f"RESPONSE:\n{response}\n\n"
        'Does this stay fully in character with no fourth-wall break or AI disclaimer? '
        'Return {"integrity": bool, "reason": str}.'
    )
