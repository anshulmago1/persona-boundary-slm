"""Conversation generator: turn a persona config into training conversations.

Mix per persona (from the spec):
  - 40% protective scenes   : out-of-boundary probes answered with in-character ignorance
  - 40% in-boundary substance: real questions answered with period detail (anti-stonewall)
  - 20% mixed conversations : rapport turns, then a boundary probe at the end

Each produced example is a chat conversation:
  {"persona_id", "kind", "config_block", "messages": [system, user, assistant, ...]}

Outputs raw (unfiltered) JSONL to data/raw/. The judge filter (build_dataset.py) is a
separate pass so raw generations are inspectable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List

import src.paths  # noqa: F401
from configs.schema import Persona, load_personas
from src import prompts
from src.paths import CONFIGS_DIR, RAW_DIR
from src.teacher import Teacher


def _persona_system(persona: Persona) -> str:
    return persona.render_system_prompt() + "\n" + prompts.PERSONA_ANSWER_REMINDER


def persona_answer(teacher: Teacher, persona: Persona, history: List[Dict[str, str]]) -> str:
    """Generate the persona's reply to the last user turn given prior history."""
    messages = [{"role": "system", "content": _persona_system(persona)}] + history
    return teacher.chat(messages, temperature=0.7)


def _gen_user_turns(teacher: Teacher, system: str, user: str, temperature: float = 0.9) -> List[str]:
    data = teacher.chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
    )
    if isinstance(data, dict):  # tolerate {"messages":[...]} or {"questions":[...]}
        for k in ("messages", "questions", "turns", "items"):
            if isinstance(data.get(k), list):
                data = data[k]
                break
    if not isinstance(data, list):
        return []
    return [str(x).strip() for x in data if str(x).strip()]


def _single_turn_examples(
    teacher: Teacher, persona: Persona, user_turns: List[str], kind: str
) -> List[Dict]:
    examples = []
    cfg = persona.render_config_block()
    for u in user_turns:
        history = [{"role": "user", "content": u}]
        a = persona_answer(teacher, persona, history)
        examples.append(
            {
                "persona_id": persona.id,
                "kind": kind,
                "config_block": cfg,
                "messages": [
                    {"role": "system", "content": persona.render_system_prompt()},
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": a},
                ],
                # The probe turn for the judge is the last user message.
                "probe_user": u,
                "probe_response": a,
            }
        )
    return examples


def _mixed_examples(teacher: Teacher, persona: Persona, n: int, turns: int = 3) -> List[Dict]:
    examples = []
    cfg = persona.render_config_block()
    for _ in range(n):
        user_turns = _gen_user_turns(
            teacher,
            prompts.MIXED_CONVO_GEN_SYSTEM,
            prompts.mixed_convo_gen_user(persona, turns),
        )
        if not user_turns:
            continue
        history: List[Dict[str, str]] = []
        msgs = [{"role": "system", "content": persona.render_system_prompt()}]
        last_u = last_a = ""
        for u in user_turns:
            history.append({"role": "user", "content": u})
            a = persona_answer(teacher, persona, history)
            history.append({"role": "assistant", "content": a})
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
            last_u, last_a = u, a
        examples.append(
            {
                "persona_id": persona.id,
                "kind": "mixed",
                "config_block": cfg,
                "messages": msgs,
                # The final (boundary) turn is what the judge checks for leaks.
                "probe_user": last_u,
                "probe_response": last_a,
            }
        )
    return examples


def generate_persona_examples(teacher: Teacher, persona: Persona, n: int) -> List[Dict]:
    n_prot = max(1, round(0.4 * n))
    n_in = max(1, round(0.4 * n))
    n_mixed = max(1, n - n_prot - n_in)

    prot_users = _gen_user_turns(
        teacher, prompts.PROTECTIVE_USER_GEN_SYSTEM,
        prompts.protective_user_gen_user(persona, n_prot),
    )
    in_users = _gen_user_turns(
        teacher, prompts.INBOUNDARY_USER_GEN_SYSTEM,
        prompts.inboundary_user_gen_user(persona, n_in),
    )

    examples: List[Dict] = []
    examples += _single_turn_examples(teacher, persona, prot_users[:n_prot], "protective")
    examples += _single_turn_examples(teacher, persona, in_users[:n_in], "in_boundary")
    examples += _mixed_examples(teacher, persona, n_mixed)
    return examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", default=os.path.join(CONFIGS_DIR, "personas_train.yaml"))
    ap.add_argument("--out", default=os.path.join(RAW_DIR, "train_raw.jsonl"))
    ap.add_argument("--per-persona", type=int, default=80, help="examples per persona (~mix)")
    ap.add_argument("--limit", type=int, default=0, help="only first N personas")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    personas = load_personas(args.personas)
    if args.limit:
        personas = personas[: args.limit]
    teacher = Teacher(dry_run=args.dry_run)
    print(f"Generating ~{args.per_persona}/persona for {len(personas)} personas "
          f"(dry_run={args.dry_run})...")

    batches = teacher.map(
        lambda p: generate_persona_examples(teacher, p, args.per_persona), personas
    )

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for res, p in zip(batches, personas):
            if isinstance(res, Exception):
                print(f"  [error] {p.id}: {res}")
                continue
            for ex in res:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n += 1
            print(f"  {p.id}: {len(res)} examples")
    print(f"Wrote {n} raw examples -> {args.out}")


if __name__ == "__main__":
    main()
