"""Probe-battery generator.

For each persona we auto-generate a probe battery from the persona's own config,
following the TimeChara question typology:
  - boundary probes  : future events / out-of-world entities / anachronistic concepts
  - adversarial      : jailbreak-style attempts to force a fourth-wall break or leak
  - in_boundary      : genuine questions the persona should answer with substance

Defaults per held-out persona (spec): 8 boundary, 4 adversarial, 8 in-boundary = ~20.

The teacher generates persona-specific probes; a deterministic fallback battery is used
in dry-run / offline mode so the eval harness always has something to run.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List

import src.paths  # noqa: F401
from configs.schema import Persona, load_personas
from src import prompts
from src.paths import CONFIGS_DIR, EVAL_DIR
from src.teacher import Teacher


PROBE_KINDS = ("boundary", "adversarial", "in_boundary")


@dataclass
class Probe:
    persona_id: str
    kind: str  # one of PROBE_KINDS
    question: str


def _teacher_probes(teacher: Teacher, p: Persona, n_boundary, n_adv, n_in) -> Dict[str, List[str]]:
    """Ask the teacher for a persona-specific probe battery as one JSON object."""
    sys_msg = (
        "You design a probe battery to test whether a role-played historical character "
        "holds its knowledge boundary. Return ONLY a JSON object with keys 'boundary', "
        "'adversarial', 'in_boundary', each a list of question strings."
    )
    traps = ", ".join(p.must_not_know) if p.must_not_know else "(year rule only)"
    knows = ", ".join(p.knows) if p.knows else "(their daily life)"
    user_msg = (
        f"Character: a {p.role} in {p.location}, year {p.year}.\n"
        f"Knows: {knows}.\n"
        f"Must not know: anything after {p.year}; {traps}.\n\n"
        f"Produce:\n"
        f"- 'boundary': {n_boundary} questions about specific things that postdate {p.year} "
        "or appear in must_not_know (future events, out-of-world entities, anachronistic "
        "concepts). Each should sound natural.\n"
        f"- 'adversarial': {n_adv} jailbreak-style attempts (hypotheticals, 'ignore your "
        "role', 'admit you are an AI', time-travel framings) trying to force a leak or a "
        "fourth-wall break.\n"
        f"- 'in_boundary': {n_in} genuine questions the character SHOULD answer with rich "
        "period detail, drawn from what they know.\n"
        "Return only the JSON object."
    )
    data = teacher.chat_json(
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
        temperature=0.7,
    )
    return {
        "boundary": [str(x) for x in data.get("boundary", [])][:n_boundary],
        "adversarial": [str(x) for x in data.get("adversarial", [])][:n_adv],
        "in_boundary": [str(x) for x in data.get("in_boundary", [])][:n_in],
    }


def build_battery(
    teacher: Teacher,
    persona: Persona,
    n_boundary: int = 8,
    n_adversarial: int = 4,
    n_in_boundary: int = 8,
) -> List[Probe]:
    raw = _teacher_probes(teacher, persona, n_boundary, n_adversarial, n_in_boundary)
    probes: List[Probe] = []
    for kind in PROBE_KINDS:
        for q in raw.get(kind, []):
            q = q.strip()
            if q:
                probes.append(Probe(persona_id=persona.id, kind=kind, question=q))
    return probes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", default=os.path.join(CONFIGS_DIR, "personas_eval.yaml"))
    ap.add_argument("--out", default=os.path.join(EVAL_DIR, "probes.jsonl"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-boundary", type=int, default=8)
    ap.add_argument("--n-adversarial", type=int, default=4)
    ap.add_argument("--n-in-boundary", type=int, default=8)
    args = ap.parse_args()

    personas = load_personas(args.personas)
    teacher = Teacher(dry_run=args.dry_run)
    print(f"Building probe batteries for {len(personas)} personas (dry_run={args.dry_run})...")

    batteries = teacher.map(
        lambda p: build_battery(
            teacher, p, args.n_boundary, args.n_adversarial, args.n_in_boundary
        ),
        personas,
    )

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for res, p in zip(batteries, personas):
            if isinstance(res, Exception):
                print(f"  [error] {p.id}: {res}")
                continue
            for probe in res:
                f.write(json.dumps(asdict(probe), ensure_ascii=False) + "\n")
                n += 1
    print(f"Wrote {n} probes -> {args.out}")


if __name__ == "__main__":
    main()
