"""Fill persona seeds' knows / must_not_know via the teacher, then split to train/eval.

Usage:
    python -m src.build_configs                 # real teacher calls (needs OPENAI_API_KEY)
    python -m src.build_configs --dry-run       # offline stub, for wiring tests
    python -m src.build_configs --limit 5       # only fill the first 5 seeds

Outputs:
    configs/personas_train.yaml
    configs/personas_eval.yaml
"""

from __future__ import annotations

import argparse
import os

import src.paths  # noqa: F401  (sys.path bootstrap)
from configs.schema import Persona, load_personas, dump_personas
from src import prompts
from src.paths import CONFIGS_DIR
from src.teacher import Teacher


def fill_persona(teacher: Teacher, p: Persona) -> Persona:
    """Populate knows / must_not_know for a single persona (idempotent-ish)."""
    if p.is_filled():
        return p
    messages = [
        {"role": "system", "content": prompts.CONFIG_BUILDER_SYSTEM},
        {
            "role": "user",
            "content": prompts.config_builder_user(
                p.role, p.location, p.year, p.persona_summary
            ),
        },
    ]
    data = teacher.chat_json(messages, temperature=0.4)
    knows = [str(x).strip() for x in data.get("knows", []) if str(x).strip()]
    traps = [str(x).strip() for x in data.get("must_not_know", []) if str(x).strip()]
    # Drop the generic "anything after year" if the teacher added it anyway.
    traps = [t for t in traps if "after the year" not in t.lower() and "after year" not in t.lower()]
    p.knows = knows
    p.must_not_know = traps
    errs = p.validation_errors()
    if errs:
        # Non-fatal: keep what we have but warn so the operator can inspect.
        print(f"  [warn] {p.id}: {errs}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=os.path.join(CONFIGS_DIR, "persona_seeds.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="use offline teacher stub")
    ap.add_argument("--limit", type=int, default=0, help="only process first N seeds")
    args = ap.parse_args()

    personas = load_personas(args.seeds)
    if args.limit:
        personas = personas[: args.limit]

    teacher = Teacher(dry_run=args.dry_run)
    print(f"Filling {len(personas)} persona configs (dry_run={args.dry_run})...")

    filled = teacher.map(lambda p: fill_persona(teacher, p), personas)
    ok = [p for p in filled if isinstance(p, Persona)]
    for r, p in zip(filled, personas):
        if isinstance(r, Exception):
            print(f"  [error] {p.id}: {r}")

    train = [p for p in ok if p.split == "train"]
    eval_ = [p for p in ok if p.split == "eval"]

    train_path = os.path.join(CONFIGS_DIR, "personas_train.yaml")
    eval_path = os.path.join(CONFIGS_DIR, "personas_eval.yaml")
    dump_personas(train, train_path)
    dump_personas(eval_, eval_path)
    print(f"Wrote {len(train)} -> {train_path}")
    print(f"Wrote {len(eval_)} -> {eval_path}")


if __name__ == "__main__":
    main()
