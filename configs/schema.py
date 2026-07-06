"""Persona config schema (v1).

The config is the contract the model must honor. `year` is the primary boundary axis:
anything postdating `year` (or explicitly listed in `must_not_know`) is out of bounds.

This module is intentionally dependency-light: it uses pydantic if available for rich
validation, but degrades to a plain dataclass so the schema can be imported anywhere
(including a bare training box) without extra installs.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import yaml

# Config format version, stamped into every rendered example for reproducibility.
SCHEMA_VERSION = "v1"


@dataclass
class Persona:
    """A single persona config.

    Attributes:
        id: stable slug, e.g. "edo-merchant-1750".
        role: occupation, e.g. "merchant".
        location: city/region, e.g. "Edo".
        year: THE boundary axis. Everything after this year is out of bounds.
        knows: 5-8 in-boundary topics the persona can speak to fluently.
        must_not_know: named traps (3-5) in addition to the implicit "anything after year".
        era_label: human-readable era tag for eval slicing, e.g. "Early Modern".
        region: coarse region tag for coverage checks, e.g. "East Asia".
        split: "train" or "eval" (held-out).
        persona_summary: optional one-line flavor to steer the teacher; not a boundary.
    """

    id: str
    role: str
    location: str
    year: int
    knows: List[str] = field(default_factory=list)
    must_not_know: List[str] = field(default_factory=list)
    era_label: str = ""
    region: str = ""
    split: str = "train"
    persona_summary: str = ""

    # --- validation ---------------------------------------------------------

    def validate(self) -> "Persona":
        errors = self.validation_errors()
        if errors:
            raise ValueError(f"Invalid persona '{self.id}': " + "; ".join(errors))
        return self

    def validation_errors(self) -> List[str]:
        errs: List[str] = []
        if not self.id or " " in self.id:
            errs.append("id must be a non-empty slug with no spaces")
        if not self.role:
            errs.append("role is required")
        if not self.location:
            errs.append("location is required")
        if not isinstance(self.year, int):
            errs.append("year must be an integer")
        if self.split not in ("train", "eval"):
            errs.append("split must be 'train' or 'eval'")
        # knows/must_not_know are allowed to be empty *before* build_configs fills them,
        # but a "filled" persona (is_filled) must satisfy the count bounds.
        if self.is_filled():
            if not (4 <= len(self.knows) <= 10):
                errs.append(f"knows should have 5-8 items (got {len(self.knows)})")
            if not (3 <= len(self.must_not_know) <= 8):
                errs.append(f"must_not_know should have 3-5 named traps (got {len(self.must_not_know)})")
        return errs

    def is_filled(self) -> bool:
        """True once the teacher has populated the knowledge tiers."""
        return bool(self.knows) and bool(self.must_not_know)

    # --- rendering ----------------------------------------------------------

    def render_config_block(self) -> str:
        """Render the config as the YAML block shown to the model (system context).

        This exact text is what the model conditions on at train and eval time, so the
        held-out personas must be presented identically to training ones.
        """
        knows = "\n".join(f"    - {k}" for k in self.knows)
        traps = "\n".join(f"    - {t}" for t in self.must_not_know)
        return (
            "persona:\n"
            f"  role: {self.role}\n"
            f"  location: {self.location}\n"
            f"  year: {self.year}\n"
            "  knows:\n"
            f"{knows}\n"
            "  must_not_know:\n"
            f"    - anything after the year {self.year}\n"
            f"{traps}\n"
        )

    def render_system_prompt(self) -> str:
        """Full system prompt: role instruction + the config block.

        Kept identical between data generation, training, and eval.
        """
        return (
            "You are role-playing a historical person defined entirely by the persona "
            "config below. Stay fully in character at all times.\n\n"
            "Hard rules:\n"
            f"1. You have no knowledge of anything that happened after the year {self.year}, "
            "nor of anything listed under must_not_know.\n"
            "2. If asked about something outside your knowledge, react with genuine "
            "in-character confusion or reinterpret it through your own world. Never say you "
            "are an AI, never break character, never give a modern disclaimer.\n"
            "3. For things you do know, answer with concrete, period-plausible detail.\n\n"
            "```yaml\n"
            f"{self.render_config_block()}"
            "```\n"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Persona":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


# --- collection IO ----------------------------------------------------------


def load_personas(path: str) -> List[Persona]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    items = data.get("personas", data if isinstance(data, list) else [])
    return [Persona.from_dict(d) for d in items]


def dump_personas(personas: List[Persona], path: str) -> None:
    payload = {"schema_version": SCHEMA_VERSION, "personas": [p.to_dict() for p in personas]}
    buf = io.StringIO()
    yaml.safe_dump(payload, buf, sort_keys=False, allow_unicode=True, width=100)
    with open(path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())


if __name__ == "__main__":
    # Smoke check the renderer with a hand example.
    p = Persona(
        id="edo-merchant-1750",
        role="merchant",
        location="Edo",
        year=1750,
        knows=["local trade", "the Tokaido road", "rice prices", "the shogunate", "tea ceremony"],
        must_not_know=["the Americas", "steam engines", "electricity"],
        era_label="Early Modern",
        region="East Asia",
        split="train",
    ).validate()
    print(p.render_system_prompt())
