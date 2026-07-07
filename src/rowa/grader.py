"""Frontier grader for a single Row A decision, model-agnostic across backends.

- ``openai``   : reuses the repo's OpenAI-compatible Teacher (gpt-4o, etc.).
- ``anthropic``: the E1 baseline backend (Claude Sonnet). Lazy-imported so the repo
                 runs without the SDK until this backend is actually used.

Both return a dict: {"point": 0|1, "reason": str}. Temperature is pinned to 0 for
reproducibility; a bad/unparseable completion raises so the caller can retry or drop.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

from src.teacher import Teacher, extract_json


@dataclass
class GraderResult:
    point: int
    reason: str

    def as_dict(self) -> dict:
        return {"point": self.point, "reason": self.reason}


def _coerce(raw_obj) -> GraderResult:
    point = int(raw_obj.get("point"))
    if point not in (0, 1):
        raise ValueError(f"point must be 0 or 1, got {point!r}")
    return GraderResult(point=point, reason=str(raw_obj.get("reason", "")))


class FrontierGrader:
    def __init__(
        self,
        backend: str = "openai",
        model: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.backend = backend
        self.dry_run = dry_run
        if backend == "openai":
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
            self._teacher = Teacher(dry_run=dry_run)
            self._anthropic = None
        elif backend == "anthropic":
            self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
            self._teacher = None
            if not dry_run:
                key = os.getenv("ANTHROPIC_API_KEY")
                if not key:
                    raise RuntimeError(
                        "backend='anthropic' needs ANTHROPIC_API_KEY in .env or env."
                    )
                try:
                    import anthropic
                except ImportError as e:
                    raise RuntimeError(
                        "anthropic SDK not installed. `pip install anthropic` in the venv."
                    ) from e
                self._anthropic = anthropic.Anthropic(api_key=key)
        else:
            raise ValueError(f"unknown backend: {backend!r}")

    def grade(self, system: str, user: str) -> GraderResult:
        if self.dry_run:
            # Deterministic offline stub: passes long theses, denies short ones.
            return GraderResult(point=1 if len(user) > 400 else 0, reason="offline stub")
        if self.backend == "openai":
            raw = self._teacher.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                json_mode=True,
                max_tokens=300,
                model=self.model,
            )
            return _coerce(extract_json(raw))
        # anthropic
        msg = self._anthropic.messages.create(
            model=self.model,
            max_tokens=300,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return _coerce(extract_json(text))


class LocalGrader:
    """The fine-tuned specialist: a local Qwen model + LoRA adapter that emits {point,reason}.

    Same .grade(system, user) interface as FrontierGrader so the eval treats every grader
    uniformly. Greedy decoding for reproducibility.
    """

    def __init__(self, base_model: str, adapter_path: Optional[str] = None,
                 dry_run: bool = False, max_new_tokens: Optional[int] = None):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.dry_run = dry_run
        self.max_new_tokens = max_new_tokens or int(os.getenv("HF_MAX_NEW_TOKENS", "320"))
        self.model = f"{base_model}+{adapter_path or 'base'}"
        self._m = None
        self._tok = None

    def _load(self):
        if self._m is not None or self.dry_run:
            return
        import torch  # noqa
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.base_model)
        self._m = AutoModelForCausalLM.from_pretrained(
            self.base_model, dtype="auto", device_map="auto"
        )
        if self.adapter_path:
            from peft import PeftModel

            self._m = PeftModel.from_pretrained(self._m, self.adapter_path)
        self._m.eval()

    def grade(self, system: str, user: str) -> GraderResult:
        if self.dry_run:
            return GraderResult(point=1 if len(user) > 400 else 0, reason="offline stub")
        self._load()
        import torch

        enc = self._tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
            enable_thinking=False,
        )
        enc = {k: v.to(self._m.device) for k, v in enc.items()}
        n = enc["input_ids"].shape[-1]
        with torch.no_grad():
            out = self._m.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self._tok.eos_token_id,
            )
        text = self._tok.decode(out[0][n:], skip_special_tokens=True).strip()
        try:
            return _coerce(extract_json(text))
        except Exception:
            # Salvage the decision from a truncated JSON (reason cut off): the point is
            # all that's scored.
            m = re.search(r'"point"\s*:\s*([01])', text)
            if m:
                return GraderResult(point=int(m.group(1)), reason=text[:200])
            raise
