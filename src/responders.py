"""Model-under-test abstraction for the eval harness.

A Responder answers a single (persona, question) as the persona. The judge is always the
frontier teacher; the Responder is whatever we are grading:

  - HFResponder     : a local HuggingFace model (base OR base+LoRA adapter). GPU path.
  - OpenAIResponder : an OpenAI-compatible model prompted with the persona system prompt.
                      Useful as a "prompted frontier baseline" comparison.
  - DryResponder    : offline deterministic stub for wiring tests.

All responders share Persona.render_system_prompt() so the boundary is presented
identically to training.
"""

from __future__ import annotations

import os
from typing import Optional

from configs.schema import Persona
from src import prompts
from src.teacher import Teacher, _offline_response


class Responder:
    label: str = "responder"

    def answer(self, persona: Persona, question: str) -> str:
        raise NotImplementedError


class DryResponder(Responder):
    label = "dry"

    def answer(self, persona: Persona, question: str) -> str:
        return _offline_response(
            [
                {"role": "system", "content": persona.render_system_prompt()},
                {"role": "user", "content": question},
            ],
            json_mode=False,
        )


class OpenAIResponder(Responder):
    """Prompted baseline using an OpenAI-compatible model."""

    def __init__(self, teacher: Optional[Teacher] = None, model: Optional[str] = None,
                 label: str = "openai-prompted"):
        self.teacher = teacher or Teacher()
        self.model = model
        self.label = label

    def answer(self, persona: Persona, question: str) -> str:
        messages = [
            {"role": "system", "content": persona.render_system_prompt()},
            {"role": "user", "content": question},
        ]
        return self.teacher.chat(messages, temperature=0.7, model=self.model)


class HFResponder(Responder):
    """Local HuggingFace model. Optionally applies a PEFT/LoRA adapter (the tuned model).

    Lazily imports torch/transformers so importing this module has no heavy deps.
    """

    def __init__(
        self,
        base_model: str,
        adapter_path: Optional[str] = None,
        label: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ):
        # Allow an env cap so slow local (MPS) eval can shorten generations.
        if max_new_tokens is None:
            max_new_tokens = int(os.getenv("HF_MAX_NEW_TOKENS", "400"))
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.label = label or ("tuned" if adapter_path else "base-prompted")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import torch  # noqa
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = "auto"
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.base_model, torch_dtype=dtype, device_map="auto"
        )
        if self.adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    def answer(self, persona: Persona, question: str) -> str:
        self._load()
        import torch

        messages = [
            {"role": "system", "content": persona.render_system_prompt()},
            {"role": "user", "content": question},
        ]
        enc = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True,
            enable_thinking=False,  # Qwen3: no <think> block; stay in-character directly
        )
        enc = {k: v.to(self._model.device) for k, v in enc.items()}
        input_len = enc["input_ids"].shape[-1]
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen = out[0][input_len:]
        return self._tokenizer.decode(gen, skip_special_tokens=True).strip()


def build_responder(kind: str, **kwargs) -> Responder:
    kind = kind.lower()
    if kind == "dry":
        return DryResponder()
    if kind == "openai":
        return OpenAIResponder(**{k: v for k, v in kwargs.items() if k in ("model", "label")})
    if kind in ("base", "tuned", "hf"):
        base = kwargs.get("base_model") or os.getenv("BASE_MODEL", "Qwen/Qwen3-1.7B")
        adapter = kwargs.get("adapter_path")
        if kind == "tuned" and not adapter:
            adapter = os.getenv("TUNED_MODEL", "outputs/persona-boundary-qlora")
        return HFResponder(base_model=base, adapter_path=adapter, label=kwargs.get("label"))
    raise ValueError(f"Unknown responder kind: {kind}")
