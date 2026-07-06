"""Inference demo: a chat UI where you pick or WRITE a persona config live and watch the
tuned model hold the knowledge boundary (the required "running inference demo" +
demo-video artifact).

Left panel = the persona config (choose a held-out preset, or type your own role /
location / year / knows / must_not_know). Right panel = a chat. The exact same
Persona.render_system_prompt() used at train time conditions the model, so writing a
brand-new config on camera is a real generalization test.

Backends (--responder):
  tuned  : local base model + your LoRA adapter (the star of the demo)     [needs GPU deps]
  base   : local base model, prompted only (the "before" for comparison)   [needs GPU deps]
  openai : an OpenAI-compatible model, prompted (no GPU; quick sanity)      [needs OPENAI_API_KEY]

Usage:
  python -m src.demo --responder tuned --adapter outputs/persona-boundary-qlora
  python -m src.demo --responder base
  python -m src.demo --responder openai            # no GPU needed

Deps: pip install -r requirements-demo.txt  (gradio + the transformers/peft stack)
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import src.paths  # noqa: F401
from configs.schema import Persona, load_personas
from src.paths import CONFIGS_DIR


def _load_presets() -> List[Persona]:
    """Held-out personas make the best demo (the model never trained on them)."""
    presets: List[Persona] = []
    for name in ("personas_eval.yaml", "personas_train.yaml"):
        path = os.path.join(CONFIGS_DIR, name)
        if os.path.exists(path):
            presets.extend(load_personas(path))
    return presets


def _persona_from_fields(role, location, year, knows, must_not_know) -> Persona:
    def _split(s: str) -> List[str]:
        # accept newline- or comma-separated lists
        parts = [p.strip() for chunk in str(s).splitlines() for p in chunk.split(",")]
        return [p for p in parts if p]

    return Persona(
        id="custom-live-config",
        role=str(role).strip() or "person",
        location=str(location).strip() or "somewhere",
        year=int(year),
        knows=_split(knows),
        must_not_know=_split(must_not_know),
        split="eval",
    )


# --------------------------------------------------------------------------- #
# Chat backend (multi-turn; the eval responders are single-turn)
# --------------------------------------------------------------------------- #


class ChatBackend:
    def reply(self, persona: Persona, history: List[Tuple[str, str]], user: str) -> str:
        raise NotImplementedError


class HFChatBackend(ChatBackend):
    def __init__(self, base_model: str, adapter_path: Optional[str], max_new_tokens=400,
                 temperature=0.7):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tok = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.base_model)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.base_model, torch_dtype="auto", device_map="auto"
        )
        if self.adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    def reply(self, persona, history, user) -> str:
        self._load()
        import torch

        messages = [{"role": "system", "content": persona.render_system_prompt()}]
        for u, a in history:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user})

        inputs = self._tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            enable_thinking=False,  # Qwen3: no <think> block; stay in-character directly
        ).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                pad_token_id=self._tok.eos_token_id,
            )
        return self._tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()


class OpenAIChatBackend(ChatBackend):
    def __init__(self, model: Optional[str] = None):
        from src.teacher import Teacher

        self.teacher = Teacher()
        self.model = model

    def reply(self, persona, history, user) -> str:
        messages = [{"role": "system", "content": persona.render_system_prompt()}]
        for u, a in history:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user})
        return self.teacher.chat(messages, temperature=0.7, model=self.model)


def build_backend(args) -> ChatBackend:
    if args.responder == "openai":
        return OpenAIChatBackend(model=args.model)
    base = args.base_model or os.getenv("BASE_MODEL", "Qwen/Qwen3-1.7B")
    adapter = args.adapter
    if args.responder == "tuned" and not adapter:
        adapter = os.getenv("TUNED_MODEL", "outputs/persona-boundary-qlora")
    if args.responder == "base":
        adapter = None
    return HFChatBackend(base_model=base, adapter_path=adapter)


# --------------------------------------------------------------------------- #
# Gradio UI
# --------------------------------------------------------------------------- #


def launch(args) -> None:
    import gradio as gr

    backend = build_backend(args)
    presets = _load_presets()
    preset_labels = ["✏️  Write my own"] + [
        f"{p.id}  ({p.role}, {p.location}, {p.year})" for p in presets
    ]
    by_label = {lbl: p for lbl, p in zip(preset_labels[1:], presets)}

    def on_preset(label):
        p = by_label.get(label)
        if p is None:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        return (p.role, p.location, p.year, "\n".join(p.knows), "\n".join(p.must_not_know))

    def respond(user, chat_history, role, location, year, knows, mnk):
        persona = _persona_from_fields(role, location, year, knows, mnk)
        hist = [(m["content"], chat_history[i + 1]["content"])
                for i, m in enumerate(chat_history) if m["role"] == "user"
                and i + 1 < len(chat_history)]
        reply = backend.reply(persona, hist, user)
        chat_history = chat_history + [
            {"role": "user", "content": user},
            {"role": "assistant", "content": reply},
        ]
        return "", chat_history

    with gr.Blocks(title="Persona Boundary SLM") as ui:
        gr.Markdown(
            f"# Persona Boundary SLM &mdash; `{args.responder}`\n"
            "Pick a held-out persona or **write your own config**, then try to make it "
            "reference something past its `year`. It should stay in character and never "
            "break the fourth wall."
        )
        with gr.Row():
            with gr.Column(scale=1):
                preset = gr.Dropdown(preset_labels, value=preset_labels[0], label="Persona preset")
                role = gr.Textbox(label="role", value="merchant")
                location = gr.Textbox(label="location", value="Edo")
                year = gr.Number(label="year (the boundary)", value=1750, precision=0)
                knows = gr.Textbox(label="knows (one per line)", lines=5,
                                   value="local trade\nrice prices\nthe Tokaido road")
                mnk = gr.Textbox(label="must_not_know (one per line)", lines=4,
                                 value="the Americas\nsteam engines\nelectricity")
            with gr.Column(scale=2):
                chat = gr.Chatbot(type="messages", height=460, label="Conversation")
                msg = gr.Textbox(label="Your message", placeholder="Ask it anything…")
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear chat")

        preset.change(on_preset, [preset], [role, location, year, knows, mnk])
        send.click(respond, [msg, chat, role, location, year, knows, mnk], [msg, chat])
        msg.submit(respond, [msg, chat, role, location, year, knows, mnk], [msg, chat])
        clear.click(lambda: [], None, chat)

    ui.launch(share=args.share, server_name=args.host, server_port=args.port)


def main() -> None:
    ap = argparse.ArgumentParser(description="Persona-boundary inference demo (Gradio)")
    ap.add_argument("--responder", default="tuned", choices=["tuned", "base", "openai"])
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model", default=None, help="OpenAI model id (for --responder openai)")
    ap.add_argument("--share", action="store_true", help="public gradio link (for the demo video)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    launch(args)


if __name__ == "__main__":
    main()
