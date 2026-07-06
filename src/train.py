"""QLoRA / LoRA supervised fine-tuner for the persona-boundary model.

Consumes chat-format JSONL ({"messages": [system, user, assistant, ...]}) as produced
by src/build_dataset.py, and trains a LoRA adapter on a small open base model.

Portable by design (the assignment runs on whatever GPU you can get):
  - CUDA  -> 4-bit QLoRA. Uses Unsloth if installed (~2x faster / ~70% less VRAM),
             else falls back to TRL + PEFT + bitsandbytes.
  - MPS    -> plain LoRA in bf16 (bitsandbytes/4-bit is CUDA-only). Use a small base
             (e.g. Qwen3-0.6B) for an Apple-Silicon experiment run.
  - CPU    -> plain LoRA, fp32. Fine for tiny sanity runs only; a real run is far too slow.

`--check` validates the dataset and prints stats with NO heavy deps (no torch/transformers
required), so the offline smoke test can exercise this step in the plain pipeline venv.

Usage:
  python -m src.train --data data/filtered/train.jsonl --check          # validate only
  python -m src.train --data data/filtered/train.jsonl                   # SFT (needs a GPU)
  python -m src.train --data data/filtered/train.jsonl \
      --base-model Qwen/Qwen3-0.6B --output outputs/persona-boundary-qlora
  python -m src.train --dpo --dpo-data data/filtered/dpo_pairs.jsonl \
      --adapter outputs/persona-boundary-qlora --output outputs/persona-boundary-dpo
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import src.paths  # noqa: F401  (sys.path bootstrap + ensures dirs exist)
from src.paths import FILTERED_DIR

DEFAULT_BASE = os.getenv("BASE_MODEL", "Qwen/Qwen3-1.7B")
DEFAULT_OUTPUT = os.getenv("TUNED_MODEL", os.path.join("outputs", "persona-boundary-qlora"))

# LoRA target modules for Qwen/Llama-family attention + MLP projections.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# --------------------------------------------------------------------------- #
# Dataset IO + validation (dependency-light: used by --check)
# --------------------------------------------------------------------------- #

VALID_ROLES = {"system", "user", "assistant"}


def read_chat_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: invalid JSON ({e})") from e
    return rows


def validate_rows(rows: List[Dict]) -> Dict:
    """Structural check of {"messages":[...]} SFT rows. Returns a stats dict; raises on
    the first fatal shape error so a broken dataset never silently trains."""
    if not rows:
        raise ValueError("dataset is empty")

    n_turns: List[int] = []
    n_chars: List[int] = []
    n_assistant = 0
    for idx, row in enumerate(rows):
        msgs = row.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise ValueError(f"row {idx}: missing/empty 'messages' list")
        has_assistant = False
        for m in msgs:
            if m.get("role") not in VALID_ROLES:
                raise ValueError(f"row {idx}: bad role {m.get('role')!r}")
            if not str(m.get("content", "")).strip():
                raise ValueError(f"row {idx}: empty content in a {m.get('role')} turn")
            if m["role"] == "assistant":
                has_assistant = True
        if not has_assistant:
            raise ValueError(f"row {idx}: no assistant turn to learn from")
        n_assistant += sum(1 for m in msgs if m["role"] == "assistant")
        n_turns.append(len(msgs))
        n_chars.append(sum(len(m.get("content", "")) for m in msgs))

    return {
        "n_examples": len(rows),
        "n_assistant_turns": n_assistant,
        "avg_turns": round(sum(n_turns) / len(n_turns), 2),
        "max_turns": max(n_turns),
        "avg_chars": round(sum(n_chars) / len(n_chars), 1),
        "max_chars": max(n_chars),
    }


def run_check(data_path: str) -> None:
    rows = read_chat_jsonl(data_path)
    stats = validate_rows(rows)
    print(f"[train --check] dataset OK: {data_path}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


# --------------------------------------------------------------------------- #
# Device / backend selection
# --------------------------------------------------------------------------- #


def detect_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #


def _render_texts(rows: List[Dict], tokenizer) -> List[str]:
    """Render each conversation to a single training string via the model's chat template
    (identical to how it is presented at inference by src/responders.py)."""
    return [
        tokenizer.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        for r in rows
    ]


def train_sft(args) -> None:
    rows = read_chat_jsonl(args.data)
    validate_rows(rows)
    device = detect_device()
    print(f"[train] SFT on {len(rows)} examples | base={args.base_model} | device={device}")

    # -- Fast path: Unsloth (CUDA only). Fall back to plain TRL+PEFT on ANY failure
    #    (missing install, or version/env incompatibility) so a run always completes. --
    if device == "cuda" and not args.no_unsloth:
        try:
            _train_sft_unsloth(args, rows)
            return
        except Exception as e:  # noqa: BLE001 - fall back rather than abort the run
            print(f"[train] Unsloth path failed ({type(e).__name__}: {e}); "
                  "falling back to TRL + PEFT + bitsandbytes.")

    _train_sft_trl(args, rows, device)


def _train_sft_unsloth(args, rows: List[Dict]) -> None:
    from unsloth import FastLanguageModel
    from datasets import Dataset

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=LORA_TARGETS,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    ds = Dataset.from_dict({"text": _render_texts(rows, tokenizer)})
    trainer = _make_sft_trainer(model, tokenizer, ds, args, bf16=True, peft_config=None)
    trainer.train()
    _save(model, tokenizer, args.output)


def _train_sft_trl(args, rows: List[Dict], device: str) -> None:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict = {}
    if device == "cuda":
        # 4-bit QLoRA via bitsandbytes.
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        # MPS / CPU: no bitsandbytes. Plain LoRA; bf16 on MPS, fp32 on CPU.
        model_kwargs["torch_dtype"] = torch.bfloat16 if device == "mps" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )
    ds = Dataset.from_dict({"text": _render_texts(rows, tokenizer)})
    trainer = _make_sft_trainer(
        model, tokenizer, ds, args, bf16=(device == "cuda"), peft_config=peft_config
    )
    trainer.train()
    _save(trainer.model, tokenizer, args.output)


def _make_sft_trainer(model, tokenizer, ds, args, bf16: bool, peft_config=None):
    """Build an SFTTrainer that works across TRL versions.

    TRL renames arguments between releases (tokenizer->processing_class,
    max_seq_length->max_length, and dataset_text_field moved onto SFTConfig). Rather than
    pin a version, inspect the installed signatures and pass only what they accept."""
    import inspect
    from trl import SFTTrainer, SFTConfig

    cfg_params = set(inspect.signature(SFTConfig.__init__).parameters)
    cfg_kw = {
        "output_dir": args.output,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "save_strategy": "epoch",
        "bf16": bf16,
        "report_to": "none",
        "seed": args.seed,
    }
    if "dataset_text_field" in cfg_params:
        cfg_kw["dataset_text_field"] = "text"
    # sequence-length cap: renamed max_seq_length -> max_length across TRL versions
    if "max_seq_length" in cfg_params:
        cfg_kw["max_seq_length"] = args.max_seq_len
    elif "max_length" in cfg_params:
        cfg_kw["max_length"] = args.max_seq_len
    config = SFTConfig(**{k: v for k, v in cfg_kw.items() if k in cfg_params})

    trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    tr_kw = {"model": model, "train_dataset": ds, "args": config}
    if peft_config is not None and "peft_config" in trainer_params:
        tr_kw["peft_config"] = peft_config
    # tokenizer arg renamed to processing_class in newer TRL
    if "processing_class" in trainer_params:
        tr_kw["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        tr_kw["tokenizer"] = tokenizer
    return SFTTrainer(**tr_kw)


# --------------------------------------------------------------------------- #
# DPO (stretch rung 1: preference tuning on top of the SFT adapter)
# --------------------------------------------------------------------------- #


def train_dpo(args) -> None:
    """DPO on {prompt(messages), chosen, rejected} pairs from build_dataset.py --dpo."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel
    from trl import DPOTrainer, DPOConfig

    rows = _read_jsonl(args.dpo_data)
    device = detect_device()
    print(f"[train] DPO on {len(rows)} pairs | base={args.base_model} | device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _fmt(r: Dict) -> Dict:
        prompt = tokenizer.apply_chat_template(
            r["prompt"], tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "chosen": r["chosen"], "rejected": r["rejected"]}

    ds = Dataset.from_list([_fmt(r) for r in rows])

    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
    if args.adapter:  # start DPO from the SFT adapter
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)

    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            beta=0.1,
            logging_steps=10,
            bf16=(device == "cuda"),
            report_to="none",
            seed=args.seed,
        ),
        train_dataset=ds,
        tokenizer=tokenizer,
        peft_config=(
            None
            if args.adapter
            else LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGETS,
            )
        ),
    )
    trainer.train()
    _save(trainer.model, tokenizer, args.output)


def _read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #


def _save(model, tokenizer, output: str) -> None:
    os.makedirs(output, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    print(f"[train] saved adapter -> {output}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description="QLoRA/LoRA SFT (+DPO) for persona-boundary model")
    ap.add_argument("--data", default=os.path.join(FILTERED_DIR, "train.jsonl"),
                    help="chat-format SFT JSONL")
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--check", action="store_true",
                    help="validate the dataset and exit (no torch needed)")
    ap.add_argument("--no-unsloth", action="store_true",
                    help="skip the Unsloth fast path; use plain TRL+PEFT+bitsandbytes QLoRA")

    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--dpo", action="store_true", help="run DPO instead of SFT")
    ap.add_argument("--dpo-data", default=os.path.join(FILTERED_DIR, "dpo_pairs.jsonl"))
    ap.add_argument("--adapter", default=None, help="SFT adapter to start DPO from")

    args = ap.parse_args()

    if args.check:
        run_check(args.data)
        return
    if args.dpo:
        train_dpo(args)
        return
    train_sft(args)


if __name__ == "__main__":
    main()
