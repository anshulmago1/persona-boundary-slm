"""Publish the two required submission artifacts to the Hugging Face Hub:

  1. the dataset  (configs + filtered SFT conversations + probe batteries)
  2. the model    (the LoRA adapter produced by src/train.py)

Auth: `huggingface-cli login` once, or set HF_TOKEN in the environment.

Usage:
  python -m src.push_to_hub dataset --repo <user>/persona-boundary-data
  python -m src.push_to_hub model   --repo <user>/persona-boundary-qwen3-1.7b \
      --adapter outputs/persona-boundary-qlora
"""

from __future__ import annotations

import argparse
import os

import src.paths  # noqa: F401
from src.paths import CONFIGS_DIR, DATA_DIR


def _api(token):
    from huggingface_hub import HfApi

    return HfApi(token=token or os.getenv("HF_TOKEN"))


def push_dataset(args) -> None:
    """Publish ONLY the fully-synthetic Row A training data + metrics.

    IMPORTANT: never upload data/rowa/gold*.jsonl or data/rowa/pdfs/ — those contain
    verbatim College Board text (copyrighted). Only synthetic SFT/DPO + result tables ship.
    """
    api = _api(args.token)
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True, private=args.private)
    # Fully-synthetic SFT/DPO (v1 + v2) + the metrics table + the card. NEVER the gold /
    # real-student theses or PDFs (copyrighted, gitignored, kept local).
    safe = [
        "train.jsonl", "dpo.jsonl",
        "train_v2.jsonl", "dev_v2.jsonl", "dpo_v2.jsonl",
        "eval_results.csv", "DATASET_CARD.md",
    ]
    rowa = os.path.join(DATA_DIR, "rowa")
    for name in safe:
        fp = os.path.join(rowa, name)
        if os.path.isfile(fp):
            api.upload_file(
                path_or_fileobj=fp,
                path_in_repo=("README.md" if name == "DATASET_CARD.md" else name),
                repo_id=args.repo, repo_type="dataset",
            )
            print(f"[hub] uploaded {name}")
    print(f"[hub] dataset -> https://huggingface.co/datasets/{args.repo}")


def push_model(args) -> None:
    api = _api(args.token)
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)
    api.upload_folder(folder_path=args.adapter, repo_id=args.repo, repo_type="model")
    print(f"[hub] model -> https://huggingface.co/{args.repo}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Push dataset/model to the Hugging Face Hub")
    ap.add_argument("--token", default=None, help="HF token (else uses login / HF_TOKEN)")
    ap.add_argument("--private", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dataset")
    d.add_argument("--repo", required=True)
    d.set_defaults(func=push_dataset)

    m = sub.add_parser("model")
    m.add_argument("--repo", required=True)
    m.add_argument("--adapter", default=os.path.join("outputs", "rowa-thesis-qlora"))
    m.set_defaults(func=push_model)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
