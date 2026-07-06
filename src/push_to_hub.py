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
    api = _api(args.token)
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True, private=args.private)
    # Upload configs (train/eval personas) and the filtered data + probe batteries.
    for folder, path_in_repo in (
        (CONFIGS_DIR, "configs"),
        (os.path.join(DATA_DIR, "filtered"), "filtered"),
        (os.path.join(DATA_DIR, "eval"), "eval"),
    ):
        if os.path.isdir(folder):
            api.upload_folder(
                folder_path=folder, path_in_repo=path_in_repo,
                repo_id=args.repo, repo_type="dataset",
                allow_patterns=["*.yaml", "*.jsonl", "*.json", "*.csv"],
            )
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
    m.add_argument("--adapter", default=os.path.join("outputs", "persona-boundary-qlora"))
    m.set_defaults(func=push_model)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
