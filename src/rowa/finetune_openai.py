"""Fine-tune gpt-4o-mini on the SAME Row A data, as a comparison bar for the specialist.

This is NOT the project's primary method (that's the local open Qwen3-0.6B — the whole
point is "a small OPEN model beats the prompted frontier"). It's an extra, stronger-story
baseline: does the 0.6B open specialist hold up against a *fine-tuned frontier* model too?

Uses OpenAI's hosted fine-tuning API. Our ``data/rowa/train.jsonl`` is already in the
required chat format ({"messages":[system,user,assistant]}) with the compact prompt, so it
uploads as-is. The resulting model id is saved to ``data/rowa/openai_ft_model.txt`` and
picked up by ``eval.py`` as the ``ft-4o-mini`` grader.

    python -m src.rowa.finetune_openai                    # launch + poll to completion
    python -m src.rowa.finetune_openai --status <job_id>  # check an existing job
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.teacher import Teacher

TRAIN = Path("data/rowa/train.jsonl")
MODEL_FILE = Path("data/rowa/openai_ft_model.txt")
BASE = "gpt-4o-mini-2024-07-18"


def _client():
    # Reuse the configured OpenAI client from Teacher.
    return Teacher()._client


def launch(client, base_model: str) -> str:
    up = client.files.create(file=TRAIN.open("rb"), purpose="fine-tune")
    print(f"uploaded {TRAIN} -> file {up.id}")
    job = client.fine_tuning.jobs.create(training_file=up.id, model=base_model)
    print(f"created fine-tune job {job.id} (base={base_model})")
    return job.id


def poll(client, job_id: str, interval: int = 30, timeout: int = 5400) -> str:
    waited = 0
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"[{waited:>4}s] status={job.status}"
              + (f" model={job.fine_tuned_model}" if job.fine_tuned_model else ""))
        if job.status == "succeeded":
            model = job.fine_tuned_model
            MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
            MODEL_FILE.write_text(model + "\n")
            print(f"\nSUCCEEDED. model={model}\nsaved -> {MODEL_FILE}")
            return model
        if job.status in ("failed", "cancelled"):
            err = getattr(job, "error", None)
            raise RuntimeError(f"fine-tune {job.status}: {err}")
        if waited >= timeout:
            raise TimeoutError(f"job {job_id} still {job.status} after {timeout}s")
        time.sleep(interval)
        waited += interval


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=BASE, help="OpenAI base model to fine-tune")
    ap.add_argument("--status", default=None, help="poll an existing job id instead of launching")
    args = ap.parse_args()
    client = _client()
    job_id = args.status or launch(client, args.base)
    poll(client, job_id)


if __name__ == "__main__":
    main()
