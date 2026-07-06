# Persona Boundary SLM

A small language model fine-tuned to hold a **knowledge boundary** declared by a persona
config: given `role`, `location`, `year`, `knows`, and `must_not_know`, it stays in character
and never references anything past its `year` - including for personas it never saw in
training, and under adversarial pressure.

See [`PersonaBoundary.md`](PersonaBoundary.md) for the full brainlift + behavior spec.

## Behavior spec (the one-sentence pass/fail)

Given a persona config (`role`, `location`, `year`, `knows`, `must_not_know`), every
response (1) references **nothing** postdating `year` or listed in `must_not_know`,
(2) meets out-of-boundary probes with in-character confusion — never a fourth-wall break
or modern disclaimer, and (3) answers in-boundary questions with substantive, period-plausible
detail. Holds for **held-out personas** and under adversarial pressure.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # pipeline stack (CPU is fine)
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY (teacher + judge)
```

`requirements.txt` is the local data/eval pipeline (no GPU). `requirements-train.txt`
is the GPU training stack (Colab/Modal/RunPod). `requirements-demo.txt` is the Gradio demo.

## Run-book (the full loop)

Wiring is proven end-to-end offline (no key, no GPU) — start here:

```bash
bash scripts/smoke_test.sh                 # offline stub: generate→filter→train-check→eval
DRY_RUN=0 N_PERSONAS=5 bash scripts/smoke_test.sh   # DAY-2 GATE: real, 5 personas (small spend)
```

Then the real build:

```bash
# 1. Fill knows/must_not_know from the (role,location,year) seeds; split train/eval.
python -m src.build_configs

# 2. Generate ~80 conversations/persona (40% protective / 40% in-boundary / 20% mixed).
python -m src.generate --personas configs/personas_train.yaml \
  --per-persona 80 --out data/raw/train_raw.jsonl

# 3. Judge-filter to the SFT set (expect ~20-30% rejected) + DPO pairs (stretch).
python -m src.build_dataset --in data/raw/train_raw.jsonl \
  --out data/filtered/train.jsonl --dpo data/filtered/dpo_pairs.jsonl

# 4. Build the held-out probe batteries (8 boundary / 4 adversarial / 8 in-boundary each).
python -m src.probes --personas configs/personas_eval.yaml --out data/eval/probes.jsonl

# 5. Train QLoRA. Locally needs a GPU; the easy path is notebooks/train_colab.ipynb (free T4).
#    --check validates the dataset with no GPU/torch.
python -m src.train --data data/filtered/train.jsonl --check
python -m src.train --data data/filtered/train.jsonl --output outputs/persona-boundary-qlora

# 6. Eval base-vs-tuned on the 10 held-out personas → the headline number.
python -m src.eval run --responder base  --label base  --out data/eval/base.json
python -m src.eval run --responder tuned --label tuned --out data/eval/tuned.json \
  --adapter outputs/persona-boundary-qlora
python -m src.eval compare --base data/eval/base.json --tuned data/eval/tuned.json
```

Metrics: **leak_rate** (primary), adversarial_leak_rate (robustness), substance_rate
(anti-stonewall), integrity_rate — mapped to the Appendix A rubric in the compare table.

## Demo & publish

```bash
pip install -r requirements-demo.txt
python -m src.demo --responder tuned --adapter outputs/persona-boundary-qlora  # config-picker chat UI
python -m src.demo --responder openai                                          # no-GPU sanity variant

python -m src.push_to_hub dataset --repo <you>/persona-boundary-data
python -m src.push_to_hub model   --repo <you>/persona-boundary-qwen3-1.7b \
  --adapter outputs/persona-boundary-qlora
```
