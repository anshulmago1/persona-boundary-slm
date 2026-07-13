# AP World History — Row A Thesis-Point Specialist

A small open model (**Qwen3-0.6B + LoRA**) fine-tuned to grade **one** analytic rubric
point — the AP World History LEQ/DBQ **Row A thesis/claim** — better than a prompted
frontier model.

## Published artifacts (Hugging Face)

- **Model:** https://huggingface.co/anshulmago1/ap-rowa-thesis-grader-qwen3-0.6b
- **Dataset:** https://huggingface.co/datasets/anshulmago1/ap-rowa-thesis-grading

## The spiky POV

The industry default is "use the most capable model to grade." But grading a single
analytic rubric point is a **constraint** problem, not a **capability** problem. A prompted
frontier model *substitutes its own essay-quality bar*: it silently imports Row-D
("evaluate the extent / broader analytical framework") criteria onto what is actually a
binary Row-A decision, and so **false-denies minimal-but-defensible theses that real AP
readers credit**. Fine-tuning a small model on the literal Row A criterion fixes this.

See [`Brainlift.md`](Brainlift.md) for the full argument + literature.

## Behavior Spec (the falsifiable pass/fail)

> Given an AP World History LEQ/DBQ prompt and a candidate thesis, the model returns a
> single valid JSON object `{"point": 0|1, "reason": "..."}` whose `point` matches the
> official College Board **Row A** decision — awarding a *minimal-but-defensible* thesis
> (a plain claim with any one line of reasoning) and denying restatements, off-topic, or
> non-defensible claims — **without** importing higher-row criteria ("evaluate the extent,"
> analytical complexity). A stranger can mark any output pass/fail against the official label.

This spec is the data-generation rubric, the eval criterion, and the spiky POV at once.

## The evidence (measured, real data)

On **71 real College Board student theses** (scraped from official 2023–2025 scoring
commentaries, each with the reader's official Row A decision), same compact prompt:

| grader | agreement | **κ** | false-deny (56) | false-award (15) |
|---|---|---|---|---|
| base Qwen3-0.6B (untuned) | 79% | 0.10 | 0 | 14 |
| **tuned specialist** | **80%** | **0.54** | 13 | 1 |
| gpt-4o baseline | 69% | 0.38 | 21 | 1 |
| gpt-4o hardened | 75% | 0.39 | 14 | 4 |
| gpt-4o decomposed | 82% | 0.49 | 8 | 5 |

**base→tuned: κ 0.10 → 0.54** (data→behavior held) — the tuned 0.6B also **beats every
prompted gpt-4o** on κ. Separately, the litmus gate: plain gpt-4o denies **~39% of theses
real readers credited**; hardening helps but can't close it. Full table via `src/rowa/eval.py`.

## Pipeline (`src/rowa/`)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # data/eval pipeline (no GPU)
cp .env.example .env                      # add OPENAI_API_KEY (teacher/judge/frontier)
```

```bash
# 1. Scrape official AP Central PDFs (2023-2025 LEQ+DBQ). Raw PDFs stay local (copyright).
python -m src.rowa.scrape

# 2. Parse: typed rubric examples + scoring commentary (official Row A decisions);
#    gpt-4o vision transcribes the handwritten student essays. Cross-checks each label
#    (decision digit vs prose) and flags source contradictions.
python -m src.rowa.parse_pdf --vision

# 3. Unify + dedup into the gold set (held-out real students = the eval slice).
python -m src.rowa.build_gold

# 4. Frontier baseline + litmus gate (gpt-4o baseline vs hardened).
python -m src.rowa.baseline gate

# 5. Synthesize training theses across quality bands; verify labels with the decomposed
#    (bias-resistant) judge; build SFT + DPO with the compact grader prompt.
python -m src.rowa.gen_train --per-band 3 --synth-prompts 8
python -m src.rowa.build_train

# v2: generate near-boundary triplets, then make a prompt-grouped train/dev split.
python -m src.rowa.gen_contrastive --per-prompt 6
python -m src.rowa.build_train --v2

# Offline fallback when the teacher API is unavailable: derive pairs from the existing,
# decomposed-judge-verified claim/reason and category bands.
python -m src.rowa.gen_contrastive --from-existing
python -m src.rowa.build_train --v2 --no-verify

# 6. Fine-tune Qwen3-0.6B (LoRA). --check validates the dataset with no GPU.
#    Runs locally on Apple MPS (fp32 LoRA) or a Colab/CUDA GPU (4-bit QLoRA).
python -m src.train --data data/rowa/train.jsonl --check
python -m src.train --data data/rowa/train.jsonl --base-model Qwen/Qwen3-0.6B \
  --output outputs/rowa-thesis-qlora --epochs 3 --batch-size 1 --grad-accum 16 --max-seq-len 640

# v2 defaults to assistant-only loss and selects the best epoch by held-out dev loss.
python -m src.train --data data/rowa/train_v2.jsonl --eval-data data/rowa/dev_v2.jsonl \
  --base-model Qwen/Qwen3-0.6B --output outputs/rowa-thesis-qlora-v2 \
  --epochs 3 --batch-size 1 --grad-accum 16 --max-seq-len 640

# 7. Headline eval: specialist vs gpt-4o (baseline / hardened / decomposed) on real students.
python -m src.rowa.eval

# Optional calibrated binary scoring. The threshold is learned on dev_v2, never the
# 71-item real-student test set.
python -m src.rowa.eval --graders specialist specialist-calibrated \
  --adapter outputs/rowa-thesis-qlora-v2
```

## Key design decisions

- **Two-way label validation.** Every scraped student label is confirmed by the commentary's
  decision digit *and* its prose; disagreements (a known 2023 LEQ2 2C source error) are dropped.
- **Bias-resistant labeling** ([`judge_rowa.py`](src/rowa/judge_rowa.py)). Synthetic training
  labels are verified by asking only the rubric's *objective sub-questions* (defensible?
  responsive? states a reason/categories? restatement?) and computing the decision — never a
  holistic "award the point?", which would re-import the very bias we're fixing.
- **Compact prompt.** Specialist trains/runs on a ~110-token keyword prompt (Kucia et al. 2026:
  concise prompts beat full rubric-text on analytic scoring); the frontier is still given the
  full hardened rubric.
- **No leakage.** Synthetic training theses are deduped against the real-student eval slice;
  the specialist is never trained on what it's tested on.
- **Contrastive v2 boundary data.** Near-identical pairs differ by adding a reason or
  analytic categories, with an extra verified minimal/clumsy positive per group to avoid
  increasing denial bias. This directly teaches the Row A decision boundary.
- **Assistant-only objective.** System and user tokens are masked from the training loss;
  validation prompts are grouped away from training and select the best checkpoint.
- **Copyright.** Raw College Board PDFs/essays are gitignored and never published; only
  synthetic data, derived labels, and metrics are shareable.

## Demo

```bash
pip install -r requirements-demo.txt
python -m src.rowa.demo                    # specialist vs gpt-4o, side by side
python -m src.rowa.demo --no-frontier      # specialist only (no API key)
```
