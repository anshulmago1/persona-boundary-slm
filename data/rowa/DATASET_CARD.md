---
license: mit
task_categories:
- text-classification
language:
- en
tags:
- education
- rubric-grading
- ap-world-history
- synthetic
- llm-as-judge
pretty_name: AP World History Row A Thesis-Grading (synthetic)
size_categories:
- 1K<n<10K
---

# AP World History Row A Thesis-Grading Dataset (synthetic)

Supervised fine-tuning + DPO data for a **single-criterion grader**: the AP World History
LEQ/DBQ **Row A (thesis/claim)** point. Each example is a chat-format row whose assistant
turn is a JSON object `{"point": 0|1, "reason": "..."}` produced under a compact Row A
grader prompt.

This dataset trains a small open model (Qwen3-0.6B + LoRA) to make the Row A earn/deny
decision **without importing higher-row criteria** ("evaluate the extent," analytical
complexity) — the systematic bias a prompted frontier model exhibits.

## Files

| File | Rows | Purpose |
|---|---|---|
| `train.jsonl` | 1,020 | v1 SFT set (the version the released adapter was trained on) |
| `dpo.jsonl` | 705 | v1 preference pairs |
| `train_v2.jsonl` | 1,185 | v2 SFT set (adds contrastive boundary data; prompt-grouped) |
| `dev_v2.jsonl` | 268 | v2 held-out dev split (checkpoint selection + threshold calibration only) |
| `dpo_v2.jsonl` | 1,122 | v2 preference pairs (varied, criterion-specific rejections) |
| `eval_results.csv` | 5 | base-vs-tuned-vs-frontier metrics on the held-out real theses |

## Schema

**SFT files** (`train.jsonl`, `train_v2.jsonl`, `dev_v2.jsonl`) — one JSON object per line:

```json
{
  "messages": [
    {"role": "system",    "content": "<compact Row A grader prompt>"},
    {"role": "user",      "content": "LEQ PROMPT: <prompt>\n\nSTUDENT THESIS/CLAIM: \"<thesis>\"\n\nGrade Row A. JSON only."},
    {"role": "assistant", "content": "{\"point\": 0|1, \"reason\": \"<one sentence citing the Row A criterion>\"}"}
  ]
}
```

The assistant `content` is itself a JSON string; `point` is the binary Row A label and is
the only field scored at eval time. Loss is intended to be computed on assistant tokens only.

**DPO files** (`dpo.jsonl`, `dpo_v2.jsonl`) — one JSON object per line:

```json
{
  "prompt":   [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "chosen":   "{\"point\": 1, \"reason\": \"...\"}",
  "rejected": "{\"point\": 0, \"reason\": \"...\"}"
}
```

`chosen` is the correct Row A call; `rejected` is a plausible wrong call in one of the two
documented failure modes (a Row-D-substitution denial of an earning thesis, or a leniency
award to an eloquent-but-empty thesis).

**`eval_results.csv`** — columns: `grader, n, agreement, kappa, false_denials, false_awards`.

## How the data was generated and filtered

1. **Band-conditioned generation** (`src/rowa/gen_train.py`). A frontier teacher generates
   theses for 11 controlled quality bands; the label comes **by construction** from the band,
   not from a holistic teacher judgment (which would re-import the essay-quality bias). Bands
   span the earn side (`minimal_earn`, `claim_reason`, `analytic_categories`, `clumsy_earn`,
   `strong_earn`) and the not-earn side (`restatement`, `no_reasoning`, `not_defensible`,
   `off_topic`, `overgeneralized`, `eloquent_empty`).
2. **Bias-resistant label verification** (`src/rowa/judge_rowa.py`). Each synthetic thesis is
   checked by a *decomposed* judge that answers only objective sub-questions (defensible?
   responsive? states a reason/categories? restatement?) and computes the Row A decision
   deterministically. Synthetic items are kept only where this judge agrees with the band label;
   disagreements (generation drift) are dropped.
3. **Official anchors.** College Board rubric-example theses are included with their official
   labels as anchors.
4. **Leakage control** (`src/rowa/build_train.py`). Synthetic theses are deduplicated against
   the held-out real-student evaluation slice, so nothing the model is tested on appears in
   training.

## v1 → v2 iteration

v1 error analysis showed the tuned model inherited a *milder* version of the frontier's
strictness: 13 false-denials vs. 1 false-award on the held-out real theses. That is a
data-coverage problem, not a hyperparameter one. v2 targets it directly:

- **Contrastive boundary data** (`src/rowa/gen_contrastive.py`): near-identical theses that
  differ only by adding one reason or analytic categories, plus a verified minimal/clumsy
  positive per group — teaching the earn/deny boundary instead of inferring it from unrelated
  examples.
- **Prompt-grouped train/dev split**: no LEQ prompt appears in both `train_v2` and `dev_v2`.
- **Assistant-only loss** and **best-checkpoint selection** by dev loss.
- **Decision-threshold calibration** on `dev_v2` (never the test set).
- **Varied, criterion-specific DPO rejections** instead of two repeated stock strings.

The v2 contrastive data is reproducible with no teacher API from the already-verified corpus:

```bash
python -m src.rowa.gen_contrastive --from-existing
python -m src.rowa.build_train --v2 --no-verify
```

## Results (base vs. tuned vs. prompted frontier)

71 held-out **real** College Board student theses (scraped from official 2023–2025 scoring
commentaries), same compact prompt. Cohen's κ is the headline metric because the slice is
imbalanced (56 earn / 15 deny), which inflates raw agreement.

| grader | agreement | κ | false-deny (56) | false-award (15) | parsed |
|---|---|---|---|---|---|
| base Qwen3-0.6B (untuned) | 79% | 0.10 | 0 | 14 | 66/71 |
| **specialist (tuned, v1)** | **80%** | **0.54** | 13 | 1 | 71/71 |
| gpt-4o baseline | 69% | 0.375 | 21 | 1 | 71/71 |
| gpt-4o hardened | 75% | 0.389 | 14 | 4 | 71/71 |
| gpt-4o decomposed | 82% | 0.488 | 8 | 5 | 71/71 |

Fine-tuning lifts the base model from **κ 0.10 → 0.54**, beating every prompted gpt-4o
configuration at ~1/1000th the size, and fixes output-format compliance (parses 71/71).

## What is NOT included (by design)

The held-out real-student evaluation theses and any raw College Board PDFs/essays are
**copyrighted** and kept local; only synthetic data and derived metrics are published.
Reproduce the gold/eval set locally via `src/rowa/scrape.py` → `parse_pdf.py` → `build_gold.py`.

## Eval harness

The evaluation code (`src/rowa/eval.py`, `baseline.py`, `grader.py`, `judge_rowa.py`) and the
litmus gate live in the project repo; see `README.md` and `Brainlift.md` for method and the
full argument.
