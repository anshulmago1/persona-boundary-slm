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
---

# AP World History Row A Thesis-Grading Dataset (synthetic)

Supervised fine-tuning + DPO data for a **single-criterion grader**: the AP World History
LEQ/DBQ **Row A (thesis/claim)** point. Each example is a chat-format row whose assistant
turn is `{"point": 0|1, "reason": "..."}` under a compact Row A grader prompt.

## Files
- `train.jsonl` — ~1,020 SFT examples (`{"messages":[system,user,assistant]}`).
- `dpo.jsonl` — ~705 preference pairs pitting the correct Row A call against the two
  documented failure modes (Row-D-substitution denial; eloquent-empty award).
- `eval_results.csv` — base-vs-tuned-vs-frontier metrics on held-out real theses.

## Provenance & what is NOT here
Theses are **synthetically generated** across controlled quality bands (minimal-earn,
clumsy-earn, analytic-categories, restatement, not-defensible, eloquent-empty, …) and
**verified by a decomposed, bias-resistant rubric judge**. Official College Board rubric
example theses are included as labeled anchors.

**Not included (by design):** the held-out real-student evaluation theses and any raw
College Board PDFs/essays — those are copyrighted and were kept local. Only synthetic
data and derived metrics are published.

## Why
A prompted frontier model false-denies ~39% of minimal-but-defensible theses real AP
readers credit (it imports higher-row "evaluate the extent" criteria). A small open model
fine-tuned on this data grades Row A reliably — see the model card and `Brainlift.md`.
