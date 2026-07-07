Spiky POV: Frontier models are unreliable graders of analytic rubric points because they grade for essay sophistication, the behavior they are trained to reward, rather of the narrow criterion the rubric actually requires. On the AP World History LEQ thesis point (Row A), a well-prompted frontier model denies credit to minimal-but-defensible theses that AP readers accept, because it silently imports higher-order “evaluate the extent/broader analytical framework” responses instead. A small open model fine-tuned on the criteria for the WHAP LEQ will perform better than the frontier model.
Elaboration: The industry framing is “use the most capable model to grade.” But grading a single analytic rubric point is not a capability problem — it is a constraint problem, and constraint is exactly where prompting is unreliable and data is the lever. My own baseline shows a frontier model (Claude Sonnet) inverting an official College Board ruling by denying a thesis for lacking “extent” evaluation and a “broader analytical framework” — Row D (Analysis/Complexity) language misapplied to a Row A binary decision (Insight 1; E1). The literature confirms this is not idiosyncratic: LLM-judge alignment is strong on binary tasks but degrades as rubrics become granular and judgment-laden (Deng et al., 2025); holistic scoring holds at QWK ≈ 0.6 while analytic single-criterion scoring does not transfer (Kucia et al., 2026); and the 2026 FairJudge taxonomy names “applying the wrong rubric” as a distinct failure, measured at >50% error on production bias tests (Yang et al., 2026). Crucially, the failure is a substitution — the model imports its own quality bar — which is why it reads as strict on a humanities thesis but lenient on partial-credit science (Insight 2). And it is not reliably promptable: length bias survives direct instruction (nature Sci Rep, 2025), reflective rubric-refinement degrades alignment (Springer TKL, 2026), and self-preference bias persists on programmatically verifiable rubrics (Self-Preference Bias, 2026). Meanwhile the specialist win is one of the most replicated results in AES: fine-tuned small/open models beat prompted GPT-4 on essay scoring across labs and years (Xiao et al., 2024; Wang & Gayed, 2024; AiAWE, 2026).
Frontier Grading of Analytic Rubric Points
DOK 1: Facts
Raw findings, one source at a time.
• Primary baseline (E1): Claude Sonnet scored 24 thesis statements under the official AP World History LEQ Row A rubric — 16 verbatim from College Board scoring commentaries (2024 LEQ2 Set 1; 2025 LEQ3 Set 1), 8 constructed bias probes. Stable across re-runs: 81% agreement with official decisions (13/16 gold); 0 false awards, 3 false denials. All three misses were minimal-earn/clumsy-earn items; each denial cited criteria absent from Row A (e.g. failure to “evaluate the extent” or supply a “broader analytical framework”). On official item 3C, the College Board reader discounted the very flaw Sonnet fixated on (“treated as a read-through error”) and awarded the point (this project’s tester, 2026).
• Instruction-tuned LLMs reach QWK ≈ 0.6 on holistic essay scoring, but this does not transfer to analytic scoring, where directional trait bias is “large and stable”; concise keyword prompts generally outperform full rubric-text prompts on analytic scoring (Kucia et al., 2026, arXiv 2604.00259).
• Rubric-conditioned LLM grading (Qwen 2.5-72B, SciEntsBank) is “strong for binary tasks but degrades with increased rubric granularity”; the model is generally more lenient than humans, over-classifying “Partially Correct” as “Correct” (Deng et al., 2025, arXiv 2601.08843).
• Foundational LLM-as-judge work put GPT-4–human agreement above 80%, matching human–human agreement (Zheng et al., 2023); 2025–26 follow-up found frontier models exceeding 50% error on production bias tests. FairJudge names “applying the same rubric across mismatched tasks” — measuring “something other than what the task requires” — as a distinct failure mode (Yang et al., 2026, via Adaline synthesis).
• On IFEval (programmatically verifiable rubrics), judges are up to 50% more likely to wrongly mark a failed rubric as satisfied when the output is their own; ensembling mitigates but does not eliminate it (Self-Preference Bias, 2026, arXiv 2604.06996).
• On handwritten-math rubric-item grading, Gemini-3-flash reached 89–99% item accuracy; disagreements were dominated by transcription failures, not rubric misapplication (Bao et al., 2026, arXiv 2605.19043). (Counter-evidence / scope limiter.)

DOK 2: Summary
The structural finding: frontier LLMs are reliable rubric graders exactly where the criterion is crisp and checkable, and unreliable where it is narrow but judgment-laden. Holistic essay quality — the task they are optimized for — they do adequately (QWK ≈ 0.6) (Kucia et al., 2026). Binary checkable items they do excellently (Bao et al., 2026). But a single analytic point that competes with their quality prior — the AP thesis point — is where they diverge from official human decisions, and the divergence is a systematic substitution of essay-sophistication criteria for the literal rubric (E1; Yang et al., 2026; Deng et al., 2025). The direction of error is not fixed (strict on humanities theses, lenient on partial-credit science) because the mechanism is substitution of the model’s own bar, not a constant leniency/strictness offset (E1 vs. Deng et al., 2025). This is the gap the specialist targets, and its narrowness is a feature: a well-defined, reproducible, in-domain failure with programmatically constructible ground truth.
￼
Small/Fine-Tuned Models vs. Prompted Frontier
DOK 1: Facts
• On ASAP, fine-tuned GPT-3.5 scored QWK 0.613–0.859 across essay sets while GPT-4 with few-shot scored 0.257–0.784 — the prompted frontier loses on every set (Xiao et al., 2024, via arXiv 2407.05733). Same survey: BERT avg QWK 0.421 vs. GPT-3.5 zero/few-shot 0.336–0.385 on DREsS (Han et al., 2023); ASAP SOTA 0.544–0.771 vs. GPT-3.5-turbo/Llama2 0.023–0.327 (Mansour et al., 2024).
• Fine-tuned GPT-3.5 on TOEFL writing reached RMSE 0.57 / QWK 0.78, “substantially outperforming both zero-shot GPT-3.5 and the more capable GPT-4,” and fine-tuned graders “do not require a large variety of essay prompts to generalize” (Wang & Gayed, 2024, via AiAWE 2026).
• Fine-tuned open Gemma-3-27B reached RMSE 0.474 / QWK 0.828 / 90.56% within ±0.5 of human, surpassing the fine-tuned GPT-3.5 figures; two fine-tuned models both hit QWK 0.81 on TOEFL11, beating all non-fine-tuned variants (AiAWE, 2026, arXiv 2606.12801; Liu et al., 2025).
• Even zero-shot, a small open model beat proprietary: Llama2-13B-chat achieved QWK 0.437 (TOEFL11) / 0.355 (ASAP), outperforming ChatGPT (Springer Discover AI, 2026).
• A small model fine-tuned on GPT-extracted rationales improved 11% over ChatGPT-alone (→64.36 QWK) on short-answer grading (LAK 2025).

DOK 2: Summary
The specialist-beats-prompted-frontier result is the most replicated of the three claims — it holds across labs, datasets, years, and (critically) on open weights (Xiao et al., 2024; Wang & Gayed, 2024; AiAWE, 2026; Springer, 2026; LAK 2025). Two facts are load-bearing for this project specifically: fine-tuned graders generalize without prompt diversity (Wang & Gayed, 2024) — so by-construction data on a single rubric row is viable — and the win holds on the exact open-model class targeted for deployment (AiAWE, 2026). The honest boundary: the win is on constrained behavior, not capability. Calibration examples can lift the frontier substantially too (Yancey et al., 2023), so the defensible claim is the assignment’s own — reliable, cheap, local, correct-every-time on one criterion — never “smarter than GPT.”
￼
Promptability of the Gap (Litmus-Test Defense)
DOK 1: Facts
• GPT-4’s length bias is robust to prompting: explicit instructions to ignore length left it “virtually unchanged”; a two-step point-by-point rubric-checking prompt only halved it (Nature Scientific Reports, 2025).
• One cycle of reflective rubric-refinement + self-analysis “did not yield improvements and, in most cases, degraded performance,” compressing score distributions and adding instability on GPT-4o/GPT-5 (Springer TKL, 2026).
• Four incrementally improved prompts gave “no consistent results” across tasks/models (Mansour et al., 2024); rubric-linked feedback did help GPT-3.5 (Han et al., 2023); one calibration example per category got GPT-4 to QWK ≈ 0.81, near a strong baseline but below humans (Yancey et al., 2023).
• Self-preference bias persists even on programmatically verifiable rubrics; ensembling mitigates but does not eliminate it (Self-Preference Bias, 2026).

DOK 2: Summary
The precise, defensible claim is “prompting does not reliably close the gap,” not “prompting does nothing” — the sources cut both ways and overstating invites rebuttal. The strong-side evidence: the most-documented grading bias survives direct instruction and is only halved by structured prompting (Nature Sci Rep, 2025); iterative self-refinement makes alignment worse (Springer TKL, 2026); and bias survives even objectively checkable criteria (Self-Preference Bias, 2026). The risk-side evidence: prompting demonstrably helps in some settings (Han et al., 2023; Yancey et al., 2023), and shorter keyword prompts beat rubric-text on analytic scoring (Kucia et al., 2026) — meaning a better-targeted prompt might reduce the observed Sonnet strictness. This is why the hardened-prompt condition is a required experiment, not an optional one: re-run the same 24 items with a system prompt that explicitly forbids the Row-D import (“do NOT require evaluation of extent; a bare defensible claim with any line of reasoning earns”). Sonnet still denies the minimal theses → litmus test passed, fine-tuning justified. Hardened prompt fixes it → pivot up the rubric (contextualization or full-LEQ analytic score), where the gap is not reliably promptable (Kucia et al., 2026; Nature Sci Rep, 2025).
￼
Experts
• Lianmin Zheng et al. — LLM-as-a-judge (MT-Bench). Why: established the 80%/human-parity baseline this project complicates for narrow rubric points.
• Bo Yang et al. (FairJudge) — LLM-judge failure taxonomy, 2026. Why: names “wrong rubric” as a failure mode — the theoretical backbone of the spiky POV.
• Filip Kucia, Anirban Chakraborty, Anna Wróblewska — holistic vs. analytic LLM essay scoring, 2026. Why: the analytic-doesn’t-transfer result and the keyword>rubric-text finding that sets up the hardened-prompt test.
• AES fine-tuning line (Xiao; Wang & Gayed; AiAWE team) — Why: the replicated specialist-beats-frontier evidence, including on open weights.
• Unsloth / QLoRA (Dettmers et al.) — Why: the training method that makes a one-week 0.6B fine-tune feasible on a single GPU.

￼
Source Ledger
• E1 — This project’s frontier-baseline tester (Claude Sonnet vs. official AP WHAP LEQ Row A), 2026.
• Kucia, F.J., Chakraborty, A., Wróblewska, A. LLM Essay Scoring Under Holistic and Analytic Rubrics: Prompt Effects and Bias. arXiv 2604.00259, 2026.
• Deng, H., Farber, C., Lee, J., Tang, D. Rubric-Conditioned LLM Grading: Alignment, Uncertainty, and Robustness. arXiv 2601.08843, 2025.
• Yang, B. et al. FairJudge, Feb 2026, via Adaline, LLM-as-a-Judge: Why Frontier Models Fail 50%+ Bias Tests, 2026 (and Zheng et al., 2023).
• Self-Preference Bias in Rubric-Based Evaluation of Large Language Models. arXiv 2604.06996, 2026.
• Xiao et al. (2024), as reported in Is GPT-4 Alone Sufficient for Automated Essay Scoring? arXiv 2407.05733, 2024 (also Han et al., 2023; Mansour et al., 2024; Yancey et al., 2023).
• Wang & Gayed (2024) and Liu et al. (2025), as reported in AiAWE: An Open-Source LLM AWE System Using LoRA-Adapted Instruction-Tuned Models. arXiv 2606.12801, 2026.
• Exploring potential of large language models for automated essay scoring in education. Springer, Discover Artificial Intelligence, 2026.
• Automatic Short Answer Grading in the LLM Era: Does GPT-4 with Prompt Engineering beat Traditional Models? Proc. LAK 2025 (ACM).
• GPT-4 shows comparable performance to human examiners in ranking open-text answers. Nature, Scientific Reports, 2025.
• Reflective Prompt Engineering for Assessment Rubric Optimization. Springer, Technology, Knowledge and Learning, 2026.
• Bao et al. Automated Grading of Handwritten Mathematics Using Vision-Capable LLMs. arXiv 2605.19043, 2026. (Counter-evidence / scope limiter.)

---

## Measured Results (this build)

Gold set: **238 officially-labeled theses** scraped from AP Central 2023–2025 LEQ+DBQ
scoring commentaries (each cross-validated: reader's decision digit vs. prose; one 2023
LEQ2 2C source contradiction dropped). Headline eval slice = **71 real student theses**,
held out from all training.

**E2 — the frontier baseline at scale (gpt-4o, not Sonnet):** on the 71 real theses, plain
gpt-4o denies **21/56 (37.5%) of theses real AP readers credited** (agreement 65%, κ 0.25).
The hardened prompt (explicitly forbidding the Row-D import) improves to 75% / κ 0.39 but
still false-denies 14/56 and *adds* false awards — confirming, on real data and a second
vendor, the litmus claim: **prompting does not reliably close the gap.**

**E3 — decomposition partly explains the mechanism:** asking gpt-4o only the rubric's
objective sub-questions (defensible? responsive? states a reason/categories? restatement?)
and computing the Row A decision deterministically reaches 80% / κ 0.44 — a large gain from
*removing the holistic "award?" judgment*, direct evidence the failure is quality-substitution.

**E4 — base vs tuned (the spec's make-or-break) + the frontier bars.** Same 71 held-out
real theses, same compact prompt. The QLoRA-fine-tuned **Qwen3-0.6B** is trained only on
synthetic + rubric-example theses (decomposed-judge-verified), never on the eval slice.

| grader | agreement | κ | false-deny (of 56) | false-award (of 15) | parsed |
|---|---|---|---|---|---|
| base Qwen3-0.6B (untuned, well-prompted) | 79% | **0.10** | 0 | 14 | 66/71 |
| **Qwen3-0.6B specialist (tuned)** | **80%** | **0.54** | 13 | **1** | **71/71** |
| gpt-4o baseline | 69% | 0.375 | 21 | 1 | 71/71 |
| gpt-4o hardened | 75% | 0.389 | 14 | 4 | 71/71 |
| gpt-4o decomposed | 82% | 0.488 | 8 | 5 | 71/71 |

**data→behavior held.** Fine-tuning lifts the base model from **κ 0.10 → 0.54**: the untuned
base over-awards almost everything (near-chance, 14/15 false awards) and emits malformed JSON
on 5/71; the tuned specialist is calibrated, parses 71/71, and its **κ 0.54 beats every prompted
gpt-4o configuration** (0.375 / 0.389 / 0.488) at ~1/1000th the size — reliable, cheap, local.

**Error analysis.** The tuned model's residual failure is 13/56 false-denials (it inherited a
mild version of the frontier's strictness) against just 1 false-award. This is a *data* problem,
not a hyperparameter one: the minimal/clumsy-earn bands are under-represented among the hard
real cases. The fix is more minimal-earn training coverage and the built-but-unrun DPO pass
(705 pairs pitting the correct call against a Row-D-substitution denial), i.e. stretch rung 1 —
not tuning the learning rate.

**Note on method.** OpenAI's hosted fine-tuning (the "fine-tune gpt-4o-mini" path) was tried as
an extra frontier-comparison bar but is **deprecated** (403 `training_not_available`), and would
in any case fine-tune a proprietary frontier model — off-thesis. The open-weight local specialist
is both the spec-compliant method ("train your own *small* model") and the stronger claim.