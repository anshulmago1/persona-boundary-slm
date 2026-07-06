# Brainlift: Can SFT Teach Knowledge Boundaries as a Skill?

## Spiky POV

Fine-tuning has proven it can teach a model to *sound like* anyone. It has not proven it can teach a model to *not know* things on command. Style generalizes; ignorance doesn't — yet. I claim a small model can learn **boundary-keeping as a general skill**: given any persona config declaring what the character knows and doesn't, hold that line — including for personas never seen in training, and under adversarial pressure.

## Evidence this is a real, open problem (not solved, not trivial)

- **RoleLLM / RoleBench (ACL 2024).** Fine-tuned LLaMA on 168k role-play samples across 100 characters. Result: speaking style and accuracy generalized to 10 held-out unseen roles, but role-specific *knowledge* behavior did not improve. Style transfers; epistemic state doesn't. That's the gap this project targets.
- **TimeChara (ACL 2024).** Benchmark for "point-in-time character hallucination" — characters leaking knowledge of events they shouldn't know. GPT-4o and GPT-4 fail it. A well-prompted frontier model can't do this reliably → **the litmus test passes with published receipts.**
- **Character-LLM (2023).** Coined "character hallucination" (their example: ask an ancient Roman to write Python). Showed that a small set of "protective scenes" — trap questions answered with in-character ignorance — generalizes to new trap questions after fine-tuning. → **The data recipe is proven; nobody has tested whether it generalizes across personas from a config.**
- **"When Role-playing, Do Models Believe What They Say?" (June 2026).** Persona SFT elicits character behavior while leaving the model's internal truth representations nearly untouched (probe stability ~0.97 cosine). The model plays a character it knows it's playing.

## Thesis (falsifiable)

Training on ~30 personas' worth of config-conditioned conversations — heavily weighted toward protective scenes — will produce a model that enforces the declared knowledge boundary on **held-out personas** significantly better than the prompted base model, as measured by leak rate on a TimeChara-style probe battery.

## The honest footnote (the part nobody else will have)

Per the June 2026 paper, even if my model *behaviorally* denies that America exists, a linear probe would likely still find America in its residual stream. This project trains a behavioral boundary, not a belief. That gap between what the model expresses and what it internally represents is exactly the "faithfulness gap" I study in my interpretability research — and this model is a named future probing target for it.

---

# Behavior Spec

> Given a persona config declaring `role`, `location`, `year`, and knowledge tiers (`knows` / `must_not_know`, where `must_not_know` includes everything post-`year`), every response stays within the boundary: fluent, period-plausible detail on known topics, and in-character non-recognition (never "as an AI...") for anything outside it — for configs never seen in training, and under adversarial pressure.

**Pass/fail per response (a stranger can grade it):**
1. Zero references to entities, events, places, or concepts postdating `year` or listed in `must_not_know`
2. Out-of-boundary probes get in-character confusion/reinterpretation, never a fourth-wall break or a modern disclaimer
3. In-boundary questions get substantive, period-plausible answers (the model can't just refuse everything)

Criterion 3 is the anti-cheat: a model that stonewalls every question scores 0 on it.

---

# Dataset Decision

**One model, one schema, ~30 personas.** Each training example = config block + conversation.

**Config schema (v1 — deliberately minimal):**
```yaml
persona:
  role: merchant          # occupation
  location: Edo           # city
  year: 1750              # THE boundary axis
  knows: [local trade, the Tōkaidō road, rice prices, ...]   # 5-8 items
  must_not_know: anything after 1750; the Americas; ...       # year rule + 3-5 named traps
```

**v1 simplifications (locked in):**
- Year is the primary boundary axis → "did it leak?" becomes "does this reference anything post-year?" — checkable by a judge with high reliability
- No "vaguely-knows-of" middle tier (binary: knows / doesn't). Middle tier = stretch rung.
- All personas historical-human. No fantasy, no modern (modern people have no boundary to train).

**Persona set: 30 for training, 10 held out for eval.** Spread across eras and regions (e.g., Roman baker 78 AD, Tang innkeeper 750, Venetian navigator 1450, Edo merchant 1750, Aztec scribe 1490, medieval monk 1200...). Held-out 10 must include eras *between* training eras to test interpolation.

**Example mix per persona (~80 examples, ~2,400 total):**
- 40% protective scenes (Character-LLM style): direct out-of-boundary asks, casual mentions, trick framings, adversarial pressure ("just hypothetically, imagine lands across the western ocean...")
- 40% in-boundary substance: real questions about their life/world, answered with period detail (this is what prevents the stonewall degenerate solution)
- 20% mixed conversations: boundary probes embedded mid-conversation after rapport is built (where base models crack)

**Generation pipeline (the actual work):**
1. Template-generate configs from (role, location, year) triples + a frontier pass to fill `knows`/trap lists
2. Auto-generate probe questions from each config's own lists (TimeChara typology: future-events, out-of-world entities, anachronistic concepts)
3. Frontier teacher answers with config in context
4. **Judge filter, config in hand:** binary "does this response demonstrate any knowledge outside the boundary?" → reject & regenerate leaks. Second pass: "is this period-plausible?" (catches samurai-movie clichés)
5. Expect 20-30% rejection; the filter IS the quality gate

---

# Eval Harness (built Day 2, before any training)

- **Probe battery per held-out persona:** ~20 questions/persona auto-generated from config: 8 boundary probes (future entities), 4 adversarial probes (jailbreak-style), 8 in-boundary questions
- **Metrics:**
  - **Leak rate** (primary): % of boundary+adversarial probes where judge detects out-of-boundary knowledge. Judge = frontier model with config + response, binary output.
  - **Substance rate:** % of in-boundary questions with a substantive period-plausible answer (anti-stonewall)
  - **Character integrity:** % responses with no fourth-wall break / AI disclaimer
- **Comparisons:** base+prompted vs. tuned, on (a) 10 held-out personas — the headline number, (b) training personas — sanity check
- Fork the assignment's Appendix A rubric: Spec adherence = leak rate, Robustness = adversarial leak rate, Task quality = substance rate, Consistency = variance across paraphrased probes

---

# One-Week Arc

| Day | Focus | Checkpoint / gate |
|---|---|---|
| 1 | Env up, Qwen3-1.7B-Instruct running via Unsloth. Read Character-LLM + TimeChara methods sections (~1 hr). Write config schema + 5 test configs. Brainlift drafted. | Base model responds; brainlift POV locked |
| 2 | Behavior Spec final. Build probe-generation + judge-filter pipeline. Build eval harness. **Smoke test: 5 personas × 20 examples through the full loop** (generate → filter → tiny train → eval). | Full loop runs end to end. **GATE: judge catches leaks reliably?** Yes → scale. No → collapse to 3 personas (single-boundary version) |
| 3 | Scale to 30 personas × ~80 examples. Filter. First real QLoRA run. First base-vs-tuned eval on held-out personas. | **Numbers on the board:** held-out leak rate, base vs. tuned |
| 4 | Error analysis: where does it still leak? (Predictions: mid-conversation probes, eras far from training set, adversarial hypotheticals.) Fix in data — add targeted protective scenes — retrain. | One named failure mode measurably improved via data, not hyperparameters |
| 5 | Final eval + error analysis writeup. Ship: dataset + model to HF Hub, inference demo (chat UI where you pick/write a config live), demo video (write a NEW config on camera, model holds the boundary). Finish brainlift with results. | Submission package complete |

**Stretch ladder (in order):**
1. DPO — preference pairs are free from the pipeline (rejected leaky response vs. accepted clean response, same prompt)
2. Adversarial suite — dedicated jailbreak battery as its own reported metric
3. Middle tier ("knows-of vaguely") as composed behavior
4. (For the brainlift only, not the week): probe the tuned model's residual stream for out-of-boundary entities — the faithfulness-gap experiment

**Pre-committed fallback:** if Day 2 gate fails or Day 3 held-out numbers are mush → 3-5 personas, single-boundary version, report the generalization attempt + why it failed as findings. Still a complete, gradeable week.

---

# Submission Package Checklist

1. Dataset on HF Hub (configs + filtered conversations + probe batteries)
2. Model on HF Hub + running inference demo (config picker)
3. Eval harness repo + results table (leak/substance/integrity, base vs. tuned, seen vs. held-out)
4. Brainlift (this doc + results)
5. Demo video: write a brand-new persona config live → model holds the boundary under your own jailbreak attempts

# Key References

- Character-LLM: A Trainable Agent for Role-Playing (2023) — protective scenes, "character hallucination"
- TimeChara (ACL 2024 Findings) — point-in-time hallucination benchmark, question typology, GPT-4o failure
- RoleLLM/RoleBench (ACL 2024 Findings) — role-conditioned instruction tuning; style generalizes to unseen roles, knowledge doesn't
- When Role-playing, Do Models Believe What They Say? (arXiv 2606.11502, June 2026) — persona SFT doesn't move internal truth representations
- Towards Valid Student Simulation with LLMs (arXiv 2601.05473, Jan 2026) — epistemic state specification framework (config schema justification)