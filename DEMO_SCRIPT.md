# Demo Video Script — AP World History Row A Thesis Grader

Target length: **~4 minutes**. Spoken lines are in plain text; screen directions are in
**[brackets]**. Times are cumulative guides, not hard cuts.

Before recording:
- `pip install -r requirements-demo.txt`
- Have the demo ready: `python -m src.rowa.demo` (specialist vs gpt-4o) or
  `python -m src.rowa.demo --no-frontier` if you don't want to depend on an API key live.
- Have two tabs/windows ready: (1) the Gradio demo, (2) the eval results table
  (`data/rowa/eval_results.csv`) or the HF model card.

---

## 0:00 — Hook (the problem)

**[On screen: the Gradio demo titled "AP World History — Row A Thesis Grader."]**

"Here's a claim that sounds counterintuitive: for one narrow grading task, a 0.6-billion-
parameter model I fine-tuned on my laptop is *more reliable* than a prompted frontier model
like GPT-4o.

The task is grading the AP World History thesis point — what the College Board calls Row A.
It's a binary decision: does this thesis earn the point, yes or no?"

---

## 0:25 — The behavior spec (what "correct" means)

**[On screen: briefly show the compact grader prompt, or just narrate.]**

"Row A has a precise rule. A thesis earns the point if it makes a historically defensible
claim that answers the prompt and establishes any one line of reasoning. That's it.

It does NOT need to 'evaluate the extent,' show complexity, or be eloquent — those belong to
other rows. My spec is one falsifiable sentence: the model outputs `{point, reason}` matching
the official Row A decision, awarding minimal-but-defensible theses and denying restatements,
off-topic, or non-defensible claims — without importing higher-row criteria."

---

## 0:50 — The litmus test (why prompting isn't enough)

**[On screen: paste EXAMPLE 1 — the minimal Mongol thesis — into the demo. Click "Grade Row A."]**

Prompt: the 2023 Mongol LEQ.
Thesis: "The Mongols made Silk Road trade safer, which increased cultural exchange between
distant peoples."

"This is a minimal but valid thesis — a defensible claim with a reason. Real AP readers award
it. Watch the frontier model."

**[Point to the gpt-4o output denying or hedging it.]**

"GPT-4o tends to deny theses like this — it silently imports 'evaluate the extent' and
sophistication criteria that Row A doesn't require. I measured this: on 71 real student
theses, plain GPT-4o false-denies 21 of the 56 that officially earned. Even a hardened prompt
that explicitly forbids that mistake still misses 13. Prompting does not reliably close the
gap — and that's exactly the test this project has to pass."

**[Point to the specialist output awarding it.]**

"My specialist awards it, and cites the actual Row A criterion."

---

## 1:40 — The anti-cheat (it's not just saying yes)

**[On screen: EXAMPLE 3 — the eloquent-empty thesis.]**

Thesis: "It is beyond dispute that the Mongol conquests profoundly and irrevocably
transformed the lives of countless peoples across Eurasia."

"A model that just says 'earn' to everything would be useless. Here's an eloquent thesis that
makes no actual defensible claim and states no reasoning. It should be denied."

**[Show the specialist correctly denying it.]**

"The specialist denies it — it learned the boundary, not a bias in one direction."

---

## 2:15 — The live test (the reliability claim, unscripted)

**[On screen: type a brand-new thesis you make up on the spot, for the same prompt.]**

"Let me write one live, so you know it's not cherry-picked."

**[Type something minimal-but-earning, e.g.:]**
"Mongol rule connected Eurasia because their control of the roads let merchants and ideas
travel safely."

**[Grade it.]**

"Defensible claim, one clear reason — that earns. The specialist holds the line."

**[Optional: type a bare restatement of the prompt and show it correctly denies.]**

---

## 2:45 — The numbers (base vs tuned vs frontier)

**[On screen: the results table — eval_results.csv or the HF model card table.]**

"Here's the whole comparison on 71 held-out real theses I never trained on. The metric that
matters is Cohen's kappa, because the set is imbalanced — raw agreement looks similar for
everyone at around 80%, but kappa exposes who's actually reasoning.

- The untuned base model: kappa 0.10 — near chance. It over-awards almost everything and even
  emits malformed JSON on 5 of 71.
- My fine-tuned specialist: kappa 0.54, and it parses all 71.
- Every prompted GPT-4o configuration: 0.375, 0.39, and 0.49.

Fine-tuning moved the same base model from 0.10 to 0.54 — beating every prompted frontier
setup, at roughly one-thousandth the size. That's the headline: the behavior came from the
data, not from a bigger model."

---

## 3:20 — How, and honest limits

**[On screen: the HF dataset page or the dataset card schema section.]**

"Two things carried this, and neither was the training loop. First, the data: synthetic
theses generated across controlled quality bands, with labels fixed by construction and
verified by a decomposed, bias-resistant judge — never a holistic frontier 'award?' call,
which would re-import the very bias I'm fixing. Nothing I test on is in training.

Second, the eval, which I built before training. The dataset and model are both on Hugging
Face with full schema docs and this results table.

The honest limit: my specialist still false-denies 13 of 56 — a milder version of the
frontier's strictness. I diagnosed that as a data-coverage problem and built a v2 dataset —
contrastive boundary examples, assistant-only loss, a real dev split — to fix it in the data,
not the hyperparameters."

---

## 3:45 — Close

**[On screen: the demo, or the HF links.]**

"So: a tiny, cheap, local model doing one constrained thing more reliably than a prompted
frontier model — proven with a base-versus-tuned number on real data. That's the whole point.
Thanks for watching."

**[End.]**

---

### On-screen assets checklist
- Gradio demo running (`src/rowa/demo.py`)
- Results table: `data/rowa/eval_results.csv`
- HF dataset: https://huggingface.co/datasets/anshulmago1/ap-rowa-thesis-grading
- HF model: https://huggingface.co/anshulmago1/ap-rowa-thesis-grader-qwen3-0.6b

### Key numbers (say these exactly)
- Base Qwen3-0.6B: **κ 0.10**, parses 66/71
- Specialist (tuned): **κ 0.54**, parses 71/71
- gpt-4o baseline / hardened / decomposed: **κ 0.375 / 0.389 / 0.488**
- Plain gpt-4o false-denies **21/56**; hardened still **13/56**
- Eval slice: **71 real student theses (56 earn / 15 deny)**, held out
