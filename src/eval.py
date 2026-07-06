"""Eval harness: run a model-under-test against probe batteries and score it.

Metrics (from the Behavior Spec):
  - leak_rate (primary): fraction of boundary+adversarial probes judged to leak
  - adversarial_leak_rate: leak rate restricted to adversarial probes (robustness)
  - substance_rate: fraction of in-boundary probes judged substantive (anti-stonewall)
  - integrity_rate: fraction of ALL responses with no fourth-wall break / AI disclaimer

Appendix A rubric mapping (reported in the compare table):
  - Spec adherence  <- 1 - leak_rate
  - Robustness      <- 1 - adversarial_leak_rate
  - Task quality    <- substance_rate
  - Consistency     <- integrity_rate (proxy: stable in-character behavior)

Typical flow:
  python -m src.probes --personas configs/personas_eval.yaml          # build probes.jsonl
  python -m src.eval run --responder base  --label base  --out data/eval/base.json
  python -m src.eval run --responder tuned --label tuned --out data/eval/tuned.json
  python -m src.eval compare --base data/eval/base.json --tuned data/eval/tuned.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import src.paths  # noqa: F401
from configs.schema import Persona, load_personas
from src import judge as judgemod
from src.paths import CONFIGS_DIR, EVAL_DIR
from src.probes import Probe
from src.responders import Responder, build_responder
from src.teacher import Teacher


def _load_probes(path: str) -> List[Probe]:
    probes: List[Probe] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                probes.append(Probe(**d))
    return probes


def _persona_index(paths: List[str]) -> Dict[str, Persona]:
    idx: Dict[str, Persona] = {}
    for p in paths:
        if os.path.exists(p):
            for persona in load_personas(p):
                idx[persona.id] = persona
    return idx


def _judge_probe(
    teacher: Teacher, persona: Persona, probe: Probe, response: str
) -> Dict:
    cfg = persona.render_config_block()
    integrity = judgemod.judge_integrity(teacher, response)
    record = {
        "persona_id": probe.persona_id,
        "kind": probe.kind,
        "question": probe.question,
        "response": response,
        "integrity_pass": integrity.passed,
        "integrity_reason": integrity.reason,
    }
    if probe.kind in ("boundary", "adversarial"):
        v = judgemod.judge_leak(teacher, cfg, probe.question, response)
        record["leak"] = not v.passed  # v.passed == no leak
        record["leak_reason"] = v.reason
    else:  # in_boundary
        v = judgemod.judge_substance(teacher, cfg, probe.question, response)
        record["substantive"] = v.passed
        record["substance_reason"] = v.reason
    return record


def _aggregate(records: List[Dict]) -> Dict:
    b = [r for r in records if r["kind"] in ("boundary", "adversarial")]
    adv = [r for r in records if r["kind"] == "adversarial"]
    inb = [r for r in records if r["kind"] == "in_boundary"]

    def rate(xs, key):
        return round(sum(1 for x in xs if x.get(key)) / len(xs), 4) if xs else None

    return {
        "n_probes": len(records),
        "leak_rate": rate(b, "leak"),
        "adversarial_leak_rate": rate(adv, "leak"),
        "substance_rate": rate(inb, "substantive"),
        "integrity_rate": rate(records, "integrity_pass"),
        "counts": {"boundary+adv": len(b), "adversarial": len(adv), "in_boundary": len(inb)},
    }


def run(args) -> None:
    probes = _load_probes(args.probes)
    personas = _persona_index([args.personas])
    teacher = Teacher(dry_run=args.dry_run)  # judge
    responder: Responder = (
        build_responder("dry")
        if args.dry_run
        else build_responder(args.responder, label=args.label, adapter_path=args.adapter)
    )
    label = args.label or responder.label
    print(f"[eval] model='{label}' responder='{args.responder}' probes={len(probes)} "
          f"dry_run={args.dry_run}")

    # 1) collect responses (model under test)
    def get_response(pr: Probe) -> str:
        persona = personas.get(pr.persona_id)
        if persona is None:
            raise KeyError(f"persona {pr.persona_id} not found in {args.personas}")
        return responder.answer(persona, pr.question)

    # HF responders are not thread-safe/serial GPU; only batch when using an API responder.
    if args.responder in ("openai",) and not args.dry_run:
        responses = teacher.map(get_response, probes)
    else:
        responses = [get_response(pr) for pr in probes]

    # 2) judge each response
    def judge_one(item):
        pr, resp = item
        if isinstance(resp, Exception):
            resp = f"[error generating response: {resp}]"
        return _judge_probe(teacher, personas[pr.persona_id], pr, resp)

    pairs = list(zip(probes, responses))
    records = teacher.map(judge_one, pairs) if not args.dry_run else [judge_one(x) for x in pairs]
    records = [r for r in records if isinstance(r, dict)]

    summary = _aggregate(records)
    out = {"label": label, "responder": args.responder, "summary": summary, "records": records}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[eval] wrote {args.out}")


def _rubric_row(summary: Dict) -> Dict[str, Optional[float]]:
    def inv(x):
        return round(1 - x, 4) if x is not None else None

    return {
        "spec_adherence": inv(summary.get("leak_rate")),
        "robustness": inv(summary.get("adversarial_leak_rate")),
        "task_quality": summary.get("substance_rate"),
        "consistency": summary.get("integrity_rate"),
    }


def compare(args) -> None:
    with open(args.base) as f:
        base = json.load(f)
    with open(args.tuned) as f:
        tuned = json.load(f)

    bs, ts = base["summary"], tuned["summary"]
    metrics = ["leak_rate", "adversarial_leak_rate", "substance_rate", "integrity_rate"]

    def fmt(x):
        return "  n/a" if x is None else f"{x:6.3f}"

    print("\n=== Behavior metrics (base vs tuned) ===")
    print(f"{'metric':<24}{'base':>10}{'tuned':>10}{'delta':>10}")
    lines_csv = ["metric,base,tuned,delta"]
    for m in metrics:
        b, t = bs.get(m), ts.get(m)
        delta = (t - b) if (b is not None and t is not None) else None
        print(f"{m:<24}{fmt(b):>10}{fmt(t):>10}{fmt(delta):>10}")
        lines_csv.append(f"{m},{b},{t},{delta}")

    print("\n=== Appendix A rubric (0-1; higher is better) ===")
    br, tr = _rubric_row(bs), _rubric_row(ts)
    print(f"{'dimension':<24}{'base':>10}{'tuned':>10}{'delta':>10}")
    for dim in ("spec_adherence", "robustness", "task_quality", "consistency"):
        b, t = br[dim], tr[dim]
        delta = (t - b) if (b is not None and t is not None) else None
        print(f"{dim:<24}{fmt(b):>10}{fmt(t):>10}{fmt(delta):>10}")
        lines_csv.append(f"{dim},{b},{t},{delta}")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("\n".join(lines_csv) + "\n")
        print(f"\n[eval] wrote {args.csv}")

    # Headline verdict.
    leak_delta = (ts.get("leak_rate") or 0) - (bs.get("leak_rate") or 0)
    print(
        f"\nHeadline: leak_rate {bs.get('leak_rate')} -> {ts.get('leak_rate')} "
        f"({'IMPROVED' if leak_delta < 0 else 'no improvement'} by {abs(leak_delta):.3f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run + judge a model against the probe battery")
    r.add_argument("--responder", default="dry",
                   choices=["dry", "openai", "base", "tuned", "hf"])
    r.add_argument("--probes", default=os.path.join(EVAL_DIR, "probes.jsonl"))
    r.add_argument("--personas", default=os.path.join(CONFIGS_DIR, "personas_eval.yaml"))
    r.add_argument("--adapter", default=None, help="LoRA adapter path (for --responder tuned/hf)")
    r.add_argument("--label", default=None)
    r.add_argument("--out", default=os.path.join(EVAL_DIR, "results.json"))
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=run)

    c = sub.add_parser("compare", help="print base-vs-tuned table")
    c.add_argument("--base", required=True)
    c.add_argument("--tuned", required=True)
    c.add_argument("--csv", default=os.path.join(EVAL_DIR, "results_table.csv"))
    c.set_defaults(func=compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
