"""Reproduce the E1 frontier baseline and run the hardened-prompt litmus gate.

    # single condition
    python -m src.rowa.baseline run --condition baseline --backend openai \
        --out data/rowa/baseline_openai.json

    # the gate: run baseline AND hardened, compare, print verdict
    python -m src.rowa.baseline gate --backend openai

    # wiring check, no API key / no spend
    python -m src.rowa.baseline gate --dry-run

The gate question (from the brainlift): does the hardened prompt -- which explicitly
forbids importing Row-D "evaluate the extent / broader analytical framework" criteria
-- fix the frontier's false denials of minimal-but-defensible theses?
  - Still false-denies minimal theses  -> gap is NOT reliably promptable -> fine-tune.
  - Fixes the false denials             -> pivot up-rubric (contextualization / full LEQ).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from src.rowa import rubric
from src.rowa.grader import FrontierGrader

GOLD_PATH = Path("data/rowa/gold_all.jsonl")


def load_gold(path: Path = GOLD_PATH) -> List[dict]:
    """Normalize either schema (hand-ported `prompt_id`/`source`/`cat`, or the unified
    `prompt`/`subset`) into {id, prompt, thesis, label, subset, cat}."""
    out = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            prompt = r.get("prompt") or rubric.PROMPTS.get(r.get("prompt_id"), "")
            out.append(
                {
                    "id": r.get("id"),
                    "prompt": prompt,
                    "thesis": r["thesis"],
                    "label": r["label"],
                    "subset": r.get("subset", r.get("source", "?")),
                    "cat": r.get("cat", r.get("source_detail", "")),
                }
            )
    return out


def grade_all(grader: FrontierGrader, items: List[dict], condition: str,
              workers: int = 6) -> List[dict]:
    from concurrent.futures import ThreadPoolExecutor

    system = rubric.grader_system(condition)

    def _grade(it):
        user = rubric.grader_user(it["prompt"], it["thesis"])
        try:
            res = grader.grade(system, user)
            point, reason, err = res.point, res.reason, None
        except Exception as e:  # noqa: BLE001 - one bad item shouldn't sink the run
            point, reason, err = None, "", str(e)[:200]
        return {
            "id": it["id"], "subset": it["subset"], "cat": it["cat"], "label": it["label"],
            "point": point,
            "agree": (point == it["label"]) if point is not None else None,
            "reason": reason, "error": err,
        }

    w = 1 if grader.dry_run else workers
    if w == 1:
        # Sequential: local (MPS) model.generate misbehaves inside a worker thread, and
        # there's nothing to parallelize anyway.
        return [_grade(it) for it in items]
    with ThreadPoolExecutor(max_workers=w) as ex:
        return list(ex.map(_grade, items))


def _kappa(rows: List[dict]) -> Optional[float]:
    """Cohen's kappa between official labels and predictions."""
    n = len(rows)
    if n == 0:
        return None
    po = sum(r["agree"] for r in rows) / n
    p_lab1 = sum(r["label"] for r in rows) / n
    p_pred1 = sum(r["point"] for r in rows) / n
    pe = p_lab1 * p_pred1 + (1 - p_lab1) * (1 - p_pred1)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def metrics(rows: List[dict]) -> dict:
    graded = [r for r in rows if r["point"] is not None]
    sample = [r for r in graded if r["subset"] == "sample"]  # real students = headline

    def rate(arr):
        return round(sum(r["agree"] for r in arr) / len(arr), 3) if arr else None

    def subrate(sub):
        return rate([r for r in graded if r["subset"] == sub])

    false_awards = [r for r in graded if r["label"] == 0 and r["point"] == 1]
    false_denials = [r for r in graded if r["label"] == 1 and r["point"] == 0]
    # The POV's target harm: real student theses that officially EARNED but get denied.
    sample_earn = [r for r in sample if r["label"] == 1]
    sample_deny_gold = [r for r in sample if r["label"] == 0]
    sample_false_denials = [r for r in sample_earn if r["point"] == 0]
    sample_false_awards = [r for r in sample_deny_gold if r["point"] == 1]

    return {
        "n_graded": len(graded),
        "n_errors": len(rows) - len(graded),
        "agreement_overall": rate(graded),
        "kappa_overall": _kappa(graded),
        "agreement_sample": subrate("sample"),
        "kappa_sample": _kappa(sample),
        "agreement_rubric": subrate("rubric"),
        "false_awards": len(false_awards),
        "false_denials": len(false_denials),
        "sample_false_denials": len(sample_false_denials),
        "sample_earn_total": len(sample_earn),
        "sample_false_awards": len(sample_false_awards),
        "sample_deny_total": len(sample_deny_gold),
    }


def _pct(x):
    return "  -  " if x is None else f"{round(x * 100):>3}%"


def print_rows(rows: List[dict], title: str, only_wrong: bool = True):
    shown = [r for r in rows if (not only_wrong or r["agree"] is not True)]
    print(f"\n=== {title} ({'disagreements only' if only_wrong else 'all'}: "
          f"{len(shown)}/{len(rows)}) ===")
    print(f"{'id':>3} {'subset':<10} {'cat':<22} {'truth':>5} {'pred':>4} {'':2} reason")
    for r in shown:
        pred = "err" if r["point"] is None else r["point"]
        mark = "?" if r["agree"] is None else ("OK" if r["agree"] else "XX")
        reason = (r["reason"] or r["error"] or "")[:60]
        print(
            f"{str(r['id']):>3} {str(r['subset'])[:10]:<10} {str(r['cat'])[:22]:<22} "
            f"{r['label']:>5} {str(pred):>4} {mark:>2} {reason}"
        )


def print_metrics(m: dict, label: str):
    print(f"\n--- metrics [{label}] ---")
    print(f"  graded / errors          : {m['n_graded']} / {m['n_errors']}")
    print(f"  agreement all / kappa    : {_pct(m['agreement_overall'])} / {m['kappa_overall']}")
    print(f"  agreement sample / kappa : {_pct(m['agreement_sample'])} / {m['kappa_sample']}"
          "   <- real students (headline)")
    print(f"  agreement rubric         : {_pct(m['agreement_rubric'])}")
    print(f"  false awards (0->1)      : {m['false_awards']}")
    print(f"  false denials (1->0)     : {m['false_denials']}")
    print(
        f"  REAL false denials       : {m['sample_false_denials']}/{m['sample_earn_total']} "
        "real earned theses wrongly DENIED (the POV's target harm)"
    )
    print(
        f"  REAL false awards        : {m['sample_false_awards']}/{m['sample_deny_total']} "
        "real denied theses wrongly AWARDED"
    )


def gate_verdict(base_m: dict, hard_m: dict) -> str:
    b, bt = base_m["sample_false_denials"], base_m["sample_earn_total"]
    h = hard_m["sample_false_denials"]
    if h >= 2 and h >= b * 0.5:
        return (
            f"GATE PASSED. On real student theses, the hardened prompt still false-denies "
            f"{h}/{bt} that officially EARNED (baseline denied {b}). Explicitly forbidding the "
            "Row-D import does NOT reliably close the gap -> fine-tuning a Row-A specialist is justified."
        )
    if h == 0:
        return (
            f"GATE NOT PASSED. The hardened prompt drove real false-denials {b} -> 0. Prompting "
            "closes this gap -> pivot up-rubric (contextualization / full analytic score)."
        )
    return (
        f"GATE MARGINAL. Hardened prompt cut real false-denials {b} -> {h}/{bt}. "
        "Fine-tuning still defensible; widen the battery to confirm."
    )


def run_condition(args, condition: str) -> dict:
    grader = FrontierGrader(backend=args.backend, model=args.model, dry_run=args.dry_run)
    items = load_gold(Path(args.gold))
    if args.limit:
        items = items[: args.limit]
    rows = grade_all(grader, items, condition)
    m = metrics(rows)
    out = {
        "backend": args.backend,
        "model": grader.model,
        "condition": condition,
        "metrics": m,
        "rows": rows,
    }
    return out


def cmd_run(args):
    out = run_condition(args, args.condition)
    print_rows(out["rows"], f"{out['model']} / {args.condition}")
    print_metrics(out["metrics"], args.condition)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


def cmd_gate(args):
    base = run_condition(args, "baseline")
    hard = run_condition(args, "hardened")
    print_rows(base["rows"], f"{base['model']} / baseline")
    print_metrics(base["metrics"], "baseline")
    print_rows(hard["rows"], f"{hard['model']} / hardened")
    print_metrics(hard["metrics"], "hardened")
    verdict = gate_verdict(base["metrics"], hard["metrics"])
    print("\n" + "=" * 72 + f"\n{verdict}\n" + "=" * 72)
    combined = {"baseline": base, "hardened": hard, "verdict": verdict}
    outp = Path(args.out or "data/rowa/litmus_gate.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(combined, indent=2))
    print(f"wrote {outp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="openai", choices=["openai", "anthropic"])
    ap.add_argument("--model", default=None, help="override model id")
    ap.add_argument("--gold", default=str(GOLD_PATH), help="gold jsonl path")
    ap.add_argument("--limit", type=int, default=None, help="grade only first N items")
    ap.add_argument("--dry-run", action="store_true", help="offline stub, no API calls")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="grade one condition")
    r.add_argument("--condition", default="baseline", choices=["baseline", "hardened"])
    r.add_argument("--out", default=None)
    r.set_defaults(func=cmd_run)

    g = sub.add_parser("gate", help="run baseline + hardened, print litmus verdict")
    g.add_argument("--out", default=None)
    g.set_defaults(func=cmd_gate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
