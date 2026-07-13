"""Headline eval: fine-tuned specialist vs prompted frontier on real student theses.

Graders compared (all emit a Row A 0/1 decision, scored against official labels):
  - frontier-baseline  : gpt-4o, plain rubric prompt (the industry default)
  - frontier-hardened  : gpt-4o, rubric + explicit "don't import Row-D" clause
  - frontier-decomposed: gpt-4o answering only objective sub-questions (judge_rowa)
  - specialist         : the fine-tuned Qwen3-0.6B + LoRA adapter

Eval set defaults to the held-out real-student slice (subset=='sample'), which the
specialist never trained on.

    python -m src.rowa.eval --graders frontier-baseline frontier-hardened specialist
    python -m src.rowa.eval --dry-run          # wiring only
    python -m src.rowa.eval --subset all       # include rubric examples too
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List

from src.rowa import rubric
from src.rowa.baseline import load_gold, grade_all, metrics, print_rows
from src.rowa.grader import FrontierGrader, LocalGrader, GraderResult
from src.teacher import Teacher, extract_json

RESULTS = Path("data/rowa/eval_results.json")
CSV = Path("data/rowa/eval_results.csv")


class DecomposedGrader:
    """gpt-4o answering the rubric's objective sub-questions -> deterministic Row A label."""

    def __init__(self, dry_run: bool = False):
        from src.rowa.judge_rowa import decomposed_grade
        self._fn = decomposed_grade
        self.teacher = Teacher(dry_run=dry_run)
        self.dry_run = dry_run
        self.model = "gpt-4o-decomposed"

    def grade(self, system: str, user: str) -> GraderResult:
        # system/user unused; decomposed grader builds its own prompt from (prompt, thesis).
        raise NotImplementedError  # graded via grade_items below


def _grade_decomposed(g: DecomposedGrader, items):
    from concurrent.futures import ThreadPoolExecutor

    def _one(it):
        try:
            d = g._fn(g.teacher, it["prompt"], it["thesis"])
            point, reason, err = d.label, d.reason, None
        except Exception as e:  # noqa: BLE001
            point, reason, err = None, "", str(e)[:200]
        return {"id": it["id"], "subset": it["subset"], "cat": it["cat"], "label": it["label"],
                "point": point, "agree": (point == it["label"]) if point is not None else None,
                "reason": reason, "error": err}

    w = 1 if g.dry_run else 6
    with ThreadPoolExecutor(max_workers=w) as ex:
        return list(ex.map(_one, items))


def _make(kind: str, args):
    if kind == "frontier-baseline":
        return ("baseline", FrontierGrader(backend="openai", model=args.model, dry_run=args.dry_run))
    if kind == "frontier-hardened":
        return ("hardened", FrontierGrader(backend="openai", model=args.model, dry_run=args.dry_run))
    if kind == "frontier-decomposed":
        return ("hardened", DecomposedGrader(dry_run=args.dry_run))
    if kind == "base":
        # The untuned base model, well-prompted (compact) — the spec's litmus baseline.
        base = os.getenv("BASE_MODEL", "Qwen/Qwen3-0.6B")
        return ("compact", LocalGrader(base, adapter_path=None, dry_run=args.dry_run))
    if kind in ("specialist", "specialist-calibrated"):
        base = os.getenv("BASE_MODEL", "Qwen/Qwen3-0.6B")
        adapter = args.adapter or os.getenv("TUNED_MODEL", "outputs/rowa-thesis-qlora")
        # MUST match the prompt the specialist was trained with (compact), or its learned
        # behavior does not transfer.
        return ("compact", LocalGrader(base, adapter, dry_run=args.dry_run))
    if kind == "ft-4o-mini":
        # Fine-tuned frontier comparison bar; trained on the compact-prompt data.
        model = os.getenv("OPENAI_FT_MODEL")
        mf = Path("data/rowa/openai_ft_model.txt")
        if not model and mf.exists():
            model = mf.read_text().strip()
        if not model:
            raise ValueError("no fine-tuned model id; run src.rowa.finetune_openai first")
        return ("compact", FrontierGrader(backend="openai", model=model, dry_run=args.dry_run))
    raise ValueError(kind)


def _calibration_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record["messages"]
            system = next(m["content"] for m in messages if m["role"] == "system")
            user = next(m["content"] for m in messages if m["role"] == "user")
            answer = next(m["content"] for m in messages if m["role"] == "assistant")
            rows.append({"system": system, "user": user,
                         "label": int(extract_json(answer)["point"])})
    return rows


def _calibrate(grader: LocalGrader, path: Path) -> dict:
    """Choose the kappa-maximizing threshold on development data only."""
    rows = _calibration_rows(path)
    scored = [(grader.point_probability(r["system"], r["user"]), r["label"]) for r in rows]
    values = sorted(set(score for score, _ in scored))
    candidates = [0.0, 1.0]
    candidates += [(a + b) / 2 for a, b in zip(values, values[1:])]
    candidates += values

    best = None
    for threshold in candidates:
        metric_rows = [
            {"point": int(score >= threshold), "label": label,
             "agree": int(score >= threshold) == label, "subset": "dev"}
            for score, label in scored
        ]
        m = metrics(metric_rows)
        key = (m["kappa_overall"], m["agreement_overall"], -abs(threshold - 0.5))
        if best is None or key > best[0]:
            best = (key, threshold, m)
    grader.threshold = best[1]
    print(f"calibrated threshold={best[1]:.4f} on {len(rows)} development rows "
          f"(kappa={best[2]['kappa_overall']}, agreement={best[2]['agreement_overall']})")
    return {"threshold": best[1], "n": len(rows), "metrics": best[2]}


def run(args):
    items = load_gold(Path(args.gold))
    if args.subset != "all":
        items = [it for it in items if it["subset"] == args.subset]
    if args.limit:
        items = items[: args.limit]
    print(f"eval set: {len(items)} theses (subset={args.subset})")

    table, results = [], {}
    for kind in args.graders:
        condition, grader = _make(kind, args)
        calibration = None
        if kind == "specialist-calibrated":
            calibration = _calibrate(grader, Path(args.calibration_data))
        if isinstance(grader, DecomposedGrader):
            rows = _grade_decomposed(grader, items)
        else:
            workers = 1 if kind == "specialist" else 6
            rows = grade_all(grader, items, condition, workers=workers)
        m = metrics(rows)
        results[kind] = {"model": getattr(grader, "model", kind), "condition": condition,
                         "calibration": calibration, "metrics": m, "rows": rows}
        table.append((kind, m))
        if args.show_errors:
            print_rows(rows, f"{kind}", only_wrong=True)

    # comparison table
    print("\n" + "=" * 92)
    print(f"{'grader':22} {'n':>4} {'agree':>7} {'kappa':>7} {'false-DENY':>12} {'false-AWARD':>12}")
    print("-" * 92)
    for kind, m in table:
        fd = f"{m['false_denials']}"
        fa = f"{m['false_awards']}"
        print(f"{kind:22} {m['n_graded']:>4} {_p(m['agreement_overall']):>7} "
              f"{str(m['kappa_overall']):>7} {fd:>12} {fa:>12}")
    print("=" * 92)
    print("false-DENY = officially-earned theses wrongly scored 0 (the POV's target harm)")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2))
    with CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["grader", "n", "agreement", "kappa", "false_denials", "false_awards"])
        for kind, m in table:
            w.writerow([kind, m["n_graded"], m["agreement_overall"], m["kappa_overall"],
                        m["false_denials"], m["false_awards"]])
    print(f"wrote {RESULTS} and {CSV}")


def _p(x):
    return "  -  " if x is None else f"{round(x*100)}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graders", nargs="+",
                    default=["frontier-baseline", "frontier-hardened", "frontier-decomposed", "specialist"])
    ap.add_argument("--gold", default="data/rowa/gold_all.jsonl")
    ap.add_argument("--subset", default="sample", help="sample | rubric | handported | all")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--calibration-data", default="data/rowa/dev_v2.jsonl",
                    help="development set used only by specialist-calibrated")
    ap.add_argument("--model", default=None, help="frontier model override")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--show-errors", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
