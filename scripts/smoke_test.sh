#!/usr/bin/env bash
# End-to-end smoke test (the Day-2 gate): 5 personas x ~20 examples through the FULL loop:
#   build_configs -> generate -> filter (+DPO) -> train (check) -> probes -> eval -> compare
#
# By default runs fully OFFLINE (no API key, no GPU) using the deterministic stubs, so it
# verifies wiring anywhere. Set DRY_RUN=0 to run for real (needs OPENAI_API_KEY; training
# still runs in --check mode unless you have a GPU).
#
# Usage:
#   bash scripts/smoke_test.sh            # offline wiring test
#   DRY_RUN=0 bash scripts/smoke_test.sh  # real teacher calls (costs tokens)
set -euo pipefail

cd "$(dirname "$0")/.."

DRY_RUN="${DRY_RUN:-1}"
N_PERSONAS="${N_PERSONAS:-5}"
PER_PERSONA="${PER_PERSONA:-20}"

if [[ "$DRY_RUN" == "1" ]]; then
  DRY="--dry-run"
  echo "### SMOKE TEST (OFFLINE stub mode) ###"
else
  DRY=""
  echo "### SMOKE TEST (REAL teacher; needs OPENAI_API_KEY) ###"
fi

# Activate venv if present.
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

SMK=data/smoke
mkdir -p "$SMK"

echo
echo "== [1/6] Build persona configs =="
python -m src.build_configs $DRY

echo
echo "== [2/6] Generate conversations ($N_PERSONAS personas x ~$PER_PERSONA) =="
python -m src.generate \
  --personas configs/personas_train.yaml \
  --per-persona "$PER_PERSONA" --limit "$N_PERSONAS" \
  --out "$SMK/raw.jsonl" $DRY

echo
echo "== [3/6] Judge-filter -> SFT + DPO =="
python -m src.build_dataset \
  --in "$SMK/raw.jsonl" --out "$SMK/train.jsonl" \
  --dpo "$SMK/dpo.jsonl" --reject-log "$SMK/rejected.jsonl" $DRY

echo
echo "== [4/6] Train dataset check (no GPU needed) =="
python -m src.train --data "$SMK/train.jsonl" --check

echo
echo "== [5/6] Build probe battery (eval personas) =="
python -m src.probes \
  --personas configs/personas_eval.yaml \
  --out "$SMK/probes.jsonl" $DRY

echo
echo "== [6/6] Eval (base-vs-tuned wiring) + compare =="
python -m src.eval run --responder dry \
  --personas configs/personas_eval.yaml \
  --probes "$SMK/probes.jsonl" --label base --out "$SMK/base.json" --dry-run
python -m src.eval run --responder dry \
  --personas configs/personas_eval.yaml \
  --probes "$SMK/probes.jsonl" --label tuned --out "$SMK/tuned.json" --dry-run
python -m src.eval compare --base "$SMK/base.json" --tuned "$SMK/tuned.json" \
  --csv "$SMK/results_table.csv"

echo
echo "### SMOKE TEST COMPLETE — full loop ran end to end. Artifacts in $SMK/ ###"
