"""Path bootstrap so scripts run both as `python -m src.x` and `python src/x.py`."""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CONFIGS_DIR = os.path.join(REPO_ROOT, "configs")
DATA_DIR = os.path.join(REPO_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
FILTERED_DIR = os.path.join(DATA_DIR, "filtered")
EVAL_DIR = os.path.join(DATA_DIR, "eval")

for _d in (RAW_DIR, FILTERED_DIR, EVAL_DIR):
    os.makedirs(_d, exist_ok=True)
