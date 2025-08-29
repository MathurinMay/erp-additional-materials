#!/usr/bin/env bash
set -euo pipefail

"""
Run Salesforce Uni2TS (Moirai-MoE) models:
- small and base variants

Output:
- Per-model prediction files : results/uni2ts
- Combined metrics : results/uni2ts/uni2ts_metrics.csv
"""

source .venv/bin/activate

OUTDIR="results/uni2ts"
mkdir -p "$OUTDIR"

python -m src.uni2ts \
  --data data/final_data.csv \
  --oos-start 2016-01-01 \
  --contexts 5 21 252 512 \
  --models Salesforce/moirai-moe-1.0-R-small Salesforce/moirai-moe-1.0-R-base \
  --device auto \
  --outdir "$OUTDIR"