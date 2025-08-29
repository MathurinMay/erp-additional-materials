#!/usr/bin/env bash
set -euo pipefail

"""
Run Amazon Chronos TSFMs:
- chronos-bolt-tiny, chronos-bolt-mini
- chronos-t5-tiny,   chronos-t5-mini

Output:
- Per-model prediction files : results/chronos
- Combined metrics : results/chronos/chronos_metrics
"""

source .venv/bin/activate

OUTDIR="results/chronos"
mkdir -p "$OUTDIR"

python -m src.chronos \
  --data data/final_data.csv \
  --oos-start 2016-01-01 \
  --contexts 5 21 252 512 \
  --models amazon/chronos-bolt-tiny amazon/chronos-bolt-mini amazon/chronos-t5-tiny amazon/chronos-t5-mini \
  --device auto \
  --outdir "$OUTDIR" \
  "$@"