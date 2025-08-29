#!/usr/bin/env bash
set -euo pipefail

"""
Run Google TimesFM v1.0 (200M, PyTorch port)

Output:
- Per-model prediction files : results/timesfm_v1
- Combined metrics : results/timesfm_v1/timesfm_v1_metrics.csv
"""

# Load conda into the shell
source /opt/anaconda3/etc/profile.d/conda.sh

# Activate the tfm-env environment
conda activate tfm-env

OUTDIR="results/timesfm_v1"
mkdir -p "$OUTDIR"

python -m src.timesfm_v1 \
  --data data/final_data.csv \
  --oos-start 2016-01-01 \
  --contexts 5 21 252 512 \
  --model google/timesfm-1.0-200m-pytorch \
  --device auto \
  --outdir "$OUTDIR"