#!/usr/bin/env bash
set -euo pipefail

"""
Run all benchmark ML models:
- Linear model: OLS, Lasso, Ridge, ElasticNet
- Dimension-Reduction method: PCR, PLS
- Tree-based model: Random Forest, GBRT, XGBoost
- Generalised Linear model: GLM-Spline
- Neural network: NN1–NN5

Output:
- Per-model prediction files : results/predictions
- Combined metrics : results/predictions/benchmark_metrics.csv
"""

# Run all benchmark models over default horizons (5,21,252,512)
python -m src.benchmark_models \
  --data data/final_data.csv \
  --outdir results/predictions \
  --oos-start 2016-01-01