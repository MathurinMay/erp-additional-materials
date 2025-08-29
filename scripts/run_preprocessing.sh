#!/usr/bin/env bash
set -euo pipefail

"""
Preprocessing pipeline
- Reads CRSP + Fama-French raw CSVs
- Filters for healthcare SIC codes
- Selects top-50 PERMNOs by market cap
- Computes daily excess returns
- Saves final dataset : data/final_data.cs
"""

echo "[RUNNING] Preprocessing..."
python -m src.preprocessing

echo "[DONE]"