# Tools
This folder contains Python utilities for post-processing model outputs and portfolio evaluation.

1. merge_predictions.py
Merges per-model prediction CSVs into a wide format, one file per lag/context length.  
Also supports creating an ensemble forecast.
```bash
python -m tools.merge_predictions \
  --window 5 21 252 512 \
  --make-ensemble
```


2. load_sp500.py
Downloads S&P 500 excess returns and aggregate market capitalisation from WRDS,
saving it as data/sp500_excess_return_with_mcap.csv.
```bash
python tools/load_sp500.py \
    --out data/sp500_excess_return_with_mcap.csv
```

3. run_portfolios.py
Builds long–short portfolios (equal-weighted and value-weighted) from merged predictions.
Computes risk and return statistics including Sharpe ratio and cumulative log returns.
```bash
python tools/run_portfolios.py \
  --merged-dir merged_results \
  --sp500 data/sp500_excess_return_with_mcap.csv
```

Notes:
- These tools should be run after model predictions have been generated.
- Outputs is stored in merged_results/ and portfolio_outputs/