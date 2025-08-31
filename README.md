# Exploring Advanced Financial Applications of Large Language Models

## Project Overview
This repository benchmarks a wide range of models for daily excess return forecasting.
It covers both **traditional ML models** and **time-series foundation models (TSFMs)**:

- **Linear / Regularised Models:** OLS, Ridge, Lasso, ElasticNet
- **Generalised Linear Models:** GLM-Spline
- **Dimension Reduction:** Principal Component Regression (PCR), Partial Least Squares (PLS)
- **Tree-Based Models:** Random Forest, Gradient Boosted Regression Trees (GBRT), XGBoost
- **Neural Networks:** NN1–NN5 (shallow networks)
- **Time-Series Foundation Models (TSFMs):** Chronos (Bolt/T5), TimesFM, Uni2TS (Moirai)

---

## Research Context
The empirical setting focuses on the **U.S. healthcare sector**, specifically the top 50 firms ranked by market capitalisation.  

- **Data Source:** CRSP (daily stock prices and shares outstanding), Fama-French risk-free rate, and S&P 500 index returns via WRDS
- **Estimation Period:** 2000-01-01 to 2015-12-31  
- **Out-of-Sample Period:** 2016–01-01 to 2024-12-31   
- **Target Variable:** Daily excess return  

The research questions are:  
1.	Can TSFMs, namely Chronos, TimesFM, and Uni2TS, outperform a group of benchmark models in predicting daily excess returns for healthcare stocks?
2.	How does predictive accuracy vary across different historical time windows (5, 21, 252, and 512 trading days)?
3.	How consistently do TSFMs produce accurate predictions over time, and to what extent are they prone to overfitting or look-ahead bias?
4.	How effective are model predictions when used to construct portfolios, and how do returns compare across models?

This Extended Research Project (ERP) was undertaken as part of the MSc Data Science program at the University of Manchester.

---

## End-to-End Pipeline

To reproduce the results, follow these steps in order:

### 0) Environment Setup

The repository uses separate requirement files for reproducibility.

- **Benchmarks and other TSFMs (Chronos and Uni2TS):**
  ```
  conda create -n venv python=3.11 -y
  conda activate venv
  pip install -r requirements.txt --no-cache-dir
  ```
- **TimesFM (Torch backend):**
  ```
  conda create -n tfm-env python=3.11 -y
  conda activate tfm-env
  pip install -r requirements-timesfm.txt --no-cache-dir
  ```

### 1) Data Availability 

Due to licensing restrictions, the dataset (CRSP/WRDS) **cannot be provided in this repository**.  
To run the experiments, users must obtain access to WRDS and prepare the following input files:

1. `crsp_dataset.csv`  
  Contains the stock-level panel with the following required columns:
    - `PERMNO` : Unique stock identifier  
    - `SICCD` : SIC industry classification code  
    - `DlyCalDt` : Trading date (YYYY-MM-DD)  
    - `DlyPrc` : Daily stock price  
    - `DlyRet` : Daily stock return  
    - `ShrOut` : Shares outstanding

2. `riskfree_rate.csv`  
  Contains the Fama-French daily risk-free rate with the following columns:
    - `DlyCalDt` : Trading date (YYYY-MM-DD)  
    - `rf`   : Daily risk-free rate

3. `sp500_excess_return_with_mcap.csv`
   S&P500 used as a benchmark in portfolio evaluation, with the following columns:  
     - `DlyCalDt` : Trading date (YYYY-MM-DD)  
     - `excess_ret` : Daily excess return of S&P 500  
     - `mcap` : Aggregate market capitalization

**Important:**  
- Place these files inside the `data/` directory.
- `crsp_dataset.csv` and `riskfree_rate.csv` must be downloaded from WRDS
- `sp500_excess_return_with_mcap.csv` can be generated directly from WRDS via the provided script:  
   ```
   python tools/load_sp500.py --out data/sp500_excess_return_with_mcap.csv
   ```

### 2) Preprocess

Preprocessing scripts (in `src/preprocessing.py`) will clean CRSP data, filter by SIC codes for the healthcare sector, merge with Fama-French data, compute daily excess returns, and generate the final analysis dataset (`final_data.csv`).
```
python -m src.preprocessing \
  --crsp data/crsp_dataset.csv \
  --ff   data/riskfree_rate.csv \
  --out  data/final_data.csv
```

### 3) Modelling, Evaluation, and Example Usage

You can run all benchmark models and TSFMs using the provided shell scripts:
```
scripts/run_benchmark_models.sh
scripts/run_chronos.sh
scripts/run_timesfm_v1.sh
scripts/run_timesfm_v2.sh
scripts/run_uni2ts.sh
```

Alternatively, each model can be run individually. For example:
- Benchmark (Lasso) across multiple lags:
```
python -m src.benchmarks.lasso \
  --data data/final_data.csv \
  --oos-start 2016-01-01 \
  --lags 5 21 252 512 \
  --outdir results/benchmarks
```
- TimesFM v1.0 with context length = 5:
```
python -m src.timesfm_v1 \
  --data data/final_data.csv \
  --oos-start 2016-01-01 \
  --contexts 5 \
  --model google/timesfm-1.0-200m-pytorch \
  --device auto \
  --outdir results/timesfm_v1
```
Outputs:
- Predictions and metrics will appear in `results/ subfolders.`
- Each model produces a per-model prediction file:
  - TSFMs: `*_predictions_ctx{N}.csv` (e.g., amazon_chronos-bolt-tiny_predictions_ctx252.csv)
  - Benchmarks: `*_predictions_{N}d.csv` (e.g., lasso_predictions_252d.csv)

### 4) Merge Predictions

Combine per-model CSVs into one file per window:
```
python -m tools.merge_predictions \
  --window 5 21 252 512 \
  --make-ensemble
```
Output appears in `merged_results/.`

### 5) Portfolio Evaluation

Build long-short portfolios (EW & VW, with & without transaction costs) using merged files:
```
python tools/run_portfolios.py \
  --merged-dir merged_results \
  --sp500 data/sp500_excess_return_with_mcap.csv
```
Results are saved in `portfolio_outputs/.`


---

## Evaluation Framework

The project evaluates forecasts using two approaches:

1. **Statistical Evaluation**
   - Out-of-sample R²
   - Mean Square Error
   - Directional Accuracy and Upward/Downward Accuracy
   - Top/Bottom-10% Accuracy by market capitalisation
     
2. **Economic Evaluation**
   - Portfolio construction based on model forecasts
   - Long-short portfolio strategies
   - Equal-weighted and value-weighted portfolios both with and without transaction costs
   - Performance measured via Sharpe ratio and Cumulative log returns

---

## Repository Structure 
```
data/                   # Input datasets (must be user-provided or generated)
  crsp_dataset.csv
  riskfree_rate.csv
  sp500_excess_return_with_mcap.csv

src/                    # Source code
  config.py             # Column constants & paths
  preprocessing.py      # Preprocessing steps
  metrics.py            # Evaluation metrics (R², MSE, accuracy, etc.)
  benchmark_models.py   # Traditional ML models (OLS, Lasso, Ridge, RF, etc.)
  chronos.py            # Chronos Bolt (Tiny/Mini) and T5 (Tiny/Mini)
  timesfm_v1.py         # TimesFM 1.0 200M
  timesfm_v2.py         # TimesFM 2.0 500M
  uni2ts.py             # Uni2TS (Moirai-MoE Small/Base)

scripts/
  run_benchmark_models.sh  # Run all benchmark models (OLS, Lasso, RF, etc.)
  run_chronos.sh           # Run Chronos models (Bolt/T5)
  run_timesfm_v1.sh        # Run TimesFM v1.0 (200M)
  run_timesfm_v2.sh        # Run TimesFM v2.0 (500M)
  run_uni2ts.sh            # Run Uni2TS (Moirai-MoE Small/Base)

requirements/              # Environment specs
  requirements.txt         # Environment for Benchmark models and TSFMs (Chronos and Uni2TS)
  requirements-timesfm.txt # Environment for TimesFM

tools/                  # Helper scripts
  merge_predictions.py  # Merge predictions across models (same window length)
  load_sp500.py         # Fetch S&P500 benchmark via WRDS
  run_portfolios.py     # Portfolio construction

README.md
```
