# Scripts
This folder contains shell scripts that automate running all benchmark models and time-series foundation models (TSFMs). 

Before running for the first time, make sure the scripts are executable:
```bash
chmod +x scripts/*.sh
```

## Available Scripts
1. run_preprocessing.sh
: Preprocess raw CRSP and Fama–French data into final_data.csv.
2. run_benchmark_models.sh
: Runs all benchmark models (OLS, Lasso, Ridge, ElasticNet, GBRT, XGB, RF, PCR, PLS, GLM-Spline, NN1–NN5).
3. run_chronos.sh
: Runs Amazon Chronos models (Bolt Tiny/Mini, T5 Tiny/Mini) across context lengths.
4. run_timesfm_v1.sh
: Runs Google TimesFM v1.0 (200M, PyTorch port).
5. run_timesfm_v2.sh
: Runs Google TimesFM v2.0 (500M, PyTorch port).
6. run_uni2ts.sh
: Runs Salesforce Uni2TS (Moirai-MoE Small/Base).

## Example Usage:
```bash
./run_benchmark_models.sh
./run_chronos.sh
./run_timesfm_v1.sh
```