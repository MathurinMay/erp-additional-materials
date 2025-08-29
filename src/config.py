from pathlib import Path
import pandas as pd

# Column names
COL_DATE     = "DlyCalDt"
COL_PERMNO   = "PERMNO"
COL_SIC      = "SICCD"
COL_PRICE    = "DlyPrc"
COL_RETURN   = "DlyRet"
COL_SHROUT   = "ShrOut"
COL_RF       = "rf"
COL_EXCESS   = "excess_ret"
COL_MCAP     = "market_cap"

HEALTHCARE_SIC = [(2830,2836), # pharmaceuticals
                  (3840,3851), # medical instruments
                  (8000,8099)] # health services
TOP_N = 50
CLIP_LOW, CLIP_HIGH = 0.01, 0.99

# Default paths
DATA_DIR = Path("data")
RESULT_DIR = Path("results")
PRED_DIR = Path("Predictions")

DEFAULT_DATA   = DATA_DIR / "final_data.csv"
DEFAULT_OUTDIR = RESULT_DIR / "predictions"
CRSP_PATH =  DATA_DIR / "crsp_dataset.csv"
FF_PATH   = DATA_DIR / "riskfree_rate.csv"

EST_START, EST_END = "2000-01-01", "2015-12-31"
OOS_START, OOS_END = "2016-01-01", "2024-12-31"
DEFAULT_OOS    = pd.Timestamp(OOS_START)
