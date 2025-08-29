"""
Preprocess CRSP + Fama-French to produce the modelling dataset for the ERP project.

Steps:
1) Read raw CSVs (not distributed): CRSP daily and FF daily risk-free rate
2) Filter Healthcare by SIC ranges: 2830-2836, 3840-3851, and 8000-8099
3) Split in-sample (From 2000-01-01 to 2015-12-31) and out-of-sample (From 2016-01-01 to 2024-12-31)
4) Select top-50 PERMNOs by average market cap over estimation period
5) Merge RF, compute daily excess returns (DlyRet - rf)
6) Clip outliers at 1st/99th percentiles (on excess_ret)
7) Save final dataset to data/final_data.csv
8) Print brief summary stats (for EDA)

Usage:
    python src/preprocessing.py \
        --crsp data/crsp_dataset.csv \
        --ff   data/riskfree_rate.csv \
        --out  data/final_data.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from src.config import (
    COL_DATE, 
    COL_PERMNO, 
    COL_SIC, 
    COL_PRICE, 
    COL_RETURN, 
    COL_SHROUT,
    COL_RF, 
    COL_EXCESS,
    COL_MCAP,
    EST_START, EST_END, 
    OOS_START, OOS_END,
    DEFAULT_DATA,
    HEALTHCARE_SIC,
    TOP_N,
    CLIP_LOW, CLIP_HIGH,
    CRSP_PATH, FF_PATH
)

# Settings
OUT_PATH  = DEFAULT_DATA

# Functions
def msg(x): print(f"[preprocessing] {x}")

def read_crsp(path: Path) -> pd.DataFrame:
    """
    Read CRSP daily dataset from CSV.
    Ensures required columns exist and converts date column to datetime.
    """
    msg(f"Reading CRSP: {path}")
    df = pd.read_csv(path)
    needed = {COL_PERMNO,COL_SIC,COL_DATE,COL_PRICE,COL_RETURN,COL_SHROUT}
    miss = needed - set(df.columns)
    if miss: raise ValueError(f"CRSP missing columns: {sorted(miss)}")
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    return df

def read_ff(path: Path) -> pd.DataFrame:
    """
    Read Fama-French daily risk-free CSV.
    Original column is 'date', renamed to COL_DATE ('DlyCalDt') for consistency with CRSP.
    """
    msg(f"Reading Fama-French RF: {path}")
    ff = pd.read_csv(path)
    needed = {"date",COL_RF}
    miss = needed - set(ff.columns)
    if miss: raise ValueError(f"FF missing columns: {sorted(miss)}")
    ff["date"] = pd.to_datetime(ff["date"])
    ff = ff.rename(columns={"date":COL_DATE})
    return ff[[COL_DATE,COL_RF]]

def filter_healthcare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter healthcare sector by SIC codes and drop missing and duplicate rows.
    """
    mask = pd.Series(False, index=df.index)
    for lo, hi in HEALTHCARE_SIC:
        mask |= df[COL_SIC].between(lo, hi)
    out = df[mask].copy()

    before = len(out)
    out = out.dropna().drop_duplicates()
    removed = before - len(out)
    msg(f"After SIC filter and clean: {len(out):,} rows (removed {removed:,})")
    return out

def enforce_common_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only PERMNOs that have observations on every trading day.
    """
    msg("Enforcing common trading calendar...")
    # build union calendar
    cal = np.sort(df[COL_DATE].unique())
    expected = len(cal)

    counts = df.groupby(COL_PERMNO)[COL_DATE].nunique()
    valid_ids = counts[counts == expected].index.tolist()
    removed_ids = counts.index.difference(valid_ids).tolist()
    msg(f"Total trading days: {expected}")
    msg(f"Valid stocks: {len(valid_ids)} | Removed (insufficient coverage): {len(removed_ids)}")
    out = df[df[COL_PERMNO].isin(valid_ids)].reset_index(drop=True)
    return out

def top_by_mcap(df: pd.DataFrame) -> list[int]:
    """
    Select top-N PERMNOs by average market cap over the estimation window.
    """
    est = df[(df[COL_DATE]>=EST_START) & (df[COL_DATE]<=EST_END)].copy()
    est[COL_MCAP] = est[COL_PRICE].abs()*est[COL_SHROUT]*1000.0
    ids = est.groupby(COL_PERMNO)[COL_MCAP].mean().sort_values(ascending=False).head(TOP_N).index.tolist()
    msg(f"Selected top {len(ids)} PERMNOs by avg mcap in {EST_START} .. {EST_END}")
    return ids

def add_excess_returns(df: pd.DataFrame, ff: pd.DataFrame) -> pd.DataFrame:
    """
    Merge with risk-free rate and compute excess returns.
    """
    m = df.merge(ff, on=COL_DATE, how="left")
    if m[COL_RF].isna().any():
        msg("Missing rf detected; forward-filling.")
        m[COL_RF] = m[COL_RF].ffill()
    m[COL_EXCESS] = m[COL_RETURN] - m[COL_RF]
    return m

def clip_excess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip excess returns to specified quantiles to handle outliers.
    """
    ql = df[COL_EXCESS].quantile(CLIP_LOW)
    qh = df[COL_EXCESS].quantile(CLIP_HIGH)
    out = df.copy()
    out[COL_EXCESS] = out[COL_EXCESS].clip(ql, qh)
    msg(f"Clipped excess_ret to [{ql:.6f}, {qh:.6f}] (quantiles {CLIP_LOW:.0%}/{CLIP_HIGH:.0%})")
    return out

def summarize(df: pd.DataFrame) -> None:
    """
    Print descriptive statistics of excess returns for in-sample and out-of-sample periods.
    """
    ins = df[(df[COL_DATE]>=EST_START) & (df[COL_DATE]<=EST_END)][COL_EXCESS]
    oos = df[(df[COL_DATE]>=OOS_START) & (df[COL_DATE]<=OOS_END)][COL_EXCESS]
    msg("In-sample describe():"); print(ins.describe())
    msg("Out-of-sample describe():"); print(oos.describe())

def run() -> None:
    """
    Main execution pipeline:
    1. Load CRSP and FF data
    2. Filter healthcare SIC and clean
    3. Enforce common calendar
    4. Select top-N by market capitalisation
    5. Merge risk-free and compute excess returns
    6. Clip outliers
    7. Save final dataset
    """
    crsp = read_crsp(CRSP_PATH)
    ff   = read_ff(FF_PATH)
    hc   = filter_healthcare(crsp)
    hc   = enforce_common_calendar(hc)

    ids  = top_by_mcap(hc)
    panel = hc[hc[COL_PERMNO].isin(ids)].copy()

    panel = add_excess_returns(panel, ff)
    panel = clip_excess(panel)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_PATH, index=False)
    msg(f"Saved final dataset: {OUT_PATH} ({len(panel):,} rows)")

    summarize(panel)
    msg("Done.")

if __name__ == "__main__":
    run()