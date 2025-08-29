#!/usr/bin/env python3
"""
Merge per-model prediction CSVs into wide CSVs for one or more window lengths.

It supports the filename patterns used in this project:
- Benchmarks:   *_predictions_{N}d.csv
- TSFMs:        *_predictions_ctx{N}.csv

For each file we keep the common keys:
  ["PERMNO", "DlyCalDt", "market_cap", "y_true"]
and rename 'y_pred' -> 'y_pred_<model_name>' where <model_name> is extracted
from the filename (e.g., amazon_chronos-bolt-tiny).

Optionally add an ensemble column (mean across all model predictions) and print quick metrics.
"""

from __future__ import annotations
from pathlib import Path
from functools import reduce
import argparse
import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# project constants
from src.config import (COL_DATE, COL_PERMNO, COL_MCAP)

Y_TRUE = "y_true"
Y_PRED = "y_pred"

# ----------------- helpers -----------------
def model_colname_from_filename(fp: Path) -> str:
    name = fp.stem  # without .csv
    m = re.search(r"(.+?)_predictions_ctx\d+$", name)
    if m: return m.group(1)
    m = re.search(r"(.+?)_predictions_\d+d$", name)
    if m: return m.group(1)
    return name

def read_and_prepare(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)

    # normalize date col -> DlyCalDt
    if COL_DATE in df.columns:
        pass
    elif "date" in df.columns:
        df = df.rename(columns={"date": COL_DATE})
    else:
        raise ValueError(f"{fp.name} is missing a date column (DlyCalDt or 'date').")

    required = {COL_PERMNO, COL_DATE, COL_MCAP, Y_TRUE, Y_PRED}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{fp.name} is missing columns: {missing}")

    model_name = model_colname_from_filename(fp)
    df = df[[COL_PERMNO, COL_DATE, COL_MCAP, Y_TRUE, Y_PRED]].copy()
    df = df.rename(columns={Y_PRED: f"y_pred_{model_name}"})
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    return df

def compute_quick_metrics(df: pd.DataFrame, pred_col: str) -> dict:
    y = df[Y_TRUE].to_numpy()
    yhat = df[pred_col].to_numpy()
    mask = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[mask], yhat[mask]
    if y.size == 0:
        return {"count": 0, "MSE": np.nan, "R2_OOS": np.nan, "Directional_Acc": np.nan}
    mse = float(mean_squared_error(y, yhat))
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_naive = float(np.sum(y ** 2))
    r2_oos = float(1 - ss_res / ss_naive) if ss_naive != 0 else np.nan
    acc = float(np.mean(np.sign(y) == np.sign(yhat)))
    return {"count": int(y.size), "MSE": mse, "R2_OOS": r2_oos, "Directional_Acc": acc}

def merge_one_window(window: int, folders: list[Path], out_dir: Path, make_ensemble: bool) -> None:
    # discover files
    patterns = [
        f"*_predictions_ctx{window}.csv",  # TSFMs
        f"*_predictions_{window}d.csv",    # benchmarks
    ]
    files: list[Path] = []
    for folder in folders:
        if folder.exists():
            for pat in patterns:
                files.extend(folder.glob(pat))

    if not files:
        print(f"[merge:{window}] No files found in {', '.join(map(str, folders))}")
        return

    print(f"[merge:{window}] Found {len(files)} file(s):")
    for f in files:
        print("   -", f)

    # read/standardize
    dfs = []
    for fp in files:
        try:
            dfs.append(read_and_prepare(fp))
        except Exception as e:
            print(f"[merge:{window}] Skipped {fp.name}: {e}")

    if not dfs:
        print(f"[merge:{window}] Nothing readable — skipping.")
        return

    # outer merge on keys
    keys = [COL_PERMNO, COL_DATE, COL_MCAP, Y_TRUE]
    merged = reduce(lambda L, R: pd.merge(L, R, on=keys, how="outer"), dfs)
    merged.sort_values([COL_DATE, COL_PERMNO], inplace=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"merged_window_{window}.csv"
    merged.to_csv(out_path, index=False)
    print(f"[merge:{window}] Saved -> {out_path}")

    if make_ensemble:
        pred_cols = [c for c in merged.columns if c.startswith("y_pred_")]
        if len(pred_cols) >= 2:
            merged["y_pred_ensemble"] = merged[pred_cols].mean(axis=1, skipna=True)
            cols_to_report = pred_cols + ["y_pred_ensemble"]
        else:
            cols_to_report = pred_cols

        print(f"\n[merge:{window}] Quick metrics:")
        for col in cols_to_report:
            m = compute_quick_metrics(merged, col)
            print(f"  {col}: count={m['count']}  MSE={m['MSE']:.6g}  R2_OOS={m['R2_OOS']:.6g}  DirAcc={m['Directional_Acc']:.4f}")

        out_path2 = out_dir / f"merged_window_{window}_with_ensemble.csv"
        merged.to_csv(out_path2, index=False)
        print(f"[merge:{window}] Saved -> {out_path2}")

# ----------------- main -----------------
def main():
    parser = argparse.ArgumentParser(description="Merge per-model prediction CSVs into wide files (one or more windows).")
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["results/chronos", "results/predictions", "results/uni2ts", "Predictions"],
        help="Folders to search for prediction files.",
    )
    # New: multiple windows
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        help="One or more window lengths, e.g. --windows 5 21 252 512",
    )
    # Backward-compat: single --window still accepted
    parser.add_argument(
        "--window",
        type=int,
        help="(Deprecated) single window length; prefer --windows.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("merged_results"),
        help="Output folder for merged CSVs.",
    )
    parser.add_argument(
        "--make-ensemble",
        action="store_true",
        help="Add y_pred_ensemble and print quick metrics.",
    )
    args = parser.parse_args()

    folders = [Path(f) for f in args.folders]
    out_dir = args.out

    # normalize windows list
    windows: list[int] = []
    if args.windows:
        windows = args.windows
    elif args.window is not None:
        windows = [args.window]
        print("[warn] --window is deprecated; use --windows.")
    else:
        # default to common four if none provided
        windows = [5, 21, 252, 512]
        print("[info] No window(s) specified; defaulting to 5 21 252 512.")

    for W in windows:
        merge_one_window(W, folders, out_dir, args.make_ensemble)

if __name__ == "__main__":
    main()