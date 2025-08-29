import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.config import (
    COL_DATE, 
    COL_MCAP, 
)

def evaluate_predictions(y_true, y_pred, df=None, col_date=COL_DATE, col_mcap=COL_MCAP):
    """
    Evaluate predictions with statistical metrics.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Statistical metrics
    mse = mean_squared_error(y_true, y_pred)
    ss_res   = float(np.sum((y_true - y_pred) ** 2))
    ss_naive = float(np.sum(y_true ** 2))
    r2_oos   = 1.0 - ss_res / ss_naive if ss_naive != 0 else np.nan

    acc = float((np.sign(y_true) == np.sign(y_pred)).mean())
    up_mask   = y_true > 0
    down_mask = y_true < 0
    acc_up   = float((np.sign(y_pred[up_mask])   ==  1).mean()) if up_mask.any() else np.nan
    acc_down = float((np.sign(y_pred[down_mask]) == -1).mean()) if down_mask.any() else np.nan

    # optional
    top5_acc = bot5_acc = np.nan
    if df is not None and col_date in df and col_mcap in df:
        df = df.copy()
        df["y_true"] = y_true
        df["y_pred"] = y_pred
        df["rank"]   = df.groupby(col_date)[col_mcap].rank(ascending=False, method="first")
        max_rank = df.groupby(col_date)["rank"].transform("max")

        top5_m  = df["rank"] <= 5
        bot5_m  = df["rank"] >= (max_rank - 5 + 1)

        if top5_m.any():
            top5_acc = float((np.sign(df.loc[top5_m, "y_pred"]) == np.sign(df.loc[top5_m, "y_true"])).mean())
        if bot5_m.any():
            bot5_acc = float((np.sign(df.loc[bot5_m, "y_pred"]) == np.sign(df.loc[bot5_m, "y_true"])).mean())

    return {
        "R2_OOS": round(r2_oos, 4),
        "MSE": round(mse, 6),
        "Directional_Accuracy": round(acc, 4),
        "Upward_Accuracy":  round(acc_up, 4)   if not np.isnan(acc_up) else np.nan,
        "Downward_Accuracy":round(acc_down, 4) if not np.isnan(acc_down) else np.nan,
        "Top5_Accuracy":   round(top5_acc,4) if not np.isnan(top5_acc) else np.nan,
        "Bottom5_Accuracy":round(bot5_acc,4) if not np.isnan(bot5_acc) else np.nan,
    }