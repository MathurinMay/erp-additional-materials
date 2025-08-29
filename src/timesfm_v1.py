#!/usr/bin/env python3
"""
TimesFM v1.0 runner for your ERP project.

- Loads the same panel (data/final_data.csv)
- Builds rolling contexts per PERMNO
- Predicts one-step-ahead excess return using TimesFM v1.0
- Evaluates with the shared src.metrics.evaluate_predictions
- Saves per-point predictions and a combined metrics CSV

CLI examples:
  python -m src.timesfm_v1
  python -m src.timesfm_v1 --windows 5 21 --model google/timesfm-1.0-200m-pytorch
  python -m src.timesfm_v1 --outdir results/timesfm --device auto
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import argparse

import numpy as np
import pandas as pd
import torch
import timesfm

from tqdm import tqdm

from src.config import (
    COL_DATE, 
    COL_PERMNO, 
    COL_EXCESS, 
    COL_MCAP, 
    COL_PRICE, 
    COL_SHROUT,
    DEFAULT_DATA, 
    DEFAULT_OOS, 
    PRED_DIR
)
from src.metrics import evaluate_predictions

# TimesFM 1.0 (PyTorch port)
from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint


def msg(s: str) -> None:
    print(f"[timesfm-v1] {s}")


def ensure_mcap(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure market_cap exists; compute |price| * shrout * 1000 if missing."""
    out = df.copy()
    if COL_MCAP not in out.columns:
        out[COL_MCAP] = out[COL_PRICE].abs() * out[COL_SHROUT] * 1000.0
        msg("Computed market_cap from DlyPrc*ShrOut*1000.")
    return out


def build_oos_contexts(
    df: pd.DataFrame, history_len: int, oos_start: pd.Timestamp
) -> Tuple[List[torch.Tensor], List[float], List[dict]]:
    """
    Build (contexts, targets, records) for the OOS period only.
      - context: last `history_len` values of excess return
      - target:  next-step realized excess return
      - records: metadata (PERMNO, date, market_cap)
    """
    df = df.sort_values([COL_PERMNO, COL_DATE]).copy()
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])

    contexts: List[torch.Tensor] = []
    targets:  List[float] = []
    records:  List[dict] = []

    for permno, grp in df.groupby(COL_PERMNO, sort=False):
        values = grp[COL_EXCESS].to_numpy()
        dates  = grp[COL_DATE].to_numpy()
        mcaps  = grp[COL_MCAP].to_numpy()
        n = len(values)
        if n <= history_len:
            continue

        for i in range(n - history_len):
            t_idx  = i + history_len
            t_date = dates[t_idx]
            if t_date >= oos_start:
                ctx = torch.tensor(values[i:t_idx], dtype=torch.float32)  # shape [history_len]
                contexts.append(ctx)
                targets.append(float(values[t_idx]))
                records.append({
                    COL_PERMNO: permno,
                    COL_DATE:   t_date,
                    COL_MCAP:   float(mcaps[t_idx]),
                })
    return contexts, targets, records


def pick_device(opt: str = "auto") -> str:
    """Choose execution device: 'mps' (Apple), 'cuda', or 'cpu'."""
    if opt != "auto":
        return opt
    try:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def pick_context_len(history_len: int) -> int:
    """
    Map your research lag window to TimesFM's preferred context sizes.
    - ≤ 21  → 32
    - ≤ 256 → 256
    - else  → 512
    """
    if history_len <= 21:
        return 32
    elif history_len <= 256:
        return 256
    else:
        return 512


def run_one(
    data_path: Path,
    model_name: str,
    lag_window: int,
    oos_start: str | pd.Timestamp,
    device_str: str,
    outdir: Path,
) -> pd.DataFrame:
    """
    Run TimesFM v1.0 for one lag window:
      - history_len is derived via pick_context_len(lag_window)
      - returns one-row metrics DataFrame
    """
    history_len = pick_context_len(lag_window)
    device = pick_device(device_str)

    # Load data
    msg(f"Loading data: {data_path}")
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Cannot find data file: {data_path}")
    df = pd.read_csv(data_path)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    df = ensure_mcap(df)

    # Build contexts
    oos_ts = pd.Timestamp(oos_start)
    contexts, targets, records = build_oos_contexts(df, history_len, oos_ts)
    if not contexts:
        raise ValueError("No OOS contexts created. Check dates/history length/data coverage.")

    # Load TimesFM
    msg(f"Loading TimesFM v1.0: {model_name} (device={device}, context_len={history_len})")
    tfm = TimesFm(
        hparams=TimesFmHparams(
            backend=device,          # "cpu" | "cuda" | "mps"
            context_len=history_len, # model's expected history size
            horizon_len=1,
        ),
        checkpoint=TimesFmCheckpoint(huggingface_repo_id=model_name),
    )
    # (Torch backend safety) try move model
    try:
        tfm._model.to(device)  # may be no-op for non-torch backend
    except Exception:
        pass

    # Inference
    preds = []
    freqs = [0] * len(contexts)  # dummy freq per API
    msg(f"Inference on {len(contexts)} contexts...")
    with torch.no_grad():
        # Attempt batched path
        try:
            batch = torch.stack(contexts).to(device)  # [B, L]
            pred, _ = tfm.forecast(batch, freq=freqs)  # -> [B, 1]
            pred = pred.squeeze(-1) if hasattr(pred, "ndim") and pred.ndim > 1 else pred
            preds = pred.detach().cpu().numpy().tolist()
        except Exception:
            # Fallback: loop
            for ctx in tqdm(contexts, desc=f"TimesFM v1.0 ctx={history_len}"):
                p, _ = tfm.forecast([ctx], freq=[0])
                p = p.squeeze() if hasattr(p, "squeeze") else p[0]
                preds.append(float(p))

    # Assemble results
    results = pd.DataFrame.from_records(records)
    results["y_true"] = np.array(targets, dtype=float)
    results["y_pred"] = np.array(preds,   dtype=float)

    # Metrics (shared helper)
    metrics = evaluate_predictions(
        y_true=results["y_true"].to_numpy(),
        y_pred=results["y_pred"].to_numpy(),
        df=results, col_date=COL_DATE, col_mcap=COL_MCAP
    )

    # Save per-point predictions
    outdir.mkdir(parents=True, exist_ok=True)
    safe_model = model_name.replace("/", "_")
    out_csv = outdir / f"{safe_model}_predictions_ctx{lag_window}.csv"
    results.to_csv(out_csv, index=False)
    msg(f"Saved predictions -> {out_csv}")

    # One-row summary
    return pd.DataFrame([{
        "Model": model_name,
        "Lag_Window": int(lag_window),
        "Context_Len_Used": int(history_len),
        **metrics,
    }])


def parse_args():
    p = argparse.ArgumentParser(description="Run TimesFM v1.0 across research lag windows.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to preprocessed data CSV")
    p.add_argument("--oos-start", type=str, default=str(DEFAULT_OOS.date()), help="OOS start date (YYYY-MM-DD)")
    p.add_argument("--contexts", type=int, nargs="+", default=[5, 21, 252, 512],
                   help="Research lag windows to evaluate")
    p.add_argument("--model", type=str, default="google/timesfm-1.0-200m-pytorch",
                   help="TimesFM v1.0 model id (PyTorch port)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"],
                   help="Execution device")
    p.add_argument("--outdir", type=Path, default=Path("results/timesfm_v1"),
                   help="Where to save predictions and the combined metrics CSV")
    return p.parse_args()


def main():
    ns = parse_args()
    summaries: list[pd.DataFrame] = []
    for L in ns.contexts:
        msg(f"Context length: {L}")
        summary = run_one(
            data_path=ns.data,
            model_name=ns.model,
            lag_window=L,
            oos_start=ns.oos_start,
            device_str=ns.device,
            outdir=ns.outdir,
        )
        summaries.append(summary)

    # Save combined metrics
    metrics = pd.concat(summaries, ignore_index=True)
    ns.outdir.mkdir(parents=True, exist_ok=True)
    out_csv = ns.outdir / "timesfm_v1_metrics.csv"
    metrics.to_csv(out_csv, index=False)
    msg(f"Saved combined metrics -> {out_csv}")


if __name__ == "__main__":
    main()
    