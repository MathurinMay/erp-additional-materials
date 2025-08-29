"""
Run Chronos Time-Series Foundation Models as benchmarks.

- Loads the same panel (data/final_data.csv) as the classic benchmarks
- Builds rolling contexts of length L per PERMNO
- Predicts next-step excess return with Chronos models
- Evaluates OOS (from 2016-01-01) with MSE, R2_OOS, directional accuracies, top/bottom-5 by mcap
- Saves per-model predictions and a combined metrics table
"""

from __future__ import annotations
from pathlib import Path
from typing import List
import argparse

import numpy as np
import pandas as pd
import torch
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
    PRED_DIR,
)
from src.metrics import evaluate_predictions

# Chronos pipeline
from chronos import BaseChronosPipeline

def msg(s: str) -> None:
    print(f"[chronos] {s}")

def ensure_mcap(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure market_cap exists; compute |price| * shrout * 1000 if missing."""
    out = df.copy()
    if COL_MCAP not in out.columns:
        out[COL_MCAP] = out[COL_PRICE].abs() * out[COL_SHROUT] * 1000.0
        msg("Computed market_cap from DlyPrc*ShrOut*1000.")
    return out


def build_oos_contexts(
    df: pd.DataFrame,
    context_len: int,
    oos_start: pd.Timestamp,
) -> tuple[list[torch.Tensor], list[float], list[dict]]:
    """
    Build (contexts, targets, records) for the out-of-sample (OOS) period only.

    - context: the rolling history of excess returns of length L used by the TSFMs
    - target:  the next-day (one-step-ahead) realized excess return
    - records: metadata for each target point (PERMNO, target date, market_cap)
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

        if len(values) <= context_len:
            continue

        # Rolling windows: history [i .. i+L-1] -> predict the value at i+L
        for i in range(len(values) - context_len):
            target_idx  = i + context_len
            target_date = dates[target_idx]
            if target_date >= oos_start:
                ctx = torch.tensor(values[i : i + context_len], dtype=torch.float32)
                tgt = float(values[target_idx])
                contexts.append(ctx)
                targets.append(tgt)
                records.append({
                    COL_PERMNO: permno,
                    COL_DATE:   target_date,
                    COL_MCAP:   float(mcaps[target_idx]),
                })

    return contexts, targets, records


def pick_device(opt: str) -> str:
    """Choose execution device."""
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


def run_one_model(
    data_path: Path,
    model_name: str,
    context_len: int,
    oos_start: str | pd.Timestamp,
    device: str,
    outdir: Path,
) -> pd.DataFrame:
    """
    Run one Chronos model with a given context length:
    - Load the panel
    - Build rolling contexts for the OOS period
    - Predict one step ahead
    - Compute evaluation metrics and write predictions to disk

    Returns a one-row DataFrame of summary metrics.
    """
    msg(f"Loading data: {data_path}")
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Cannot find data file: {data_path}")

    df = pd.read_csv(data_path)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    df = ensure_mcap(df)

    # Prepare OOS contexts
    oos_ts = pd.Timestamp(oos_start)
    contexts, targets, records = build_oos_contexts(df, context_len=context_len, oos_start=oos_ts)

    if len(contexts) == 0:
        raise ValueError("No OOS contexts created. Check dates/context length/data coverage.")

    # Load Chronos model
    msg(f"Loading Chronos pipeline: {model_name} (device={device})")
    pipeline = BaseChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    )

    # Inference (one context at a time for simplicity and stability)
    preds: List[float] = []
    for ctx in tqdm(contexts, desc=f"{model_name} Context length={context_len}"):
        forecast = pipeline.predict(context=ctx.unsqueeze(0), prediction_length=1)
        preds.append(float(forecast[0][0]))

    # Results
    results = pd.DataFrame.from_records(records)
    results["y_true"] = np.array(targets, dtype=float)
    results["y_pred"] = np.array(preds,   dtype=float)

    # Metrics 
    metrics = evaluate_predictions(
        y_true=results["y_true"].to_numpy(),
        y_pred=results["y_pred"].to_numpy(),
        df=results,          
        col_date=COL_DATE,
        col_mcap=COL_MCAP,
    )

    # Save predictions
    outdir.mkdir(parents=True, exist_ok=True)
    safe_model = model_name.replace("/", "_")
    out_csv = outdir / f"{safe_model}_predictions_ctx{context_len}.csv"
    results.to_csv(out_csv, index=False)
    msg(f"Saved predictions -> {out_csv}")

    return pd.DataFrame([{"Model": model_name, "Context_Length": int(context_len), **metrics}])


def parse_args():
    p = argparse.ArgumentParser(description="Run Chronos TSFMs across multiple context lengths.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--oos-start", type=str, default=str(DEFAULT_OOS.date()))
    p.add_argument("--contexts", type=int, nargs="+", default=[5, 21, 252, 512])
    p.add_argument("--models", type=str, nargs="+",
                   default=["amazon/chronos-bolt-tiny", "amazon/chronos-bolt-mini",
                            "amazon/chronos-t5-tiny",   "amazon/chronos-t5-mini"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--outdir", type=Path, default=PRED_DIR)
    return p.parse_args()


def main():
    ns = parse_args()
    device = pick_device(ns.device)

    all_summaries: list[pd.DataFrame] = []
    for L in ns.contexts:
        msg(f"Context length: {L}")
        for model_name in ns.models:
            msg(f"Running {model_name} (Context length={L}) ...")
            summary = run_one_model(
                data_path=ns.data,
                model_name=model_name,
                context_len=L,
                oos_start=ns.oos_start,
                device=device,
                outdir=ns.outdir,
            )
            all_summaries.append(summary)

    metrics = pd.concat(all_summaries, ignore_index=True)
    ns.outdir.mkdir(parents=True, exist_ok=True)
    metrics_out = ns.outdir / "chronos_metrics.csv"
    metrics.to_csv(metrics_out, index=False)
    msg(f"Saved combined metrics -> {metrics_out}")


if __name__ == "__main__":
    main()