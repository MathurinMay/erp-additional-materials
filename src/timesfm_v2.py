# src/timesfm_v2.py
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
    COL_DATE, COL_PERMNO, COL_EXCESS, COL_MCAP, COL_PRICE, COL_SHROUT,
    DEFAULT_DATA, DEFAULT_OOS, PRED_DIR
)
from src.metrics import evaluate_predictions
from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint


def msg(s: str) -> None:
    print(f"[timesfm-v2] {s}")


# ---------- helpers shared with v1 ----------
def pick_device(opt: str = "auto") -> str:
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
    Map research lag window -> TimesFM's preferred context size.
      ≤ 21  -> 32
      ≤ 256 -> 256
      else  -> 512
    """
    if history_len <= 21:
        return 32
    elif history_len <= 256:
        return 256
    else:
        return 512


def ensure_mcap(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if COL_MCAP not in out.columns:
        out[COL_MCAP] = out[COL_PRICE].abs() * out[COL_SHROUT] * 1000.0
        msg("Computed market_cap from |Price|*ShrOut*1000.")
    return out


def clean_series(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def build_oos_contexts(
    df: pd.DataFrame, history_len: int, oos_start: pd.Timestamp
) -> Tuple[List[np.ndarray], List[float], List[dict]]:
    """
    Build (contexts, targets, records) for OOS only.
      - context: last `history_len` values of excess return
      - target : next-step realized excess return
      - records: metadata (PERMNO, date, market_cap)
    """
    df = df.sort_values([COL_PERMNO, COL_DATE]).copy()
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])

    contexts: List[np.ndarray] = []
    targets:  List[float] = []
    records:  List[dict] = []

    for permno, grp in df.groupby(COL_PERMNO, sort=False):
        values = grp[COL_EXCESS].to_numpy()
        dates  = grp[COL_DATE].to_numpy()
        mcaps  = grp[COL_MCAP].to_numpy()
        n = len(values)
        if n <= history_len:
            continue

        values = clean_series(values)

        for i in range(n - history_len):
            t_idx  = i + history_len
            t_date = dates[t_idx]
            if t_date >= oos_start:
                ctx = values[i:t_idx].astype(np.float32)  # [history_len]
                contexts.append(ctx)
                targets.append(float(values[t_idx]))
                records.append({
                    COL_PERMNO: permno,
                    COL_DATE:   t_date,
                    COL_MCAP:   float(mcaps[t_idx]),
                })
    return contexts, targets, records
# -------------------------------------------


def run_one(
    data_path: Path,
    model_name: str,
    lag_window: int,
    oos_start: str | pd.Timestamp,
    device_str: str,
    outdir: Path,
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Run TimesFM v2.0 for one lag window:
      - v2 context length is derived via pick_context_len(lag_window)
      - returns one-row metrics DataFrame
    """
    ctx_len = pick_context_len(lag_window)
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
    contexts_np, targets, records = build_oos_contexts(df, lag_window, oos_ts)
    if not contexts_np:
        raise ValueError("No OOS contexts created. Check dates/history length/data coverage.")

    # pad/trim to v2 context_len (left-pad zeros)
    proc_tensors: List[torch.Tensor] = []
    for ctx in contexts_np:
        L = len(ctx)
        if L < ctx_len:
            arr = np.pad(ctx, (ctx_len - L, 0)).astype(np.float32)
        elif L > ctx_len:
            arr = ctx[-ctx_len:].astype(np.float32)
        else:
            arr = ctx
        proc_tensors.append(torch.tensor(arr, dtype=torch.float32))

    # Load TimesFM v2
    msg(f"Loading TimesFM v2.0: {model_name} (device={device}, ctx_len={ctx_len}, batch={batch_size})")

    hparams = TimesFmHparams(
        backend=device,
        horizon_len=1,
        context_len=ctx_len,
        num_layers=50,                 
        use_positional_embedding=False 
    )

    tfm = TimesFm(
        hparams=hparams,
        checkpoint=TimesFmCheckpoint(huggingface_repo_id=model_name),
    )
    try:
        tfm._model.to(device)
    except Exception:
        pass

    # Inference
    preds: List[float] = []
    msg(f"Inference on {len(proc_tensors)} contexts...")
    try:
        for i in tqdm(range(0, len(proc_tensors), batch_size),
                      desc=f"TimesFM v2 ctx={ctx_len}", unit="batch"):
            chunk = proc_tensors[i:i+batch_size]
            if not chunk:
                break
            batch = torch.stack(chunk).to(device)  # [B, ctx_len]
            freq = [0] * batch.shape[0]
            with torch.no_grad():
                out, _ = tfm.forecast(batch, freq=freq)  # [B,1] or [B]
            if isinstance(out, torch.Tensor):
                out = out.squeeze(-1).detach().cpu().numpy().tolist()
            else:
                out = np.asarray(out).squeeze(-1).tolist()
            preds.extend([float(v) for v in out])
    except KeyboardInterrupt:
        msg("Interrupted by user. Saving partial results...")

    # Assemble results
    results = pd.DataFrame.from_records(records)
    results["y_true"] = np.asarray(targets, dtype=float)[:len(preds)]
    results["y_pred"] = np.asarray(preds,   dtype=float)

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
        "Context_Len_Used": int(ctx_len),
        **metrics,
    }])


# -------- v1-style CLI wrapper for v2 --------
def parse_args():
    p = argparse.ArgumentParser(description="Run TimesFM v2.0 across research lag windows (v1-style CLI).")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to preprocessed data CSV")
    p.add_argument("--oos-start", type=str, default=str(DEFAULT_OOS.date()), help="OOS start date (YYYY-MM-DD)")
    p.add_argument("--contexts", type=int, nargs="+", default=[5, 21, 252, 512],
                   help="Research lag windows to evaluate")
    p.add_argument("--model", type=str, default="google/timesfm-2.0-500m-pytorch",
                   help="TimesFM v2.0 model id (PyTorch port)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"],
                   help="Execution device")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for inference")
    p.add_argument("--outdir", type=Path, default=Path("results/timesfm_v2"),
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
            batch_size=ns.batch_size,
        )
        summaries.append(summary)

    # Save combined metrics
    metrics = pd.concat(summaries, ignore_index=True)
    ns.outdir.mkdir(parents=True, exist_ok=True)
    out_csv = ns.outdir / "timesfm_v2_metrics.csv"
    metrics.to_csv(out_csv, index=False)
    msg(f"Saved combined metrics -> {out_csv}")


if __name__ == "__main__":
    main()