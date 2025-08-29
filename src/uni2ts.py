"""
Run Salesforce Uni2TS (Moirai-MoE) as benchmarks.

- Loads the same panel (data/final_data.csv)
- Builds rolling contexts of length L per PERMNO
- Predicts next-step excess return with Moirai-MoE
- Evaluates OOS (DEFAULT_OOS from config) and saves predictions + a combined metrics CSV
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

# Uni2TS (Moirai-MoE)
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule


def msg(s: str) -> None:
    print(f"[uni2ts] {s}")


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

    - context: rolling history of excess returns of length L
    - target:  next-day (one-step-ahead) realized excess return
    - records: metadata (PERMNO, target date, market_cap)
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

        for i in range(len(values) - context_len):
            target_idx  = i + context_len
            target_date = dates[target_idx]
            if target_date >= oos_start:
                ctx = torch.tensor(values[i:target_idx], dtype=torch.float32)
                contexts.append(ctx)
                targets.append(float(values[target_idx]))
                records.append({
                    COL_PERMNO: permno,
                    COL_DATE:   target_date,
                    COL_MCAP:   float(mcaps[target_idx]),
                })

    return contexts, targets, records


def pick_device(opt: str) -> str:
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
    num_samples: int = 100,
) -> pd.DataFrame:
    """
    Run one Uni2TS (Moirai-MoE) model with a given context length.
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

    # Load Uni2TS (Moirai-MoE)
    msg(f"Loading Uni2TS model: {model_name} (device={device})")
    module = MoiraiMoEModule.from_pretrained(model_name).to(device)
    model = MoiraiMoEForecast(
        module=module,
        prediction_length=1,
        context_length=context_len,
        num_samples=num_samples,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    ).to(device)
    predictor = model.create_predictor(batch_size=1)

    # Inference
    preds: List[float] = []
    for ctx in tqdm(contexts, desc=f"{model_name} Context length={context_len}"):
        series = ctx.detach().cpu().numpy().tolist()
        ds = ListDataset([{"start": pd.Timestamp("2000-01-03"), "target": series}], freq="B")
        with torch.no_grad():
            fc = list(predictor.predict(ds))[0]
            preds.append(float(fc.mean[0]))

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
    out_csv = outdir / f"{safe_model}_predictions_ctx{context_len}.csv"
    results.to_csv(out_csv, index=False)
    msg(f"Saved predictions -> {out_csv}")

    # One-row summary
    summary = pd.DataFrame([{
        "Model": model_name,
        "Context_Length": int(context_len),
        **metrics,
    }])
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Run Uni2TS (Moirai-MoE) across multiple context lengths.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to preprocessed data CSV")
    p.add_argument("--oos-start", type=str, default=str(DEFAULT_OOS.date()), help="OOS start date (YYYY-MM-DD)")
    p.add_argument("--contexts", type=int, nargs="+", default=[5, 21, 252, 512],
                   help="Context lengths (history size) to evaluate")
    p.add_argument("--models", type=str, nargs="+",
                   default=["Salesforce/moirai-moe-1.0-R-small", "Salesforce/moirai-moe-1.0-R-base"],
                   help="Uni2TS model identifiers to run")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"],
                   help="Execution device")
    p.add_argument("--outdir", type=Path, default=PRED_DIR,
                   help="Where to save per-point predictions and the combined metrics CSV")
    p.add_argument("--num-samples", type=int, default=100, help="Samples for the probabilistic forecast")
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
                num_samples=ns.num_samples,
            )
            all_summaries.append(summary)

    metrics = pd.concat(all_summaries, ignore_index=True)
    ns.outdir.mkdir(parents=True, exist_ok=True)
    metrics_out = ns.outdir / "uni2ts_metrics.csv"
    metrics.to_csv(metrics_out, index=False)
    msg(f"Saved combined metrics -> {metrics_out}")


if __name__ == "__main__":
    main()