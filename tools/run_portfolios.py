"""
Build and evaluate long-short portfolios from merged prediction files.

- Reads merged_window_{N}.csv from merged_results/
- Forms EW/VW long-short portfolios using top/bottom quantiles by model signal
- Supports transaction costs via turnover
- Outputs daily LS returns, cumulative log returns, and performance tables (Sharpe, etc.)
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# helpers
def find_id_col(df: pd.DataFrame) -> str:
    for c in ["PERMNO", "permno", "Id", "id"]:
        if c in df.columns:
            return c
    raise ValueError("Could not find an ID column (PERMNO/permno).")

def sharpe_clean(x: pd.Series) -> float:
    x = pd.Series(x).astype(float).dropna()
    sd = x.std(ddof=1)
    return float((x.mean() / sd) * np.sqrt(252)) if sd and np.isfinite(sd) else np.nan

def add_cumlog_columns(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    For each return column (everything except `date_col`), add a
    cumulative log return column named <col>_cumlog.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    for c in out.columns:
        if c == date_col:
            continue
        # make sure it's numeric, NaNs -> 0 so cumlog doesn't break on gaps
        s = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        out[f"{c}_cumlog"] = np.log1p(s).cumsum()
    return out

# portfolio engines
def long_short_portfolio(
    df: pd.DataFrame,
    date_col: str,
    actual_col: str,
    pred_col: str,
    weight_col: str | None = None,     # None = EW; else use value weights (e.g. 'market_cap')
    id_col: str = "PERMNO",
    frac: float = 0.10,                 # 10% long, 10% short
    transaction_cost: float = 0.00,    # per unit of turnover
    charge_opening: bool = True,
) -> pd.DataFrame:
    """
    Daily L-S portfolio with optional value-weights and turnover costs.

    Turnover is computed as 0.5 * L1 change between yesterday's drifted weights
    and today's target weights.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    results = []
    prev_w: pd.Series | None = None

    for d, g in df.sort_values(date_col).groupby(date_col, sort=False):
        # keep tradable set
        g = g.dropna(subset=[actual_col, pred_col]).copy()
        if weight_col is not None:
            g = g.dropna(subset=[weight_col])
            g = g[g[weight_col] > 0]

        n = len(g)
        have_port = n >= max(10, 2 * int(1 / frac))
        # target weights today
        if have_port:
            g = g.sort_values(pred_col, ascending=False)
            top_n = max(1, int(n * frac))
            bot_n = max(1, int(n * frac))
            long_g, short_g = g.head(top_n), g.tail(bot_n)

            if weight_col:
                lw = long_g[weight_col] / long_g[weight_col].sum() if long_g[weight_col].sum() > 0 else pd.Series(1 / top_n, index=long_g.index)
                sw = short_g[weight_col] / short_g[weight_col].sum() if short_g[weight_col].sum() > 0 else pd.Series(1 / bot_n, index=short_g.index)
            else:
                lw = pd.Series(1 / top_n, index=long_g.index)
                sw = pd.Series(1 / bot_n, index=short_g.index)

            w_tgt = pd.concat([
                pd.Series(lw.values,   index=long_g[id_col]),
                pd.Series(-sw.values,  index=short_g[id_col]),
            ])
        else:
            w_tgt = None

        # realized return using yesterday's weights
        if prev_w is not None:
            r_today = (df.loc[df[date_col] == d, [id_col, actual_col]]
                         .dropna(subset=[actual_col])
                         .set_index(id_col)[actual_col])
            w_used = prev_w.reindex(r_today.index).fillna(0.0)
            ret_gross = float((w_used * r_today).sum())
        else:
            ret_gross = 0.0

        # turnover & fee
        if prev_w is None:
            turnover = 0.5 * w_tgt.abs().sum() if (w_tgt is not None and charge_opening) else 0.0
        else:
            if w_tgt is None:
                turnover = 0.0
            else:
                # drift yesterday's weights by today's returns before rebalancing
                all_ids = prev_w.index.union(w_tgt.index)
                r_today_full = (df.loc[df[date_col] == d, [id_col, actual_col]]
                                  .dropna(subset=[actual_col])
                                  .set_index(id_col)[actual_col]
                                  .reindex(all_ids, fill_value=0.0))
                w_old = prev_w.reindex(all_ids, fill_value=0.0)
                port_ret = float((w_old * r_today_full).sum())
                denom = (1.0 + port_ret) if (1.0 + port_ret) != 0 else 1.0
                w_drift = (w_old * (1.0 + r_today_full)) / denom
                w_new   = w_tgt.reindex(all_ids, fill_value=0.0)
                turnover = 0.5 * (w_new - w_drift).abs().sum()

        fee = float(transaction_cost) * float(turnover)
        ret_net = ret_gross - fee
        results.append((d, ret_net))

        # carry next day
        if w_tgt is not None:
            prev_w = w_tgt

    return pd.DataFrame(results, columns=[date_col, f"ls_{pred_col}_net"])

def merge_ls_model_portfolios(
    df: pd.DataFrame,
    date_col: str,
    actual_col: str,
    model_cols: list[str],
    weight_col: str | None,
    id_col: str,
    frac: float,
    transaction_cost: float,
) -> pd.DataFrame:
    base = None
    for mc in model_cols:
        ls = long_short_portfolio(
            df, 
            date_col=date_col, 
            actual_col=actual_col, 
            pred_col=mc,
            weight_col=weight_col, 
            id_col=id_col, 
            frac=frac,
            transaction_cost=transaction_cost,
        )
        base = ls if base is None else base.merge(ls, on=date_col, how="left")
    return base

def compute_perf_table(returns_df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    rows = []
    for c in returns_df.columns:
        if c == date_col: 
            continue
        daily = pd.to_numeric(returns_df[c], errors="coerce")
        rows.append({
            "Portfolio": c,
            "Average Return (ann.)": daily.mean() * 252,
            "Volatility (ann. SD)":  daily.std(ddof=1) * np.sqrt(252),
            "Sharpe Ratio":          sharpe_clean(daily),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("Sharpe Ratio", ascending=False)


# ----------------------------
# main
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Build LS portfolios from merged prediction files.")
    p.add_argument("--merged-dir", type=Path, default=Path("merged_results"),
                   help="Folder containing merged_window_{N}.csv files.")
    p.add_argument("--windows", type=int, nargs="*", default=[5, 21, 252, 512],
                   help="Which windows to process (must exist in merged-dir).")
    p.add_argument("--sp500", type=Path, default=Path("data/sp500_excess_return_with_mcap.csv"),
                   help="CSV of S&P500 panel with columns [date, permno/PERMNO, excess_ret, mcap]. (optional)")
    p.add_argument("--date-col", type=str, default="DlyCalDt",
                   help="Date column name in merged files (default: DlyCalDt).")
    p.add_argument("--actual-col", type=str, default="y_true",
                   help="Actual (realized) excess return column.")
    p.add_argument("--mcap-col", type=str, default="market_cap",
                   help="Market cap column name for VW weights.")
    p.add_argument("--frac", type=float, default=0.10, help="Top/Bottom fraction for LS (default 0.10).")
    p.add_argument("--tcost", type=float, default=0.001, help="Transaction cost per unit turnover (default 0.001).")
    p.add_argument("--outdir", type=Path, default=Path("portfolio_outputs"),
                   help="Output folder for portfolio CSVs and metrics.")
    p.add_argument("--no-vw", action="store_true", help="Skip value-weighted LS portfolios.")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    ew_metrics_all, vw_metrics_all = [], []

    for W in args.windows:
        in_csv = args.merged_dir / f"merged_window_{W}.csv"
        if not in_csv.exists():
            print(f"[skip] {in_csv} not found.")
            continue

        print(f"\nWindow {W}")
        df = pd.read_csv(in_csv)
        # normalize columns
        if args.date_col not in df.columns:
            # fallback for older files
            if "date" in df.columns:
                df = df.rename(columns={"date": args.date_col})
            else:
                raise ValueError(f"{in_csv.name} has no date column '{args.date_col}' or 'date'.")
        id_col = find_id_col(df)

        df[args.date_col] = pd.to_datetime(df[args.date_col])
        drop_cols = {args.date_col, args.actual_col, args.mcap_col, id_col}
        model_cols = [c for c in df.columns if c.startswith("y_pred_") and c not in drop_cols]

        if not model_cols:
            print(f"[warn] No prediction columns in {in_csv.name}.")
            continue

        # EW portfolios
        ew = merge_ls_model_portfolios(
            df, args.date_col, args.actual_col, model_cols,
            weight_col=None, id_col=id_col,
            frac=args.frac, transaction_cost=args.tcost
        )
        ew_path = args.outdir / f"ls_ew_window_{W}.csv"
        ew.to_csv(ew_path, index=False)

        # also save cumulative log version
        ew_cum = add_cumlog_columns(ew, args.date_col)
        ew_cum_path = args.outdir / f"ls_ew_window_{W}_cumlog.csv"
        ew_cum.to_csv(ew_cum_path, index=False)

        ew_tbl = compute_perf_table(ew, args.date_col)
        ew_tbl.insert(0, "Lag/Window", W)
        ew_metrics_all.append(ew_tbl)

        # VW portfolios
        if not args.no_vw and args.mcap_col in df.columns:
            vw = merge_ls_model_portfolios(
                df, args.date_col, args.actual_col, model_cols,
                weight_col=args.mcap_col, id_col=id_col,
                frac=args.frac, transaction_cost=args.tcost
            )
            vw_path = args.outdir / f"ls_vw_window_{W}.csv"
            vw.to_csv(vw_path, index=False)

            # cum-log version
            vw_cum = add_cumlog_columns(vw, args.date_col)
            vw_cum_path = args.outdir / f"ls_vw_window_{W}_cumlog.csv"
            vw_cum.to_csv(vw_cum_path, index=False)

            vw_tbl = compute_perf_table(vw, args.date_col)
            vw_tbl.insert(0, "Lag/Window", W)
            vw_metrics_all.append(vw_tbl)

        print(f"[done] EW -> {ew_path} (+ {ew_cum_path})"
            + ("" if args.no_vw else f" | VW -> {vw_path} (+ {vw_cum_path})"))

    # save metrics
    if ew_metrics_all:
        ew_all = pd.concat(ew_metrics_all, ignore_index=True)
        ew_all.to_csv(args.outdir / "metrics_ew.csv", index=False)
        print(f"[saved] {args.outdir / 'metrics_ew.csv'}")

    if vw_metrics_all:
        vw_all = pd.concat(vw_metrics_all, ignore_index=True)
        vw_all.to_csv(args.outdir / "metrics_vw.csv", index=False)
        print(f"[saved] {args.outdir / 'metrics_vw.csv'}")

    print("\nAll done")

if __name__ == "__main__":
    main()