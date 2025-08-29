import argparse
import pandas as pd
import numpy as np
import wrds
from src.config import OOS_START, OOS_END  

def load_sp500(start=OOS_START, end=OOS_END) -> pd.DataFrame:
    conn = wrds.Connection()
    try:
        # 1) Constituents x Daily file
        spx = conn.raw_sql(f"""
            SELECT a.permno, a.start, a.ending,
                   b.date, b.ret, b.prc, b.shrout
            FROM crsp.msp500list AS a
            JOIN crsp.dsf AS b
              ON a.permno = b.permno
            WHERE b.date BETWEEN a.start AND a.ending
              AND b.date BETWEEN '{start}' AND '{end}'
            ORDER BY b.date, a.permno
        """, date_cols=['start','ending','date'])

        # 2) Delisting returns (daily)
        dl = conn.raw_sql(f"""
            SELECT permno, dlstdt AS date, dlret
            FROM crsp.dsedelist
            WHERE dlstdt BETWEEN '{start}' AND '{end}'
        """, date_cols=['date'])

        # 3) Risk-free rate (FF daily) 
        ff = conn.raw_sql(f"""
            SELECT date, rf
            FROM ff.factors_daily
            WHERE date BETWEEN '{start}' AND '{end}'
        """, date_cols=['date'])
        ff['rf'] = ff['rf'] / 100.0

    finally:
        conn.close()

    # Merge DLRET
    spx = spx.merge(dl, on=['permno','date'], how='left')
    spx['dlret'] = spx['dlret'].fillna(0.0)

    # Market cap: PRC*SHROUT*1000 
    spx['mcap'] = spx['prc'].abs() * spx['shrout'] * 1000.0

    # Merge RF
    spx = spx.merge(ff, on='date', how='left')
    spx['rf'] = spx['rf'].fillna(method='ffill') 

    # Total simple return including delist
    spx['ret_total'] = (1.0 + spx['ret'].fillna(0.0)) * (1.0 + spx['dlret']) - 1.0

    # Excess return per stock
    spx['excess_ret'] = spx['ret_total'] - spx['rf']

    return spx

def make_benchmark(spx: pd.DataFrame) -> pd.DataFrame:
    # Value-weighted by market cap 
    daily = spx.groupby('date', as_index=False).apply(
        lambda g: pd.Series({
            'excess_ret': np.average(g['excess_ret'].values, weights=g['mcap'].values) 
                         if g['mcap'].sum() > 0 else np.nan,
            'mcap': g['mcap'].sum()
        })
    ).reset_index(drop=True)

    return daily[['date','excess_ret','mcap']].sort_values('date')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=str, default=str(OOS_START.date()))
    ap.add_argument('--end', type=str, default=str(OOS_END.date()))
    ap.add_argument('--out', type=str, default='data/sp500_excess_return_with_mcap.csv')
    args = ap.parse_args()

    spx = load_sp500(args.start, args.end)
    bench = make_benchmark(spx)
    out = args.out
    pd.Series(dtype=object)  
    bench.to_csv(out, index=False)
    print(f"Saved benchmark -> {out}  ({len(bench):,} rows)")

if __name__ == '__main__':
    main()