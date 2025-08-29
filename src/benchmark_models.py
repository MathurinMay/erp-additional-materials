"""
Benchmark models for the ERP project.

- Loads preprocessed panel from data/final_data.csv
- Builds lagged features for specified lags (e.g., 5,21,252,512)
- Trains a set of benchmark models (OLS, Lasso, Ridge, ElasticNet, GBRT, XGB, RF, PCR, PLS, GLM-Spline, and NN1-5)
- Evaluates OOS (from 2016-01-01) with MSE, R2_OOS, directional accuracies, top/bottom-5 by mcap
- Saves per-model predictions and a combined metrics table

Usage examples:
  python -m src.benchmark_models
  python -m src.benchmark_models --lags 5 21 --models OLS Lasso
  python -m src.benchmark_models --outdir results/predictions --oos-start 2016-01-01
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, Tuple, List, Any, Sequence

import argparse
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoCV
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from collections import Counter

from src.config import (
    COL_DATE, 
    COL_PERMNO, 
    COL_PRICE, 
    COL_SHROUT,
    COL_EXCESS,
    COL_MCAP,
    DEFAULT_DATA,
    DEFAULT_OOS,
    DEFAULT_OUTDIR
)
from src.metrics import evaluate_predictions

Y_TRUE = "y_true"
Y_PRED = "y_pred"

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

_TS_cv = TimeSeriesSplit(n_splits=3)

def msg(s: str): print(f"[benchmark models] {s}")

# Feature engineering 
def make_lagged(df: pd.DataFrame, lags: int) -> pd.DataFrame:
    """
    Create lagged features day_1..day_{lags} of excess returns per PERMNO.
    Drops rows with any missing lag.
    """
    df = df.sort_values([COL_PERMNO, COL_DATE]).copy()
    for i in range(1, lags + 1):
        df[f"day_{i}"] = df.groupby(COL_PERMNO)[COL_EXCESS].shift(i)
    return df.dropna().reset_index(drop=True)

def ensure_mcap(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure market_cap exists; compute from |price|*shrout*1000 if missing."""
    out = df.copy()
    if COL_MCAP not in out.columns:
        out[COL_MCAP] = out[COL_PRICE].abs() * out[COL_SHROUT] * 1000.0
        msg("Computed market_cap from DlyPrc*ShrOut*1000.")
    return out

# Generic train/eval
def run_forecast(
    df: pd.DataFrame,
    model_name: str,
    model_func: Callable[..., Tuple[Any, np.ndarray, Dict[str, Any]]],
    lag_count: int,
    oos_start: pd.Timestamp,
    extra_kwargs: Dict[str, Any] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build lags, split IS/OOS by oos_start, fit model, compute metrics,
    return (metrics_df, test_df_with_predictions).
    """
    extra_kwargs = extra_kwargs or {}
    lag_cols = [f"day_{i}" for i in range(1, lag_count + 1)]
    use_cols = [COL_DATE, COL_PERMNO, COL_EXCESS, COL_MCAP] + lag_cols
    sub = df[use_cols].dropna().copy()

    train = sub[sub[COL_DATE] < oos_start].copy()
    test  = sub[sub[COL_DATE] >= oos_start].copy().reset_index(drop=True)

    X_train, y_train = train[lag_cols], train[COL_EXCESS]
    X_test,  y_test  = test[lag_cols],  test[COL_EXCESS]

    model, preds, meta = model_func(X_train, y_train, X_test, **extra_kwargs)
    test = test.copy()
    test[Y_PRED]    = preds
    test[Y_TRUE]  = y_test.values
    test["correct"] = (np.sign(test[Y_PRED]) == np.sign(test[Y_TRUE]))

    metric_dict = evaluate_predictions(
        y_true=test[Y_TRUE].to_numpy(),
        y_pred=test[Y_PRED].to_numpy(),
        df=test,                 
        col_date=COL_DATE,
        col_mcap=COL_MCAP
    )

    metric_dict.update(meta or {})
    metric_dict["Model"] = model_name
    metric_dict["Lags"]  = lag_count

    return pd.DataFrame([metric_dict]), test

# Model wrappers
def m_ols(X_train, y_train, X_test, **_):
    mdl = LinearRegression().fit(X_train, y_train)
    return mdl, mdl.predict(X_test), {}

def m_lasso(X_train, y_train, X_test, alphas: Sequence[float] | None = None, **_):
    alphas = list(alphas) if alphas is not None else [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    g = GridSearchCV(Lasso(max_iter=10000), {"alpha": alphas},
                     scoring="neg_mean_squared_error", cv=_TS_cv)
    g.fit(X_train, y_train); best = g.best_estimator_
    return best, best.predict(X_test), {"best_alpha": g.best_params_["alpha"]}

def m_ridge(X_train, y_train, X_test, alphas: Sequence[float] | None = None, **_):
    alphas = list(alphas) if alphas is not None else [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    g = GridSearchCV(Ridge(), {"alpha": alphas},
                     scoring="neg_mean_squared_error", cv=_TS_cv)
    g.fit(X_train, y_train); best = g.best_estimator_
    return best, best.predict(X_test), {"best_alpha": g.best_params_["alpha"]}

def m_enet(X_train, y_train, X_test,
           alphas: Sequence[float] | None = None,
           l1_ratios: Sequence[float] | None = None, **_):
    alphas    = list(alphas) if alphas is not None else [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    l1_ratios = list(l1_ratios) if l1_ratios is not None else [0.1, 0.5, 0.9]
    g = GridSearchCV(ElasticNet(max_iter=10000),
                     {"alpha": alphas, "l1_ratio": l1_ratios},
                     scoring="neg_mean_squared_error", cv=_TS_cv)
    g.fit(X_train, y_train); best = g.best_estimator_
    return best, best.predict(X_test), {"best_params": g.best_params_}

def m_glm_spline(X_train, y_train, X_test, **_):
    pipe = Pipeline([
        ("spline", SplineTransformer(include_bias=False)),
        ("lasso",  LassoCV(cv=3, max_iter=50000))
    ])
    grid = GridSearchCV(pipe,
                        {"spline__degree":[2,3], "spline__n_knots":[3,5,7,10]},
                        cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    yhat = best.predict(X_test)
    return best, yhat, {"best_degree":grid.best_params_["spline__degree"],
                        "best_n_knots":grid.best_params_["spline__n_knots"],
                        "type":"Spline+Lasso"}

def m_gbrt(X_train, y_train, X_test, **_):
    grid = GridSearchCV(GradientBoostingRegressor(),
                        {"n_estimators":[50, 100, 200],
                         "learning_rate":[0.01, 0.1, 0.3],
                        "max_depth":[2,3,5]},
                        scoring="neg_mean_squared_error", cv=_TS_cv)
    grid.fit(X_train, y_train); best = grid.best_estimator_
    return best, best.predict(X_test), {"best_params": grid.best_params_}

def m_xgb(X_train, y_train, X_test, **_):
    if not _HAS_XGB:
        return None, np.full(X_test.shape[0], np.nan), {"note":"xgboost not installed"}
    grid = GridSearchCV(XGBRegressor(),
                        {"n_estimators":[50, 100, 200],
                         "learning_rate":[0.01, 0.1, 0.3],
                         "max_depth":[2,3,5]},
                        scoring="neg_mean_squared_error", cv=_TS_cv)
    grid.fit(X_train, y_train); best = grid.best_estimator_
    return best, best.predict(X_test), {"best_params": grid.best_params_}

def m_rf(X_train, y_train, X_test, **_):
    grid = GridSearchCV(RandomForestRegressor(),
                        {"n_estimators":[100, 200],
                         "max_depth":[3, 5, 10,None],
                         "max_features":["sqrt","log2"]},
                        scoring="neg_mean_squared_error", cv=_TS_cv)
    grid.fit(X_train, y_train); best = grid.best_estimator_
    return best, best.predict(X_test), {"best_params": grid.best_params_}

PCR_GRID_BY_LAG = {
    5:   [2, 3, 4, 5],
    21:  [5, 10, 15, 20],
    252: [5, 10, 20, 40, 60, 100, 200],
    512: [5, 10, 20, 40, 60, 100, 200],
}
PLS_GRID_BY_LAG = {
    5:   [2, 3, 4, 5],
    21:  [5, 10, 15, 20],
    252: [5, 10, 20, 40, 60, 100, 200],
    512: [5, 10, 20, 40, 60, 100, 200],
}

def m_pcr(X_train, y_train, X_test, n_components_list: Sequence[int] | None = None, **_):
    n_components_list = list(n_components_list) if n_components_list is not None else [3,5,10,20]
    pipe = Pipeline([("pca", PCA()), ("lr", LinearRegression())])
    grid = GridSearchCV(pipe,
                        {"pca__n_components": n_components_list},
                        scoring="neg_mean_squared_error", cv=_TS_cv)
    grid.fit(X_train, y_train); best = grid.best_estimator_
    return best, best.predict(X_test), {"n_components": grid.best_params_["pca__n_components"]}

def m_pls(X_train, y_train, X_test, n_components_list: Sequence[int] | None = None, **_):
    n_components_list = list(n_components_list) if n_components_list is not None else [2,3,5,10,20]
    grid = GridSearchCV(PLSRegression(), {"n_components": n_components_list},
                        scoring="neg_mean_squared_error", cv=_TS_cv)
    grid.fit(X_train, y_train); best = grid.best_estimator_
    preds = best.predict(X_test).ravel()
    return best, preds, {"n_components": grid.best_params_["n_components"]}

def m_neural_networks(
    X_train, y_train, X_test,
    architecture: str = "NN1",
    seeds: list[int] = [0, 1, 2],
    **kwargs
):
    """
      - Architecture NN1..NN5 (32 -> 16 -> 8 -> 4 -> 2)
      - StandardScaler on inputs
      - GridSearchCV over learning_rate_init (max_iter fixed 500)
      - Ensemble over seeds: average predictions
    """
    arch_map = {
        "NN1": (32,),
        "NN2": (32, 16),
        "NN3": (32, 16, 8),
        "NN4": (32, 16, 8, 4),
        "NN5": (32, 16, 8, 4, 2),
    }
    if architecture not in arch_map:
        raise ValueError(f"Unknown NN architecture: {architecture}")
    hidden_layer = arch_map[architecture]

    # scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # small grid (เหมือนของเดิม)
    param_grid = {
        "learning_rate_init": [0.001, 0.01],
        "max_iter": [1000],
    }

    predictions = []
    best_params_list = []
    best_model = None

    for seed in seeds:
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer,
            activation="relu",
            solver="sgd",
            learning_rate="adaptive",
            early_stopping=True,
            n_iter_no_change=5,
            random_state=seed,
        )
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_  # keep last best for return
        y_pred = best_model.predict(X_test_scaled)
        predictions.append(y_pred)
        best_params_list.append(grid.best_params_)

    y_pred_avg = np.mean(predictions, axis=0)

    # aggregate “most common” best params across seeds
    param_counts = Counter([tuple(sorted(d.items())) for d in best_params_list])
    most_common_params = dict(param_counts.most_common(1)[0][0])

    meta = {
        "architecture": architecture,
        "hidden_layer_sizes": hidden_layer,
        "seeds_used": seeds,
        "best_hyperparams_mode": most_common_params,
    }
    return best_model, y_pred_avg, meta


MODEL_REGISTRY: Dict[str, Callable[..., Tuple[Any, np.ndarray, Dict[str, Any]]]] = {
    "OLS": m_ols,
    "Lasso": m_lasso,
    "Ridge": m_ridge,
    "ElasticNet": m_enet,
    "GLM-Spline": m_glm_spline,
    "GBRT": m_gbrt,
    "XGB": m_xgb,
    "RF": m_rf,
    "PCR": m_pcr,
    "PLS": m_pls,
    "NN1": m_neural_networks,
    "NN2": m_neural_networks,
    "NN3": m_neural_networks,
    "NN4": m_neural_networks,
    "NN5": m_neural_networks,
}

def parse_args():
    p = argparse.ArgumentParser(description="Run benchmark models over lag windows.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to preprocessed data CSV")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Where to save predictions/metrics")
    p.add_argument("--oos-start", type=str, default=str(DEFAULT_OOS.date()), help="OOS start date (YYYY-MM-DD)")
    p.add_argument("--lags", type=int, nargs="+", default=[5, 21, 252, 512], help="Lag windows") 
    p.add_argument("--models", type=str, nargs="+", default=list(MODEL_REGISTRY.keys()),
                   choices=list(MODEL_REGISTRY.keys()),
                   help="Subset of models to run")
    #Optional grids
    p.add_argument("--lasso-alphas", type=float, nargs="*", default=[1e-4,1e-3,1e-2,1e-1,1,10])
    p.add_argument("--ridge-alphas", type=float, nargs="*", default=[1e-4,1e-3,1e-2,1e-1,1,10])
    p.add_argument("--enet-alphas",  type=float, nargs="*", default=[1e-4,1e-3,1e-2,1e-1,1,10])
    p.add_argument("--enet-l1ratios",type=float, nargs="*", default=[0.1,0.5,0.9])
    p.add_argument("--pcr-components", type=int, nargs="*", default=None,
               help="Override list of n_components for PCR; if omitted, use grid by lag.")
    p.add_argument("--pls-components", type=int, nargs="*", default=None,
               help="Override list of n_components for PLS; if omitted, use grid by lag.")
    p.add_argument("--nn-seeds", type=int, nargs="*", default=[0, 1, 2])
    return p.parse_args()

def main():
    ns = parse_args()

    if not ns.data.exists():
        raise FileNotFoundError(f"Cannot find {ns.data}. Run preprocessing first.")
    df = pd.read_csv(ns.data)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    df = ensure_mcap(df)

    ns.outdir.mkdir(parents=True, exist_ok=True)
    oos = pd.Timestamp(ns.oos_start)

    # Build kwargs map for models that accept custom grids
    model_kwargs: Dict[str, Dict[str, Any]] = {
        "Lasso":      {"alphas": ns.lasso_alphas},
        "Ridge":      {"alphas": ns.ridge_alphas},
        "ElasticNet": {"alphas": ns.enet_alphas, "l1_ratios": ns.enet_l1ratios},
        "PCR":        {"n_components_list": ns.pcr_components},
        "PLS":        {"n_components_list": ns.pls_components},
        "NN1": {"architecture": "NN1", "seeds": ns.nn_seeds},
        "NN2": {"architecture": "NN2", "seeds": ns.nn_seeds},
        "NN3": {"architecture": "NN3", "seeds": ns.nn_seeds},
        "NN4": {"architecture": "NN4", "seeds": ns.nn_seeds},
        "NN5": {"architecture": "NN5", "seeds": ns.nn_seeds},
    }

    all_results: List[pd.DataFrame] = []

    for h in ns.lags:
        msg(f"Lag: {h} day lags")
        lagged = make_lagged(df, lags=h)

        model_kwargs: Dict[str, Dict[str, Any]] = {
            "Lasso":      {"alphas": ns.lasso_alphas},
            "Ridge":      {"alphas": ns.ridge_alphas},
            "ElasticNet": {"alphas": ns.enet_alphas, "l1_ratios": ns.enet_l1ratios},
            "PCR":        {"n_components_list": PCR_GRID_BY_LAG.get(h, [5])},
            "PLS":        {"n_components_list": PLS_GRID_BY_LAG.get(h, [3])},
            "NN1": {"architecture": "NN1", "seeds": ns.nn_seeds},
            "NN2": {"architecture": "NN2", "seeds": ns.nn_seeds},
            "NN3": {"architecture": "NN3", "seeds": ns.nn_seeds},
            "NN4": {"architecture": "NN4", "seeds": ns.nn_seeds},
            "NN5": {"architecture": "NN5", "seeds": ns.nn_seeds},
        }

        for name in ns.models:
            fn = MODEL_REGISTRY[name]
            msg(f"Training {name} ({h} lags)...")
            res_df, test_df = run_forecast(
                lagged, model_name=name, model_func=fn, lag_count=h,
                oos_start=oos, extra_kwargs=model_kwargs.get(name, {})
            )
            all_results.append(res_df)

            out_pred = ns.outdir / f"{name.lower()}_predictions_{h}d.csv"
            test_df.to_csv(out_pred, index=False)

    # Save combined metrics
    metrics = pd.concat(all_results, ignore_index=True)
    metrics_out = ns.outdir / "benchmark_metrics.csv"
    metrics.to_csv(metrics_out, index=False)
    msg(f"Saved metrics -> {metrics_out}")

if __name__ == "__main__":
    main()