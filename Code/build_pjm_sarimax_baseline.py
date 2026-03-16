#!/usr/bin/env python3
"""
build_pjm_sarimax_baseline.py

Compact SARIMAX baseline for hourly PJM DOM prices.

Why compact?
-------------
A full seasonal hourly SARIMAX on the entire 2018-2025 panel can be very slow.
This script keeps the model tractable by:
1) using a recent rolling sample by default,
2) encoding daily/weekly/annual seasonality with Fourier terms,
3) fitting a low-order SARIMAX with exogenous regressors.

Default target: day-ahead LMP (da_lmp)
Default input : /mnt/data/pjm_dom_weather_merged_model_ready.csv
Default output: /mnt/data/pjm_sarimax_outputs/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm


DEFAULT_INPUT = "/mnt/data/pjm_dom_weather_merged_model_ready.csv"
DEFAULT_OUTPUT_DIR = "/mnt/data/pjm_sarimax_outputs"
LOCAL_TZ = "America/New_York"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact SARIMAX baseline for PJM DOM hourly prices.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Merged PJM+weather CSV")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for outputs")
    parser.add_argument("--target", default="da_lmp", choices=["da_lmp", "rt_lmp"], help="Target price variable")
    parser.add_argument("--max-rows", type=int, default=5000, help="Use only the most recent N non-missing hourly rows")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Training fraction")
    parser.add_argument("--order", default="1,0,0", help="ARIMA order as p,d,q")
    parser.add_argument("--maxiter", type=int, default=40, help="Maximum optimizer iterations")
    return parser.parse_args()


def _parse_order(text: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError("--order must look like p,d,q, e.g. 1,0,0")
    return tuple(parts)  # type: ignore[return-value]


def load_data(path: str, target: str, max_rows: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp_utc" not in df.columns:
        raise ValueError("Input file must contain timestamp_utc")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    df = df.set_index("timestamp_utc").asfreq("h")

    needed = [target, "load_mw", "temp_c", "dewpoint_c", "wind_speed_ms", "cdd", "hdd", "rh_pct"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[needed].copy()
    df = df.dropna(subset=[target])
    if max_rows and len(df) > max_rows:
        df = df.tail(max_rows).copy()

    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[needed] = df[needed].interpolate(limit_direction="both")
    df[needed] = df[needed].ffill().bfill()

    idx_local = df.index.tz_convert(LOCAL_TZ)
    hour = idx_local.hour.to_numpy()
    dow = idx_local.dayofweek.to_numpy()
    doy = idx_local.dayofyear.to_numpy()

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["is_weekend"] = (dow >= 5).astype(int)
    df["temp_x_load"] = df["temp_c"] * df["load_mw"]
    return df


def get_exog_columns() -> List[str]:
    return [
        "load_mw",
        "temp_c",
        "dewpoint_c",
        "wind_speed_ms",
        "rh_pct",
        "cdd",
        "hdd",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
        "is_weekend",
        "temp_x_load",
    ]


def compute_metrics(actual: pd.Series, pred: pd.Series) -> dict:
    err = actual - pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(actual), 1e-6)) * 100.0)
    ss_res = float(np.sum(np.square(err)))
    ss_tot = float(np.sum(np.square(actual - actual.mean())))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"mae": mae, "rmse": rmse, "mape_pct": mape, "r2": r2}


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    order = _parse_order(args.order)
    df = load_data(args.input, args.target, args.max_rows)
    exog_cols = get_exog_columns()

    n_train = max(48, int(len(df) * args.train_frac))
    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train:].copy()

    endog_train = train[args.target]
    exog_train = train[exog_cols]
    endog_test = test[args.target]
    exog_test = test[exog_cols]

    model = SARIMAX(
        endog_train,
        exog=exog_train,
        order=order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=args.maxiter)

    fitted_train = pd.Series(res.fittedvalues, index=train.index, name="predicted")
    if len(test) > 0:
        forecast = res.get_forecast(steps=len(test), exog=exog_test)
        pred_test = pd.Series(forecast.predicted_mean, index=test.index, name="predicted")
    else:
        pred_test = pd.Series(dtype=float, name="predicted")

    pred_all = pd.concat([
        pd.DataFrame({"split": "train", "actual": endog_train, "predicted": fitted_train}),
        pd.DataFrame({"split": "test", "actual": endog_test, "predicted": pred_test}),
    ]).sort_index()
    pred_all["residual"] = pred_all["actual"] - pred_all["predicted"]
    pred_all = pred_all.reset_index().rename(columns={"timestamp_utc": "timestamp_utc"})
    pred_all.to_csv(outdir / f"{args.target}_predictions.csv", index=False)

    train_metrics = compute_metrics(endog_train.loc[fitted_train.index], fitted_train)
    test_metrics = compute_metrics(endog_test.loc[pred_test.index], pred_test) if len(test) > 0 else {}

    resid_train = endog_train.loc[fitted_train.index] - fitted_train
    lb = acorr_ljungbox(resid_train.dropna(), lags=[24], return_df=True)
    lb_p = float(lb["lb_pvalue"].iloc[0]) if not lb.empty else np.nan

    try:
        adf_p = float(adfuller(resid_train.dropna(), autolag="AIC")[1])
    except Exception:
        adf_p = np.nan

    try:
        bp_exog = sm.add_constant(train[exog_cols]).loc[resid_train.dropna().index]
        bp_stat, bp_p, _, _ = het_breuschpagan(resid_train.dropna(), bp_exog)
        bp_stat, bp_p = float(bp_stat), float(bp_p)
    except Exception:
        bp_stat, bp_p = np.nan, np.nan

    params = pd.DataFrame({"parameter": res.params.index, "value": res.params.values})
    params.to_csv(outdir / f"{args.target}_params.csv", index=False)
    (outdir / f"{args.target}_summary.txt").write_text(res.summary().as_text(), encoding="utf-8")

    metrics = {
        "target": args.target,
        "rows_used": int(len(df)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "order": list(order),
        "exog_columns": exog_cols,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "aic": float(res.aic),
        "bic": float(res.bic),
        "ljung_box_p_lag24": lb_p,
        "adf_p_residual": adf_p,
        "breusch_pagan_stat": bp_stat,
        "breusch_pagan_p": bp_p,
        "start_utc": str(df.index.min()),
        "end_utc": str(df.index.max()),
    }
    (outdir / f"{args.target}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Plot actual vs predicted on test set
    if len(test) > 0:
        plt.figure(figsize=(12, 5))
        plt.plot(test.index, endog_test, label="Actual")
        plt.plot(pred_test.index, pred_test, label="Predicted")
        plt.title(f"{args.target}: actual vs predicted (test set)")
        plt.xlabel("UTC time")
        plt.ylabel(args.target)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{args.target}_test_actual_vs_predicted.png", dpi=180)
        plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(resid_train.dropna(), bins=50)
    plt.title(f"{args.target}: training residual distribution")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outdir / f"{args.target}_residual_hist.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4))
    plot_acf(resid_train.dropna(), lags=48, ax=plt.gca())
    plt.title(f"{args.target}: ACF of training residuals")
    plt.tight_layout()
    plt.savefig(outdir / f"{args.target}_residual_acf.png", dpi=180)
    plt.close()

    print(f"Saved outputs to: {outdir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
