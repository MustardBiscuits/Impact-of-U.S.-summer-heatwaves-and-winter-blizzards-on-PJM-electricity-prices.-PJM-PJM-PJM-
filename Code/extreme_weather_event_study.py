#!/usr/bin/env python3
"""
extreme_weather_event_study.py

Event-study utilities for the PJM DOM weather-price project.

Included analyses
-----------------
1) Winter Storm Elliott window in PJM DOM
   Default event window (local time): 2022-12-23 00:00 to 2022-12-24 23:00
   Default estimation window: +/- 14 days around event, excluding event hours.

2) Summer CP-like signal windows
   Identify top-N summer weekday load days and study 16:00-19:00 local hours.

The script uses a transparent regression baseline rather than a heavy full-sample
state-space model so that it can be re-estimated around each event quickly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


DEFAULT_INPUT = "/mnt/data/pjm_dom_weather_merged_model_ready.csv"
DEFAULT_OUTPUT_DIR = "/mnt/data/pjm_event_study_outputs"
LOCAL_TZ = "America/New_York"
ELLIOTT_START = "2022-12-23 00:00"
ELLIOTT_END = "2022-12-24 23:00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Event study for PJM DOM extreme-weather price events")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Merged PJM+weather CSV")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for outputs")
    parser.add_argument("--target", default="da_lmp", choices=["da_lmp", "rt_lmp"], help="Price series to study")
    parser.add_argument("--elliott-start", default=ELLIOTT_START, help="Local start of Elliott event")
    parser.add_argument("--elliott-end", default=ELLIOTT_END, help="Local end of Elliott event")
    parser.add_argument("--window-days", type=int, default=14, help="Days before/after event for local regression window")
    parser.add_argument("--top-n-summer-days", type=int, default=15, help="Top summer weekday peak-load days to study")
    return parser.parse_args()


def load_data(path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp_utc" not in df.columns:
        raise ValueError("Input file must contain timestamp_utc")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    df["timestamp_local"] = df["timestamp_utc"].dt.tz_convert(LOCAL_TZ)
    df["date_local"] = df["timestamp_local"].dt.date.astype(str)
    df["year"] = df["timestamp_local"].dt.year
    df["month"] = df["timestamp_local"].dt.month
    df["day"] = df["timestamp_local"].dt.day
    df["hour"] = df["timestamp_local"].dt.hour
    df["weekday"] = df["timestamp_local"].dt.dayofweek
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    needed = [target, "load_mw", "temp_c", "dewpoint_c", "wind_speed_ms", "cdd", "hdd"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def regression_baseline(train: pd.DataFrame, score: pd.DataFrame, target: str) -> pd.DataFrame:
    train = train.dropna(subset=[target, "load_mw", "temp_c", "dewpoint_c", "wind_speed_ms", "cdd", "hdd"]).copy()
    score = score.copy()
    formula = (
        f"{target} ~ load_mw + temp_c + dewpoint_c + wind_speed_ms + cdd + hdd "
        "+ C(hour) + C(weekday) + C(month)"
    )
    model = smf.ols(formula=formula, data=train).fit(cov_type="HC3")
    score["expected_price"] = model.predict(score)
    score["abnormal_price"] = score[target] - score["expected_price"]
    return score, model


def elliott_event_study(df: pd.DataFrame, target: str, start_local: str, end_local: str, window_days: int, outdir: Path) -> dict:
    start = pd.Timestamp(start_local, tz=LOCAL_TZ)
    end = pd.Timestamp(end_local, tz=LOCAL_TZ)
    window_start = start - pd.Timedelta(days=window_days)
    window_end = end + pd.Timedelta(days=window_days)

    window_df = df[(df["timestamp_local"] >= window_start) & (df["timestamp_local"] <= window_end)].copy()
    window_df["is_event"] = ((window_df["timestamp_local"] >= start) & (window_df["timestamp_local"] <= end)).astype(int)

    train = window_df[window_df["is_event"] == 0].copy()
    scored, model = regression_baseline(train, window_df, target)

    scored.to_csv(outdir / f"elliott_{target}_hourly.csv", index=False)
    event_only = scored[scored["is_event"] == 1].copy()

    summary = {
        "target": target,
        "event_start_local": str(start),
        "event_end_local": str(end),
        "window_days_each_side": window_days,
        "event_hours": int(len(event_only)),
        "mean_actual_price": float(event_only[target].mean()),
        "mean_expected_price": float(event_only["expected_price"].mean()),
        "mean_abnormal_price": float(event_only["abnormal_price"].mean()),
        "cumulative_abnormal_price": float(event_only["abnormal_price"].sum()),
        "peak_actual_price": float(event_only[target].max()),
        "peak_expected_price": float(event_only["expected_price"].max()),
        "peak_abnormal_price": float(event_only["abnormal_price"].max()),
        "ols_r2_train": float(model.rsquared),
        "n_train": int(len(train)),
    }
    (outdir / f"elliott_{target}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(12, 5))
    plt.plot(scored["timestamp_local"], scored[target], label="Actual")
    plt.plot(scored["timestamp_local"], scored["expected_price"], label="Expected")
    plt.axvspan(start, end, alpha=0.2)
    plt.title(f"Elliott event study: {target}")
    plt.xlabel("Local time")
    plt.ylabel(target)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"elliott_{target}_actual_vs_expected.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.bar(event_only["timestamp_local"], event_only["abnormal_price"])
    plt.title(f"Elliott abnormal {target}")
    plt.xlabel("Local time")
    plt.ylabel("Actual - expected")
    plt.tight_layout()
    plt.savefig(outdir / f"elliott_{target}_abnormal.png", dpi=180)
    plt.close()
    return summary


def summer_cp_like_study(df: pd.DataFrame, target: str, top_n_days: int, outdir: Path) -> pd.DataFrame:
    summer = df[(df["month"].isin([6, 7, 8, 9])) & (df["is_weekend"] == 0)].copy()
    if summer.empty:
        raise ValueError("No summer weekday observations found.")

    daily_peak = (
        summer.groupby("date_local", as_index=False)
        .agg(year=("year", "first"), month=("month", "first"), daily_peak_load_mw=("load_mw", "max"))
        .sort_values(["daily_peak_load_mw", "date_local"], ascending=[False, True])
    )
    selected = daily_peak.head(top_n_days).copy()
    selected["cp_rank"] = np.arange(1, len(selected) + 1)

    event_hours = summer[(summer["date_local"].isin(selected["date_local"])) & (summer["hour"].isin([16, 17, 18, 19]))].copy()
    controls = summer[(~summer["date_local"].isin(selected["date_local"])) & (summer["hour"].isin([16, 17, 18, 19]))].copy()

    scored, model = regression_baseline(controls, event_hours, target)
    scored = scored.merge(selected[["date_local", "cp_rank", "daily_peak_load_mw"]], on="date_local", how="left")
    scored.to_csv(outdir / f"summer_cp_like_{target}_hourly.csv", index=False)

    by_day = (
        scored.groupby(["date_local", "cp_rank", "daily_peak_load_mw"], as_index=False)
        .agg(
            year=("year", "first"),
            month=("month", "first"),
            mean_actual_price=(target, "mean"),
            mean_expected_price=("expected_price", "mean"),
            mean_abnormal_price=("abnormal_price", "mean"),
            cumulative_abnormal_price=("abnormal_price", "sum"),
            mean_load_mw=("load_mw", "mean"),
            mean_temp_c=("temp_c", "mean"),
        )
        .sort_values("cp_rank")
    )
    by_day.to_csv(outdir / f"summer_cp_like_{target}_daily_summary.csv", index=False)

    meta = {
        "target": target,
        "top_n_days": int(top_n_days),
        "event_hour_count": int(len(scored)),
        "control_hour_count": int(len(controls)),
        "mean_abnormal_price_across_event_hours": float(scored["abnormal_price"].mean()),
        "cumulative_abnormal_price_across_event_hours": float(scored["abnormal_price"].sum()),
        "ols_r2_controls": float(model.rsquared),
    }
    (outdir / f"summer_cp_like_{target}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    plt.figure(figsize=(12, 5))
    plt.bar(by_day["date_local"], by_day["mean_abnormal_price"])
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Summer CP-like days: mean abnormal {target} (16:00-19:00 local)")
    plt.xlabel("Date")
    plt.ylabel("Mean actual - expected")
    plt.tight_layout()
    plt.savefig(outdir / f"summer_cp_like_{target}_mean_abnormal_by_day.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(scored["expected_price"], scored[target], alpha=0.6)
    lo = min(scored["expected_price"].min(), scored[target].min())
    hi = max(scored["expected_price"].max(), scored[target].max())
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Expected price")
    plt.ylabel("Actual price")
    plt.title(f"Summer CP-like event hours: actual vs expected {target}")
    plt.tight_layout()
    plt.savefig(outdir / f"summer_cp_like_{target}_actual_vs_expected_scatter.png", dpi=180)
    plt.close()
    return by_day


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input, args.target)
    elliott_summary = elliott_event_study(
        df=df,
        target=args.target,
        start_local=args.elliott_start,
        end_local=args.elliott_end,
        window_days=args.window_days,
        outdir=outdir,
    )
    summer_summary = summer_cp_like_study(df=df, target=args.target, top_n_days=args.top_n_summer_days, outdir=outdir)

    combined = {
        "elliott_summary_file": str(outdir / f"elliott_{args.target}_summary.json"),
        "summer_daily_summary_file": str(outdir / f"summer_cp_like_{args.target}_daily_summary.csv"),
        "target": args.target,
        "elliott_mean_abnormal_price": elliott_summary["mean_abnormal_price"],
        "summer_mean_abnormal_price": float(summer_summary["mean_abnormal_price"].mean()),
        "summer_top_n_days": int(args.top_n_summer_days),
    }
    (outdir / f"event_study_{args.target}_run_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")

    print(f"Saved outputs to: {outdir}")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
