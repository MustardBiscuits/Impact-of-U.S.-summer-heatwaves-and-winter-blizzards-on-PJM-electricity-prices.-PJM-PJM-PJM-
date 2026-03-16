#!/usr/bin/env python3
"""
Exploratory data analysis for the merged PJM DOM + weather dataset.

Default input:
    /mnt/data/pjm_dom_weather_merged_model_ready.csv

Outputs:
    /mnt/data/pjm_eda_outputs/
        - descriptive_stats.csv
        - missingness.csv
        - hourly_profile.csv
        - weekday_hour_da_lmp.csv
        - weekday_hour_rt_lmp.csv
        - monthly_summary.csv
        - pairwise_correlations.csv
        - eda_summary.json
        - histogram_*.png
        - hourly_profiles.png
        - monthly_trends.png
        - weekday_hour_heatmap_da_lmp.png
        - weekday_hour_heatmap_rt_lmp.png
        - temp_vs_price_scatter.png
        - load_vs_price_scatter.png
        - missingness_preview.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_INPUT = "/mnt/data/pjm_dom_weather_merged_model_ready.csv"
DEFAULT_OUTPUT_DIR = "/mnt/data/pjm_eda_outputs"


KEY_VARS = [
    "load_mw",
    "da_lmp",
    "rt_lmp",
    "rt_minus_da_lmp",
    "temp_c",
    "dewpoint_c",
    "rh_pct",
    "wind_speed_ms",
    "cdd",
    "hdd",
    "precip_1h_mm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA for PJM DOM + weather merged data")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input merged CSV file")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for EDA outputs")
    return parser.parse_args()


def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    if "timestamp_ept" in df.columns:
        try:
            df["timestamp_ept"] = pd.to_datetime(df["timestamp_ept"], errors="coerce", utc=True).dt.tz_convert("America/New_York")
        except Exception:
            df["timestamp_ept"] = pd.NaT
    if "timestamp_ept" not in df.columns or df["timestamp_ept"].isna().all():
        if "timestamp_utc" in df.columns:
            df["timestamp_ept"] = df["timestamp_utc"].dt.tz_convert("America/New_York")
    if "date_local" in df.columns:
        df["date_local"] = pd.to_datetime(df["date_local"], errors="coerce")
    return df


def descriptive_stats(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    n = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        s = df[col]
        rows.append(
            {
                "variable": col,
                "count": int(s.count()),
                "missing": int(s.isna().sum()),
                "missing_pct": float(s.isna().mean() * 100),
                "mean": float(s.mean()) if s.notna().any() else np.nan,
                "std": float(s.std()) if s.notna().any() else np.nan,
                "min": float(s.min()) if s.notna().any() else np.nan,
                "p25": float(s.quantile(0.25)) if s.notna().any() else np.nan,
                "median": float(s.median()) if s.notna().any() else np.nan,
                "p75": float(s.quantile(0.75)) if s.notna().any() else np.nan,
                "max": float(s.max()) if s.notna().any() else np.nan,
            }
        )
    stats = pd.DataFrame(rows).sort_values("variable")
    stats.to_csv(outdir / "descriptive_stats.csv", index=False)

    missing = (
        df.isna()
        .mean()
        .mul(100)
        .rename("missing_pct")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values("missing_pct", ascending=False)
    )
    missing.to_csv(outdir / "missingness.csv", index=False)
    return stats


def save_histograms(df: pd.DataFrame, outdir: Path) -> None:
    for col in ["load_mw", "da_lmp", "rt_lmp", "temp_c"]:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if vals.empty:
            continue
        plt.figure(figsize=(8, 5))
        plt.hist(vals, bins=50)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(outdir / f"histogram_{col}.png", dpi=180)
        plt.close()


def save_hourly_profiles(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    cols = [c for c in ["load_mw", "da_lmp", "rt_lmp", "temp_c"] if c in df.columns]
    hourly = df.groupby("hour", dropna=False)[cols].mean().reset_index()
    hourly.to_csv(outdir / "hourly_profile.csv", index=False)

    plt.figure(figsize=(10, 6))
    for col in cols:
        plt.plot(hourly["hour"], hourly[col], label=col)
    plt.xlabel("Hour of day (local)")
    plt.ylabel("Average value")
    plt.title("Average hourly profiles")
    plt.xticks(range(0, 24, 1))
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "hourly_profiles.png", dpi=180)
    plt.close()
    return hourly


def save_monthly_trends(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    temp_date_col = None
    if "timestamp_ept" in df.columns and df["timestamp_ept"].notna().any():
        temp_date_col = df["timestamp_ept"]
    elif "timestamp_utc" in df.columns and df["timestamp_utc"].notna().any():
        temp_date_col = df["timestamp_utc"]
    else:
        raise ValueError("No usable timestamp column found for monthly trends.")

    monthly = df.copy()
    monthly["year_month"] = temp_date_col.dt.to_period("M").astype(str)
    cols = [c for c in ["load_mw", "da_lmp", "rt_lmp", "temp_c"] if c in monthly.columns]
    monthly_summary = monthly.groupby("year_month")[cols].mean().reset_index()
    monthly_summary.to_csv(outdir / "monthly_summary.csv", index=False)

    plt.figure(figsize=(12, 6))
    x = np.arange(len(monthly_summary))
    for col in cols:
        plt.plot(x, monthly_summary[col], label=col)
    tick_step = max(1, len(monthly_summary) // 12)
    plt.xticks(x[::tick_step], monthly_summary["year_month"].iloc[::tick_step], rotation=45, ha="right")
    plt.xlabel("Year-month")
    plt.ylabel("Monthly average")
    plt.title("Monthly average trends")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "monthly_trends.png", dpi=180)
    plt.close()
    return monthly_summary


def _save_heatmap_matrix(matrix: pd.DataFrame, title: str, outfile: Path) -> None:
    data = matrix.to_numpy(dtype=float)
    plt.figure(figsize=(11, 4.8))
    im = plt.imshow(data, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label=matrix.columns.name or "value")
    plt.yticks(range(len(matrix.index)), matrix.index)
    plt.xticks(range(len(matrix.columns)), matrix.columns)
    plt.xlabel(matrix.columns.name or "hour")
    plt.ylabel(matrix.index.name or "weekday")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=180)
    plt.close()


def save_weekday_hour_heatmaps(df: pd.DataFrame, outdir: Path) -> None:
    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for col in ["da_lmp", "rt_lmp"]:
        if col not in df.columns:
            continue
        pivot = df.pivot_table(index="weekday", columns="hour", values=col, aggfunc="mean")
        pivot = pivot.reindex(range(7))
        pivot.index = weekday_labels
        pivot.to_csv(outdir / f"weekday_hour_{col}.csv")
        _save_heatmap_matrix(pivot, f"Average {col} by weekday and hour", outdir / f"weekday_hour_heatmap_{col}.png")


def save_scatter_plots(df: pd.DataFrame, outdir: Path) -> None:
    sample_n = min(12000, len(df))
    sample = df.sample(sample_n, random_state=42) if len(df) > sample_n else df.copy()

    if {"temp_c", "da_lmp", "rt_lmp"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        plt.scatter(sample["temp_c"], sample["da_lmp"], s=8, alpha=0.35, label="DA")
        plt.scatter(sample["temp_c"], sample["rt_lmp"], s=8, alpha=0.35, label="RT")
        plt.xlabel("Temperature (C)")
        plt.ylabel("Price ($/MWh)")
        plt.title("Temperature vs price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "temp_vs_price_scatter.png", dpi=180)
        plt.close()

    if {"load_mw", "da_lmp", "rt_lmp"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        plt.scatter(sample["load_mw"], sample["da_lmp"], s=8, alpha=0.35, label="DA")
        plt.scatter(sample["load_mw"], sample["rt_lmp"], s=8, alpha=0.35, label="RT")
        plt.xlabel("Load (MW)")
        plt.ylabel("Price ($/MWh)")
        plt.title("Load vs price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "load_vs_price_scatter.png", dpi=180)
        plt.close()


def save_missingness_preview(df: pd.DataFrame, outdir: Path) -> None:
    cols = [c for c in KEY_VARS if c in df.columns]
    preview = df[cols].head(min(1500, len(df))).isna().astype(int).T
    if preview.empty:
        return
    plt.figure(figsize=(12, max(4, len(cols) * 0.35)))
    im = plt.imshow(preview, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="Missing (1=yes)")
    plt.yticks(range(len(preview.index)), preview.index)
    plt.xlabel("Row index preview")
    plt.ylabel("Variable")
    plt.title("Missingness preview (first rows)")
    plt.tight_layout()
    plt.savefig(outdir / "missingness_preview.png", dpi=180)
    plt.close()


def save_correlations(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    cols = [c for c in KEY_VARS if c in df.columns]
    corr = df[cols].corr(numeric_only=True)
    corr.to_csv(outdir / "pairwise_correlations.csv")
    return corr


def build_summary(df: pd.DataFrame, outdir: Path) -> dict:
    summary = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "start_utc": str(df["timestamp_utc"].min()) if "timestamp_utc" in df.columns else None,
        "end_utc": str(df["timestamp_utc"].max()) if "timestamp_utc" in df.columns else None,
        "start_ept": str(df["timestamp_ept"].min()) if "timestamp_ept" in df.columns else None,
        "end_ept": str(df["timestamp_ept"].max()) if "timestamp_ept" in df.columns else None,
        "variables_present": df.columns.tolist(),
        "missing_pct_key_vars": {},
    }
    for col in KEY_VARS:
        if col in df.columns:
            summary["missing_pct_key_vars"][col] = round(float(df[col].isna().mean() * 100), 4)
    with open(outdir / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    df = ensure_datetime(df)

    if "hour" not in df.columns:
        if "timestamp_ept" in df.columns:
            df["hour"] = df["timestamp_ept"].dt.hour
        elif "timestamp_utc" in df.columns:
            df["hour"] = df["timestamp_utc"].dt.hour
    if "weekday" not in df.columns:
        if "timestamp_ept" in df.columns:
            df["weekday"] = df["timestamp_ept"].dt.weekday
        elif "timestamp_utc" in df.columns:
            df["weekday"] = df["timestamp_utc"].dt.weekday

    stats = descriptive_stats(df, outdir)
    save_histograms(df, outdir)
    hourly = save_hourly_profiles(df, outdir)
    monthly = save_monthly_trends(df, outdir)
    save_weekday_hour_heatmaps(df, outdir)
    save_scatter_plots(df, outdir)
    save_missingness_preview(df, outdir)
    corr = save_correlations(df, outdir)
    summary = build_summary(df, outdir)

    print("EDA completed.")
    print(f"Input: {input_path}")
    print(f"Output directory: {outdir}")
    print(f"Rows: {summary['n_rows']}, Columns: {summary['n_columns']}")
    print("Top missing columns:")
    top_missing = df.isna().mean().sort_values(ascending=False).head(10)
    for col, pct in top_missing.items():
        print(f"  - {col}: {pct*100:.2f}% missing")
    print("Key variable means:")
    for col in ["load_mw", "da_lmp", "rt_lmp", "temp_c"]:
        if col in df.columns:
            print(f"  - {col}: {df[col].mean():.3f}")


if __name__ == "__main__":
    main()
