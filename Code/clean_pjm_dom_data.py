#!/usr/bin/env python3
"""
Clean and merge PJM DOM hourly load, day-ahead LMP, and real-time LMP.

What this script does
---------------------
1. Reads three PJM-formatted CSVs.
2. Standardizes timestamps into a single UTC datetime column.
3. Converts UTC to America/New_York (EPT/local time).
4. Builds local calendar features: year, month, day, hour, weekday, weekend.
5. Left-joins load with DA LMP and RT LMP on an hourly UTC timestamp.
6. Writes a merged modeling-ready CSV and a small QA report.

Default inputs are matched to the files currently in /mnt/data.

Example
-------
python clean_pjm_dom_data.py \
  --load /mnt/data/pjm_dom_load_hourly_2018_2025UTC.csv \
  --da   /mnt/data/pjm_dom_da_lmp_hrl_fixed.csv \
  --rt   /mnt/data/pjm_dom_rt_lmp_hrl_merged.csv \
  --out  /mnt/data/pjm_dom_merged_model_ready.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

LOCAL_TZ = "America/New_York"


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _parse_utc(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts


def _dedupe_on_timestamp(df: pd.DataFrame, name: str) -> pd.DataFrame:
    before = len(df)
    df = df.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"], keep="last")
    after = len(df)
    if before != after:
        print(f"[{name}] dropped {before - after} duplicate timestamp rows")
    return df


def _add_local_time_features(df: pd.DataFrame) -> pd.DataFrame:
    local_ts = df["timestamp_utc"].dt.tz_convert(LOCAL_TZ)
    df["timestamp_ept"] = local_ts
    df["date_ept"] = local_ts.dt.date.astype(str)
    df["year"] = local_ts.dt.year
    df["month"] = local_ts.dt.month
    df["day"] = local_ts.dt.day
    df["hour"] = local_ts.dt.hour
    df["weekday"] = local_ts.dt.dayofweek  # Monday=0
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["iso_week"] = local_ts.dt.isocalendar().week.astype(int)
    return df


def load_load_data(path: str | Path) -> pd.DataFrame:
    df = _read_csv(path)
    expected = {"timestamp_utc", "load_mw"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Load file missing columns: {sorted(missing)}")

    out = df.copy()
    out["timestamp_utc"] = _parse_utc(out["timestamp_utc"])
    out = out.loc[out["timestamp_utc"].notna(), ["timestamp_utc", "load_mw"]].copy()
    out["load_mw"] = pd.to_numeric(out["load_mw"], errors="coerce")
    out = _dedupe_on_timestamp(out, "load")
    return out.sort_values("timestamp_utc")


def load_da_data(path: str | Path, zone: str = "DOM") -> pd.DataFrame:
    df = _read_csv(path)
    expected = {"datetime_beginning_utc", "zone", "lmp", "congestion", "marginal_loss"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"DA file missing columns: {sorted(missing)}")

    out = df.copy()
    out = out[out["zone"].astype(str).str.upper() == zone.upper()].copy()
    out["timestamp_utc"] = _parse_utc(out["datetime_beginning_utc"])
    rename_map = {
        "lmp": "da_lmp",
        "congestion": "da_congestion",
        "marginal_loss": "da_marginal_loss",
    }
    keep = ["timestamp_utc", "zone", "lmp", "congestion", "marginal_loss"]
    out = out.loc[out["timestamp_utc"].notna(), keep].rename(columns=rename_map)
    out = _dedupe_on_timestamp(out, "day_ahead")
    return out.sort_values("timestamp_utc")


def load_rt_data(path: str | Path, zone: str = "DOM") -> pd.DataFrame:
    df = _read_csv(path)
    expected = {"datetime_beginning_utc", "zone", "rt_lmp", "rt_congestion", "rt_marginal_loss"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"RT file missing columns: {sorted(missing)}")

    out = df.copy()
    out = out[out["zone"].astype(str).str.upper() == zone.upper()].copy()
    out["timestamp_utc"] = _parse_utc(out["datetime_beginning_utc"])
    keep = ["timestamp_utc", "zone", "rt_lmp", "rt_congestion", "rt_marginal_loss"]
    out = out.loc[out["timestamp_utc"].notna(), keep]
    out = _dedupe_on_timestamp(out, "real_time")
    return out.sort_values("timestamp_utc")


def build_qa_report(df: pd.DataFrame, out_json: str | Path) -> None:
    report = {
        "rows": int(len(df)),
        "start_utc": None if df.empty else df["timestamp_utc"].min().isoformat(),
        "end_utc": None if df.empty else df["timestamp_utc"].max().isoformat(),
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "duplicate_timestamp_rows": int(df.duplicated(subset=["timestamp_utc"]).sum()),
    }
    Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and merge PJM DOM hourly datasets")
    parser.add_argument("--load", default="/mnt/data/pjm_dom_load_hourly_2018_2025UTC.csv")
    parser.add_argument("--da", default="/mnt/data/pjm_dom_da_lmp_hrl_fixed.csv")
    parser.add_argument("--rt", default="/mnt/data/pjm_dom_rt_lmp_hrl_merged.csv")
    parser.add_argument("--zone", default="DOM")
    parser.add_argument("--out", default="/mnt/data/pjm_dom_merged_model_ready.csv")
    args = parser.parse_args()

    load_df = load_load_data(args.load)
    da_df = load_da_data(args.da, zone=args.zone)
    rt_df = load_rt_data(args.rt, zone=args.zone)

    merged = (
        load_df
        .merge(da_df.drop(columns=["zone"]), on="timestamp_utc", how="left")
        .merge(rt_df.drop(columns=["zone"]), on="timestamp_utc", how="left")
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    merged = _add_local_time_features(merged)

    # Put time columns first for readability.
    preferred_order = [
        "timestamp_utc", "timestamp_ept", "date_ept",
        "year", "month", "day", "hour", "weekday", "is_weekend", "iso_week",
        "load_mw", "da_lmp", "da_congestion", "da_marginal_loss",
        "rt_lmp", "rt_congestion", "rt_marginal_loss",
    ]
    cols = [c for c in preferred_order if c in merged.columns] + [c for c in merged.columns if c not in preferred_order]
    merged = merged[cols]

    out_path = Path(args.out)
    merged.to_csv(out_path, index=False)
    build_qa_report(merged, out_path.with_suffix(".qa.json"))

    print(f"Saved merged PJM file to: {out_path}")
    print(f"Saved QA report to: {out_path.with_suffix('.qa.json')}")
    print(merged.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
