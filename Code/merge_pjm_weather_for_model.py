#!/usr/bin/env python3
"""
merge_pjm_weather_for_model.py

Purpose
-------
Create one model-ready hourly panel for the PJM DOM study by merging:
1) DOM hourly load
2) DOM day-ahead LMP
3) DOM real-time LMP
4) hourly weather aggregates

The script is intentionally self-contained. It can either:
- merge from the three raw PJM source files + weather, or
- read an already merged PJM file and then attach weather.

Default file names are matched to the files currently used in this project.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


DEFAULT_LOAD = "/mnt/data/pjm_dom_load_hourly_2018_2025UTC.csv"
DEFAULT_DA = "/mnt/data/pjm_dom_da_lmp_hrl_fixed.csv"
DEFAULT_RT = "/mnt/data/pjm_dom_rt_lmp_hrl_merged.csv"
DEFAULT_WEATHER = "/mnt/data/weather_model_ready.csv"
DEFAULT_OUTPUT = "/mnt/data/pjm_dom_weather_merged_model_ready.csv"
DEFAULT_QA = "/mnt/data/pjm_dom_weather_merged_model_ready.qa.json"
LOCAL_TZ = "America/New_York"


TIME_CANDIDATES = [
    "timestamp_utc",
    "datetime_beginning_utc",
    "ts",
    "datetime_utc",
    "timestamp",
    "datetime",
    "date",
]


PJM_KEEP_ORDER = [
    "timestamp_utc",
    "timestamp_ept",
    "date_local",
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "is_weekend",
    "day_of_year",
    "load_mw",
    "da_lmp",
    "da_congestion",
    "da_marginal_loss",
    "rt_lmp",
    "rt_congestion",
    "rt_marginal_loss",
    "rt_minus_da_lmp",
]

WEATHER_PREFERRED_ORDER = [
    "temp_c",
    "dewpoint_c",
    "rh_pct",
    "heat_index_c",
    "wind_chill_c",
    "wind_speed_ms",
    "slp_hpa",
    "precip_1h_mm",
    "precip_6h_mm",
    "cdd",
    "hdd",
    "n_stations",
]


def _first_existing(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def _standardize_timestamp(df: pd.DataFrame, name: str = "timestamp_utc") -> pd.DataFrame:
    src = _first_existing(df.columns, TIME_CANDIDATES)
    if src is None:
        raise ValueError(
            f"Could not find a timestamp column. Available columns: {list(df.columns)}"
        )
    out = df.copy()
    out[name] = pd.to_datetime(out[src], utc=True, errors="coerce")
    out = out.dropna(subset=[name]).sort_values(name).drop_duplicates(subset=[name])
    if src != name:
        out = out.drop(columns=[src])
    return out


def _prepare_load(df: pd.DataFrame) -> pd.DataFrame:
    out = _standardize_timestamp(df)
    if "load_mw" not in out.columns:
        numeric_candidates = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
        if len(numeric_candidates) == 1:
            out = out.rename(columns={numeric_candidates[0]: "load_mw"})
        else:
            raise ValueError("load_mw column not found and could not infer it.")
    return out[["timestamp_utc", "load_mw"]]


def _prepare_da(df: pd.DataFrame) -> pd.DataFrame:
    out = _standardize_timestamp(df)
    rename_map = {
        "lmp": "da_lmp",
        "congestion": "da_congestion",
        "marginal_loss": "da_marginal_loss",
    }
    out = out.rename(columns=rename_map)
    keep = [c for c in ["timestamp_utc", "da_lmp", "da_congestion", "da_marginal_loss"] if c in out.columns]
    missing = {"da_lmp", "da_congestion", "da_marginal_loss"} - set(keep)
    if missing:
        raise ValueError(f"Missing DA columns after rename: {sorted(missing)}")
    return out[keep]


def _prepare_rt(df: pd.DataFrame) -> pd.DataFrame:
    out = _standardize_timestamp(df)
    keep = [c for c in ["timestamp_utc", "rt_lmp", "rt_congestion", "rt_marginal_loss"] if c in out.columns]
    missing = {"rt_lmp", "rt_congestion", "rt_marginal_loss"} - set(keep)
    if missing:
        raise ValueError(f"Missing RT columns: {sorted(missing)}")
    return out[keep]


def _prepare_weather(df: pd.DataFrame) -> pd.DataFrame:
    out = _standardize_timestamp(df)
    weather_cols = [c for c in WEATHER_PREFERRED_ORDER if c in out.columns]
    if not weather_cols:
        numeric_cols = [c for c in out.columns if c != "timestamp_utc" and pd.api.types.is_numeric_dtype(out[c])]
        weather_cols = numeric_cols
    return out[["timestamp_utc"] + weather_cols]


def _build_pjm_from_raw(load_path: str, da_path: str, rt_path: str) -> pd.DataFrame:
    load = _prepare_load(_read_csv(load_path))
    da = _prepare_da(_read_csv(da_path))
    rt = _prepare_rt(_read_csv(rt_path))

    merged = load.merge(da, on="timestamp_utc", how="left")
    merged = merged.merge(rt, on="timestamp_utc", how="left")
    return merged


def _prepare_existing_pjm(df: pd.DataFrame) -> pd.DataFrame:
    out = _standardize_timestamp(df)

    rename_map = {}
    if "lmp" in out.columns and "da_lmp" not in out.columns:
        rename_map["lmp"] = "da_lmp"
    if "congestion" in out.columns and "da_congestion" not in out.columns:
        rename_map["congestion"] = "da_congestion"
    if "marginal_loss" in out.columns and "da_marginal_loss" not in out.columns:
        rename_map["marginal_loss"] = "da_marginal_loss"
    out = out.rename(columns=rename_map)

    required_any = ["load_mw", "da_lmp", "rt_lmp"]
    missing = [c for c in required_any if c not in out.columns]
    if missing:
        raise ValueError(f"Existing PJM file is missing required columns: {missing}")

    wanted = [
        c for c in [
            "timestamp_utc",
            "load_mw",
            "da_lmp",
            "da_congestion",
            "da_marginal_loss",
            "rt_lmp",
            "rt_congestion",
            "rt_marginal_loss",
        ] if c in out.columns
    ]
    return out[wanted].copy()


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    local = out["timestamp_utc"].dt.tz_convert(LOCAL_TZ)
    out["timestamp_ept"] = local
    out["date_local"] = local.dt.date.astype(str)
    out["year"] = local.dt.year
    out["month"] = local.dt.month
    out["day"] = local.dt.day
    out["hour"] = local.dt.hour
    out["weekday"] = local.dt.dayofweek
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    out["day_of_year"] = local.dt.dayofyear
    if {"rt_lmp", "da_lmp"}.issubset(out.columns):
        out["rt_minus_da_lmp"] = out["rt_lmp"] - out["da_lmp"]
    return out


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    front = [c for c in PJM_KEEP_ORDER if c in df.columns]
    weather = [c for c in WEATHER_PREFERRED_ORDER if c in df.columns]
    rest = [c for c in df.columns if c not in set(front + weather)]
    return df[front + weather + rest]


def build_dataset(
    load_path: Optional[str],
    da_path: Optional[str],
    rt_path: Optional[str],
    weather_path: str,
    pjm_premerged_path: Optional[str] = None,
) -> pd.DataFrame:
    if pjm_premerged_path:
        pjm = _prepare_existing_pjm(_read_csv(pjm_premerged_path))
    else:
        if not (load_path and da_path and rt_path):
            raise ValueError("Need either --pjm-premerged or all of --load --da --rt.")
        pjm = _build_pjm_from_raw(load_path, da_path, rt_path)

    weather = _prepare_weather(_read_csv(weather_path))
    merged = pjm.merge(weather, on="timestamp_utc", how="left")
    merged = _add_time_features(merged)
    merged = merged.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"])
    merged = _reorder_columns(merged)
    return merged


def make_qa(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "start_utc": None if df.empty else str(df["timestamp_utc"].min()),
        "end_utc": None if df.empty else str(df["timestamp_utc"].max()),
        "duplicate_timestamps": int(df["timestamp_utc"].duplicated().sum()),
        "missing_counts": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge PJM DOM market data with weather.")
    parser.add_argument("--load", default=DEFAULT_LOAD, help="Path to DOM hourly load CSV.")
    parser.add_argument("--da", default=DEFAULT_DA, help="Path to DOM day-ahead LMP CSV.")
    parser.add_argument("--rt", default=DEFAULT_RT, help="Path to DOM real-time LMP CSV.")
    parser.add_argument("--weather", default=DEFAULT_WEATHER, help="Path to weather model-ready CSV.")
    parser.add_argument(
        "--pjm-premerged",
        default=None,
        help="Optional already merged PJM CSV. If given, --load/--da/--rt are ignored.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path.")
    parser.add_argument("--qa", default=DEFAULT_QA, help="QA JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = build_dataset(
        load_path=args.load,
        da_path=args.da,
        rt_path=args.rt,
        weather_path=args.weather,
        pjm_premerged_path=args.pjm_premerged,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    qa = make_qa(df)
    qa_path = Path(args.qa)
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")

    print(f"Saved merged dataset to: {output_path}")
    print(f"Saved QA summary to: {qa_path}")
    print(f"Rows: {len(df):,}")
    print(f"Range (UTC): {qa['start_utc']} -> {qa['end_utc']}")


if __name__ == "__main__":
    main()
