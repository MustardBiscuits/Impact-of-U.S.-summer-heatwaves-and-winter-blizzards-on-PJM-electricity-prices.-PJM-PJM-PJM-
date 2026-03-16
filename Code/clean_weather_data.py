#!/usr/bin/env python3
"""
Clean hourly weather data for modeling.

This script supports two modes automatically:

Mode A: station-level raw weather input
---------------------------------------
If the input file contains a station column, the script will:
1. infer timestamp and station columns,
2. normalize to one row per station per hour,
3. coerce meteorological fields to numeric,
4. compute derived metrics when possible,
5. aggregate hourly across stations,
6. write a model-ready regional weather file plus station coverage tables.

Mode B: already-aggregated weather input
----------------------------------------
If the input file already looks like your `weather_model_ready.csv`, the script
will standardize timestamps, sort, de-duplicate, lightly clean numeric fields,
and save a cleaned version.

Expected raw columns (flexible names are supported)
--------------------------------------------------
Timestamp: ts / time / datetime / date
Station:   station / station_id / usaf_wban / site
Optional numeric fields:
- temp_c / temperature / air_temp
- dewpoint_c / dew_point
- rh_pct / relative_humidity
- wind_speed_ms / wind_speed / wspd
- slp_hpa / pressure / sea_level_pressure
- precip_1h_mm / precip / rain
- precip_6h_mm

Example
-------
python clean_weather_data.py \
  --input /path/to/weather_raw.csv \
  --out-prefix /mnt/data/weather

python clean_weather_data.py \
  --input /mnt/data/weather_model_ready.csv \
  --out-prefix /mnt/data/weather
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TIME_CANDIDATES = [
    "ts", "timestamp", "time", "datetime", "date_time", "datehour", "valid", "datetime_utc", "time_utc"
]
STATION_CANDIDATES = [
    "station", "station_id", "site", "site_id", "usaf_wban", "icao", "name"
]
COLUMN_ALIASES = {
    "temp_c": ["temp_c", "temperature_c", "temperature", "air_temp_c", "air_temp", "temp"],
    "dewpoint_c": ["dewpoint_c", "dew_point_c", "dewpoint", "dew_point"],
    "rh_pct": ["rh_pct", "relative_humidity", "humidity", "rh", "relative_humidity_pct"],
    "wind_speed_ms": ["wind_speed_ms", "wind_speed", "wspd", "wind_ms", "wind_speed_mps"],
    "slp_hpa": ["slp_hpa", "sea_level_pressure", "pressure_hpa", "pressure", "slp"],
    "precip_1h_mm": ["precip_1h_mm", "precip_mm", "precip", "rain_mm", "precip_1h"],
    "precip_6h_mm": ["precip_6h_mm", "precip6h_mm", "rain_6h_mm", "precip_6h"],
    "heat_index_c": ["heat_index_c", "heat_index"],
    "wind_chill_c": ["wind_chill_c", "wind_chill"],
    "n_stations": ["n_stations"],
}


def _find_first(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _rename_aliases(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    cols = list(df.columns)
    for target, aliases in COLUMN_ALIASES.items():
        found = _find_first(cols, aliases)
        if found is not None and found != target:
            rename[found] = target
    if rename:
        df = df.rename(columns=rename)
    return df


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _parse_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _calc_rh_from_temp_dewpoint(temp_c: pd.Series, dewpoint_c: pd.Series) -> pd.Series:
    # Magnus approximation
    a, b = 17.625, 243.04
    gamma_td = (a * dewpoint_c) / (b + dewpoint_c)
    gamma_t = (a * temp_c) / (b + temp_c)
    rh = 100 * np.exp(gamma_td - gamma_t)
    return rh.clip(lower=0, upper=100)


def _calc_heat_index_c(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    # Approximation via Rothfusz in F, only meaningful for warm/humid conditions.
    t_f = temp_c * 9 / 5 + 32
    rh = rh_pct
    hi_f = (
        -42.379 + 2.04901523 * t_f + 10.14333127 * rh - 0.22475541 * t_f * rh
        - 6.83783e-3 * t_f**2 - 5.481717e-2 * rh**2
        + 1.22874e-3 * t_f**2 * rh + 8.5282e-4 * t_f * rh**2
        - 1.99e-6 * t_f**2 * rh**2
    )
    hi_c = (hi_f - 32) * 5 / 9
    hi_c = hi_c.where((temp_c >= 27) & (rh_pct >= 40))
    return hi_c


def _calc_wind_chill_c(temp_c: pd.Series, wind_speed_ms: pd.Series) -> pd.Series:
    v_kmh = wind_speed_ms * 3.6
    wc = 13.12 + 0.6215 * temp_c - 11.37 * (v_kmh ** 0.16) + 0.3965 * temp_c * (v_kmh ** 0.16)
    wc = wc.where((temp_c <= 10) & (v_kmh > 4.8))
    return wc


def _clean_aggregated_input(df: pd.DataFrame) -> pd.DataFrame:
    df = _rename_aliases(df.copy())
    ts_col = _find_first(df.columns, TIME_CANDIDATES + ["ts_utc"])
    if ts_col is None:
        raise ValueError("Could not infer timestamp column in aggregated weather input.")
    if ts_col != "ts":
        df = df.rename(columns={ts_col: "ts"})
    df["ts"] = _parse_ts(df["ts"])
    df = df.loc[df["ts"].notna()].copy()

    numeric_cols = [c for c in COLUMN_ALIASES if c in df.columns and c != "n_stations"] + (["n_stations"] if "n_stations" in df.columns else [])
    df = _coerce_numeric(df, numeric_cols)
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")

    if "rh_pct" not in df.columns and {"temp_c", "dewpoint_c"}.issubset(df.columns):
        df["rh_pct"] = _calc_rh_from_temp_dewpoint(df["temp_c"], df["dewpoint_c"])
    if "heat_index_c" not in df.columns and {"temp_c", "rh_pct"}.issubset(df.columns):
        df["heat_index_c"] = _calc_heat_index_c(df["temp_c"], df["rh_pct"])
    if "wind_chill_c" not in df.columns and {"temp_c", "wind_speed_ms"}.issubset(df.columns):
        df["wind_chill_c"] = _calc_wind_chill_c(df["temp_c"], df["wind_speed_ms"])
    if "cdd" not in df.columns and "temp_c" in df.columns:
        df["cdd"] = (df["temp_c"] - 18.0).clip(lower=0)
    if "hdd" not in df.columns and "temp_c" in df.columns:
        df["hdd"] = (18.0 - df["temp_c"]).clip(lower=0)

    ordered = [
        "ts", "temp_c", "dewpoint_c", "rh_pct", "heat_index_c", "wind_chill_c",
        "wind_speed_ms", "slp_hpa", "precip_1h_mm", "precip_6h_mm", "cdd", "hdd", "n_stations"
    ]
    cols = [c for c in ordered if c in df.columns] + [c for c in df.columns if c not in ordered]
    return df[cols]


def _clean_station_input(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _rename_aliases(df.copy())
    ts_col = _find_first(df.columns, TIME_CANDIDATES)
    station_col = _find_first(df.columns, STATION_CANDIDATES)
    if ts_col is None or station_col is None:
        raise ValueError("Raw weather input must contain both a timestamp column and a station column.")

    if ts_col != "ts":
        df = df.rename(columns={ts_col: "ts"})
    if station_col != "station":
        df = df.rename(columns={station_col: "station"})

    df["ts"] = _parse_ts(df["ts"])
    df = df.loc[df["ts"].notna()].copy()
    df["ts"] = df["ts"].dt.floor("H")

    numeric_candidates = [
        "temp_c", "dewpoint_c", "rh_pct", "wind_speed_ms", "slp_hpa",
        "precip_1h_mm", "precip_6h_mm", "heat_index_c", "wind_chill_c"
    ]
    df = _coerce_numeric(df, [c for c in numeric_candidates if c in df.columns])

    agg_map = {}
    for c in ["temp_c", "dewpoint_c", "rh_pct", "wind_speed_ms", "slp_hpa", "heat_index_c", "wind_chill_c"]:
        if c in df.columns:
            agg_map[c] = "mean"
    for c in ["precip_1h_mm", "precip_6h_mm"]:
        if c in df.columns:
            agg_map[c] = "max"

    station_hourly = (
        df.groupby(["station", "ts"], as_index=False)
          .agg(agg_map)
          .sort_values(["station", "ts"])
    )

    if "rh_pct" not in station_hourly.columns and {"temp_c", "dewpoint_c"}.issubset(station_hourly.columns):
        station_hourly["rh_pct"] = _calc_rh_from_temp_dewpoint(station_hourly["temp_c"], station_hourly["dewpoint_c"])
    if "heat_index_c" not in station_hourly.columns and {"temp_c", "rh_pct"}.issubset(station_hourly.columns):
        station_hourly["heat_index_c"] = _calc_heat_index_c(station_hourly["temp_c"], station_hourly["rh_pct"])
    if "wind_chill_c" not in station_hourly.columns and {"temp_c", "wind_speed_ms"}.issubset(station_hourly.columns):
        station_hourly["wind_chill_c"] = _calc_wind_chill_c(station_hourly["temp_c"], station_hourly["wind_speed_ms"])
    if "temp_c" in station_hourly.columns:
        station_hourly["cdd"] = (station_hourly["temp_c"] - 18.0).clip(lower=0)
        station_hourly["hdd"] = (18.0 - station_hourly["temp_c"]).clip(lower=0)

    region_agg = {}
    for c in [
        "temp_c", "dewpoint_c", "rh_pct", "heat_index_c", "wind_chill_c",
        "wind_speed_ms", "slp_hpa", "cdd", "hdd"
    ]:
        if c in station_hourly.columns:
            region_agg[c] = "mean"
    for c in ["precip_1h_mm", "precip_6h_mm"]:
        if c in station_hourly.columns:
            region_agg[c] = "sum"

    weather_model_ready = (
        station_hourly.groupby("ts", as_index=False)
        .agg(region_agg)
        .sort_values("ts")
    )
    weather_model_ready["n_stations"] = station_hourly.groupby("ts")["station"].nunique().values

    hours_per_station = (
        station_hourly.groupby("station", as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        .sort_values(["rows", "station"], ascending=[False, True])
    )

    span_per_station = (
        station_hourly.groupby("station", as_index=False)
        .agg(min=("ts", "min"), max=("ts", "max"))
        .sort_values("station")
    )

    return weather_model_ready, hours_per_station, span_per_station


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean weather data into model-ready hourly format")
    parser.add_argument("--input", default="/mnt/data/weather_model_ready.csv")
    parser.add_argument("--out-prefix", default="/mnt/data/weather")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_prefix = Path(args.out_prefix)
    df = pd.read_csv(input_path)

    station_col = _find_first(df.columns, STATION_CANDIDATES)
    if station_col is not None:
        weather_model_ready, hours_per_station, span_per_station = _clean_station_input(df)
        weather_model_ready.to_csv(out_prefix.with_name(out_prefix.name + "_model_ready.csv"), index=False)
        hours_per_station.to_csv(out_prefix.with_name(out_prefix.name + "_hours_per_station.csv"), index=False)
        span_per_station.to_csv(out_prefix.with_name(out_prefix.name + "_span_per_station.csv"), index=False)
        print(f"Saved aggregated weather file to: {out_prefix.with_name(out_prefix.name + '_model_ready.csv')}")
        print(f"Saved station coverage table to: {out_prefix.with_name(out_prefix.name + '_hours_per_station.csv')}")
        print(f"Saved station span table to: {out_prefix.with_name(out_prefix.name + '_span_per_station.csv')}")
        print(weather_model_ready.head(3).to_string(index=False))
    else:
        cleaned = _clean_aggregated_input(df)
        out_file = out_prefix.with_name(out_prefix.name + "_model_ready_cleaned.csv")
        cleaned.to_csv(out_file, index=False)
        print(f"Saved cleaned aggregated weather file to: {out_file}")
        print(cleaned.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
