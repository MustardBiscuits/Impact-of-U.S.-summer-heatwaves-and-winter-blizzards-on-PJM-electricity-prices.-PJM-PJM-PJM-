from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


UTC = "UTC"
LOCAL_TZ = "America/New_York"
HORIZON = 24


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    project_dir = script_path.parents[1]
    default_root = project_dir / "data" / "raw"
    default_output = project_dir / "data" / "processed"

    parser = argparse.ArgumentParser(
        description="Build cleaned PJM DOM + weather modeling tables for 24h forecasts."
    )
    parser.add_argument("--dataset-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--train-end", default="2022-12-31 23:00:00+00:00")
    parser.add_argument("--validation-end", default="2023-12-31 23:00:00+00:00")
    return parser.parse_args()


def read_csv_time(path: Path, time_col: str, rename_time_to: str = "ts") -> pd.DataFrame:
    df = pd.read_csv(path)
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    if time_col != rename_time_to:
        df = df.rename(columns={time_col: rename_time_to})
    return df


def numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def clean_continuous_feature(
    df: pd.DataFrame,
    source_col: str,
    clean_col: str,
    missing_col: str,
    *,
    interp_limit: int = 12,
    fill_zero: bool = False,
    fallback_col: str | None = None,
) -> None:
    df[missing_col] = df[source_col].isna().astype("int8")
    if fill_zero:
        df[clean_col] = df[source_col].fillna(0.0)
        return

    if fallback_col is not None:
        df[clean_col] = df[source_col].fillna(df[fallback_col])
        return

    cleaned = df[source_col].interpolate(
        method="time", limit=interp_limit, limit_direction="both"
    )
    cleaned = cleaned.ffill().bfill()
    df[clean_col] = cleaned


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    local = df.index.tz_convert(LOCAL_TZ)
    df["datetime_eastern"] = local
    df["year_eastern"] = local.year.astype("int16")
    df["month_eastern"] = local.month.astype("int8")
    df["day_eastern"] = local.day.astype("int8")
    df["hour_eastern"] = local.hour.astype("int8")
    df["dayofweek_eastern"] = local.dayofweek.astype("int8")
    df["is_weekend"] = (local.dayofweek >= 5).astype("int8")
    df["is_summer"] = local.month.isin([6, 7, 8]).astype("int8")
    df["is_winter"] = local.month.isin([12, 1, 2, 3]).astype("int8")
    df["sin_hour"] = np.sin(2 * np.pi * df["hour_eastern"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour_eastern"] / 24.0)
    df["sin_dow"] = np.sin(2 * np.pi * df["dayofweek_eastern"] / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * df["dayofweek_eastern"] / 7.0)
    df["sin_month"] = np.sin(2 * np.pi * df["month_eastern"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month_eastern"] / 12.0)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    lag_specs = {
        "lmp_clean": [1, 24, 48, 168],
        "load_mw_clean": [1, 24, 168],
        "temp_c_clean": [1, 24, 168],
        "hdd_clean": [1, 24, 168],
        "cdd_clean": [1, 24, 168],
    }
    for col, lags in lag_specs.items():
        for lag in lags:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)

    df["lmp_roll_24h_mean"] = df["lmp_clean"].shift(1).rolling(24, min_periods=12).mean()
    df["lmp_roll_24h_std"] = df["lmp_clean"].shift(1).rolling(24, min_periods=12).std()
    df["lmp_roll_168h_mean"] = df["lmp_clean"].shift(1).rolling(168, min_periods=72).mean()
    df["load_roll_24h_mean"] = df["load_mw_clean"].shift(1).rolling(24, min_periods=12).mean()
    df["temp_roll_24h_mean"] = df["temp_c_clean"].shift(1).rolling(24, min_periods=12).mean()
    return df


def add_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    targets = {
        f"target_lmp_h{h:02d}": df["lmp_observed"].shift(-h)
        for h in range(1, horizon + 1)
    }
    target_missing = {
        f"target_lmp_h{h:02d}_is_missing": df["lmp_missing"].shift(-h)
        for h in range(1, horizon + 1)
    }
    return pd.concat([df, pd.DataFrame({**targets, **target_missing}, index=df.index)], axis=1)


def write_report(
    output_dir: Path,
    hourly: pd.DataFrame,
    supervised: pd.DataFrame,
    split_counts: pd.DataFrame,
) -> None:
    report = [
        "# PJM Weather 24h Dataset Report",
        "",
        "## Inputs",
        "- PJM day-ahead DOM LMP: `PJM/pjm_dom_da_lmp_hrl_fixed.csv`",
        "- PJM DOM load: `PJM/pjm_dom_load_hourly_2018_2025UTC.csv`",
        "- Weather model-ready observations: `weather/weather_model_ready.csv`",
        "",
        "## Output row counts",
        f"- Clean hourly table rows: {len(hourly):,}",
        f"- 24h supervised table rows: {len(supervised):,}",
        "",
        "## Split counts",
        split_counts.to_markdown(index=False),
        "",
        "## Cleaning choices",
        "- LMP gaps are time-interpolated where possible and marked by `lmp_missing`.",
        "- Target columns use observed LMP only; rows with missing 24h targets are excluded from supervised splits.",
        "- Load and continuous weather fields are interpolated over short gaps, then forward/back filled.",
        "- Precipitation missing values are filled with zero and marked with missing indicators.",
        "- Heat index and wind chill are filled with temperature when physically unavailable and marked by indicators.",
        "- Price spikes are preserved; the target is not winsorized or clipped.",
    ]
    (output_dir / "cleaning_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.dataset_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    da_path = root / "PJM" / "pjm_dom_da_lmp_hrl_fixed.csv"
    load_path = root / "PJM" / "pjm_dom_load_hourly_2018_2025UTC.csv"
    weather_path = root / "weather" / "weather_model_ready.csv"

    da = read_csv_time(da_path, "datetime_beginning_utc")
    da = numeric_frame(da, ["lmp", "congestion", "marginal_loss"])
    da = (
        da.groupby("ts", as_index=True)[["lmp", "congestion", "marginal_loss"]]
        .mean()
        .sort_index()
    )

    load = read_csv_time(load_path, "timestamp_utc")
    load = numeric_frame(load, ["load_mw"])
    load = load.groupby("ts", as_index=True)[["load_mw"]].mean().sort_index()

    weather = read_csv_time(weather_path, "ts")
    weather_cols = [
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
    weather = numeric_frame(weather, weather_cols)
    weather = weather.groupby("ts", as_index=True)[weather_cols].mean().sort_index()

    start = max(da.index.min(), load.index.min(), weather.index.min())
    end = min(da.index.max(), load.index.max(), weather.index.max())
    hourly_index = pd.date_range(start=start, end=end, freq="h", tz=UTC)

    df = pd.DataFrame(index=hourly_index)
    df.index.name = "timestamp_utc"
    df = df.join(da).join(load).join(weather)

    df["lmp_observed"] = df["lmp"]
    clean_continuous_feature(df, "lmp", "lmp_clean", "lmp_missing", interp_limit=24)
    clean_continuous_feature(df, "congestion", "congestion_clean", "congestion_missing", interp_limit=24)
    clean_continuous_feature(
        df, "marginal_loss", "marginal_loss_clean", "marginal_loss_missing", interp_limit=24
    )
    clean_continuous_feature(df, "load_mw", "load_mw_clean", "load_mw_missing", interp_limit=12)

    clean_continuous_feature(df, "temp_c", "temp_c_clean", "temp_c_missing", interp_limit=6)
    clean_continuous_feature(df, "dewpoint_c", "dewpoint_c_clean", "dewpoint_c_missing", interp_limit=6)
    clean_continuous_feature(df, "rh_pct", "rh_pct_clean", "rh_pct_missing", interp_limit=6)
    clean_continuous_feature(
        df, "heat_index_c", "heat_index_c_clean", "heat_index_c_missing", fallback_col="temp_c_clean"
    )
    clean_continuous_feature(
        df, "wind_chill_c", "wind_chill_c_clean", "wind_chill_c_missing", fallback_col="temp_c_clean"
    )
    clean_continuous_feature(
        df, "wind_speed_ms", "wind_speed_ms_clean", "wind_speed_ms_missing", interp_limit=6
    )
    clean_continuous_feature(df, "slp_hpa", "slp_hpa_clean", "slp_hpa_missing", interp_limit=6)
    clean_continuous_feature(
        df, "precip_1h_mm", "precip_1h_mm_clean", "precip_1h_mm_missing", fill_zero=True
    )
    clean_continuous_feature(
        df, "precip_6h_mm", "precip_6h_mm_clean", "precip_6h_mm_missing", fill_zero=True
    )
    clean_continuous_feature(df, "cdd", "cdd_clean", "cdd_missing", interp_limit=6)
    clean_continuous_feature(df, "hdd", "hdd_clean", "hdd_missing", interp_limit=6)
    clean_continuous_feature(df, "n_stations", "n_stations_clean", "n_stations_missing", interp_limit=6)

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_targets(df, args.horizon)
    df = df.replace([np.inf, -np.inf], np.nan)

    target_cols = [f"target_lmp_h{h:02d}" for h in range(1, args.horizon + 1)]
    target_missing_cols = [
        f"target_lmp_h{h:02d}_is_missing" for h in range(1, args.horizon + 1)
    ]
    required_feature_cols = [
        "lmp_clean_lag_168h",
        "load_mw_clean_lag_168h",
        "temp_c_clean_lag_168h",
        "hdd_clean_lag_168h",
        "cdd_clean_lag_168h",
        "lmp_roll_168h_mean",
    ]

    supervised = df.dropna(subset=target_cols + required_feature_cols).copy()
    missing_target_mask = supervised[target_missing_cols].fillna(1).sum(axis=1) > 0
    supervised = supervised.loc[~missing_target_mask].copy()

    train_end = pd.Timestamp(args.train_end).tz_convert(UTC)
    val_end = pd.Timestamp(args.validation_end).tz_convert(UTC)
    supervised["split"] = np.select(
        [
            supervised.index <= train_end,
            (supervised.index > train_end) & (supervised.index <= val_end),
            supervised.index > val_end,
        ],
        ["train", "validation", "test"],
        default="unknown",
    )

    hourly_out = df.reset_index()
    supervised_out = supervised.reset_index()
    hourly_out.to_csv(output_dir / "pjm_weather_hourly_clean.csv", index=False)
    supervised_out.to_csv(output_dir / "pjm_weather_supervised_24h.csv", index=False)
    hourly_out.to_parquet(output_dir / "pjm_weather_hourly_clean.parquet", index=False)
    supervised_out.to_parquet(output_dir / "pjm_weather_supervised_24h.parquet", index=False)

    split_counts = (
        supervised.groupby("split")
        .agg(
            rows=("lmp_clean", "size"),
            start=("lmp_clean", lambda s: supervised.loc[s.index].index.min()),
            end=("lmp_clean", lambda s: supervised.loc[s.index].index.max()),
        )
        .reset_index()
    )
    split_counts.to_csv(output_dir / "split_summary.csv", index=False)

    for split_name in ["train", "validation", "test"]:
        split_df = supervised.loc[supervised["split"] == split_name].reset_index()
        split_df.to_csv(output_dir / f"{split_name}_24h.csv", index=False)
        split_df.to_parquet(output_dir / f"{split_name}_24h.parquet", index=False)

    write_report(output_dir, df, supervised, split_counts)

    print("Built cleaned 24h modeling data")
    print(f"Output directory: {output_dir}")
    print(split_counts.to_string(index=False))


if __name__ == "__main__":
    main()
