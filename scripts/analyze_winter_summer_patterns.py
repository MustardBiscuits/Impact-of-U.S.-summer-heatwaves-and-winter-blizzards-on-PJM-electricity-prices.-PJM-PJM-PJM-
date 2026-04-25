from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


UTC = "UTC"
LOCAL_TZ = "America/New_York"
MODEL_COLS = [
    "seasonal_naive_24h",
    "seasonal_naive_168h",
    "dmdc",
    "dmdc_residual_24h",
    "sarimax",
]


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    work_dir = script_path.parents[1]
    parser = argparse.ArgumentParser(
        description="Analyze PJM winter storm and summer price patterns."
    )
    parser.add_argument("--processed-dir", type=Path, default=work_dir / "data" / "processed")
    parser.add_argument("--reports-dir", type=Path, default=work_dir / "reports")
    parser.add_argument("--min-event-hours", type=int, default=6)
    parser.add_argument("--max-gap-hours", type=int, default=3)
    return parser.parse_args()


def load_hourly(processed_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(processed_dir / "pjm_weather_hourly_clean.parquet")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.set_index("timestamp_utc").sort_index()
    local = df.index.tz_convert(LOCAL_TZ)
    df["datetime_eastern"] = local
    df["year_eastern"] = local.year
    df["month_eastern"] = local.month
    df["hour_eastern"] = local.hour
    df["date_eastern"] = local.date
    df["winter_year_eastern"] = np.where(
        df["month_eastern"] == 12, df["year_eastern"] + 1, df["year_eastern"]
    )
    df["temp_change_24h_c"] = df["temp_c_clean"] - df["temp_c_clean"].shift(24)
    df["hdd_change_24h"] = df["hdd_clean"] - df["hdd_clean"].shift(24)
    df["cdd_change_24h"] = df["cdd_clean"] - df["cdd_clean"].shift(24)
    df["load_change_24h_mw"] = df["load_mw_clean"] - df["load_mw_clean"].shift(24)
    return df


def contiguous_events(
    df: pd.DataFrame,
    candidate: pd.Series,
    event_prefix: str,
    min_event_hours: int,
    max_gap_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = list(df.index[candidate.fillna(False)])
    event_rows = []
    hour_rows = []
    if not times:
        return pd.DataFrame(), pd.DataFrame()

    groups: list[list[pd.Timestamp]] = []
    current = [times[0]]
    for ts in times[1:]:
        gap_hours = (ts - current[-1]) / pd.Timedelta(hours=1)
        if gap_hours <= max_gap_hours + 1:
            current.append(ts)
        else:
            groups.append(current)
            current = [ts]
    groups.append(current)

    normal_price = (
        df.groupby(["month_eastern", "hour_eastern"])["lmp_clean"]
        .transform("median")
        .rename("normal_lmp")
    )
    enriched = df.copy()
    enriched["normal_lmp"] = normal_price
    enriched["price_premium_vs_normal"] = enriched["lmp_clean"] - enriched["normal_lmp"]

    event_number = 0
    for group in groups:
        start = min(group)
        end = max(group)
        duration_hours = int((end - start) / pd.Timedelta(hours=1)) + 1
        if duration_hours < min_event_hours:
            continue
        event_number += 1
        event_id = f"{event_prefix}_{event_number:03d}"
        event_df = enriched.loc[start:end].copy()
        peak_ts = event_df["lmp_clean"].idxmax()
        event_rows.append(
            {
                "event_id": event_id,
                "start_utc": start,
                "end_utc": end,
                "start_eastern": start.tz_convert(LOCAL_TZ),
                "end_eastern": end.tz_convert(LOCAL_TZ),
                "duration_hours": duration_hours,
                "mean_lmp": event_df["lmp_clean"].mean(),
                "max_lmp": event_df["lmp_clean"].max(),
                "peak_utc": peak_ts,
                "peak_eastern": peak_ts.tz_convert(LOCAL_TZ),
                "mean_price_premium_vs_normal": event_df["price_premium_vs_normal"].mean(),
                "max_price_premium_vs_normal": event_df["price_premium_vs_normal"].max(),
                "mean_load_mw": event_df["load_mw_clean"].mean(),
                "max_load_mw": event_df["load_mw_clean"].max(),
                "mean_temp_c": event_df["temp_c_clean"].mean(),
                "min_temp_c": event_df["temp_c_clean"].min(),
                "max_temp_c": event_df["temp_c_clean"].max(),
                "sum_precip_1h_mm": event_df["precip_1h_mm_clean"].sum(),
                "max_wind_speed_ms": event_df["wind_speed_ms_clean"].max(),
                "mean_hdd": event_df["hdd_clean"].mean(),
                "mean_cdd": event_df["cdd_clean"].mean(),
            }
        )
        tmp = event_df.reset_index()
        tmp["event_id"] = event_id
        hour_rows.append(tmp)

    event_summary = pd.DataFrame(event_rows)
    event_hours = pd.concat(hour_rows, ignore_index=True) if hour_rows else pd.DataFrame()
    return event_summary, event_hours


def detect_winter_storms(
    df: pd.DataFrame, min_event_hours: int, max_gap_hours: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    winter = df["month_eastern"].isin([12, 1, 2, 3])
    winter_df = df.loc[winter]
    winter_lmp_p95 = winter_df["lmp_clean"].quantile(0.95)
    winter_wind_p85 = winter_df["wind_speed_ms_clean"].quantile(0.85)

    candidate = winter & (
        (
            (df["hdd_clean"] >= 18.3)
            & (
                (df["precip_1h_mm_clean"] >= 0.25)
                | (df["wind_speed_ms_clean"] >= winter_wind_p85)
            )
        )
        | ((df["lmp_clean"] >= winter_lmp_p95) & (df["hdd_clean"] >= 12.0))
    )
    return contiguous_events(df, candidate, "winter_storm", min_event_hours, max_gap_hours)


def analyze_winter(df: pd.DataFrame) -> pd.DataFrame:
    winter = df["month_eastern"].isin([12, 1, 2, 3])
    winter_df = df.loc[winter].copy()
    winter_lmp_p95 = winter_df["lmp_clean"].quantile(0.95)
    return (
        winter_df.groupby("winter_year_eastern")
        .agg(
            hours=("lmp_clean", "size"),
            mean_lmp=("lmp_clean", "mean"),
            median_lmp=("lmp_clean", "median"),
            p95_lmp=("lmp_clean", lambda s: s.quantile(0.95)),
            max_lmp=("lmp_clean", "max"),
            mean_load_mw=("load_mw_clean", "mean"),
            max_load_mw=("load_mw_clean", "max"),
            mean_temp_c=("temp_c_clean", "mean"),
            min_temp_c=("temp_c_clean", "min"),
            mean_hdd=("hdd_clean", "mean"),
            max_hdd=("hdd_clean", "max"),
            max_wind_speed_ms=("wind_speed_ms_clean", "max"),
            precip_1h_total_mm=("precip_1h_mm_clean", "sum"),
            high_price_hours=("lmp_clean", lambda s: int((s >= winter_lmp_p95).sum())),
        )
        .reset_index()
    )


def detect_sudden_heat_events(
    df: pd.DataFrame, min_event_hours: int, max_gap_hours: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summer = df["month_eastern"].isin([6, 7, 8])
    summer_df = df.loc[summer].copy()
    heat_jump_threshold = max(5.0, summer_df["temp_change_24h_c"].quantile(0.90))
    cdd_jump_threshold = max(2.0, summer_df["cdd_change_24h"].quantile(0.90))
    load_jump_threshold = summer_df["load_change_24h_mw"].quantile(0.80)
    candidate = summer & (
        (df["temp_change_24h_c"] >= heat_jump_threshold)
        & (
            (df["cdd_change_24h"] >= cdd_jump_threshold)
            | (df["load_change_24h_mw"] >= load_jump_threshold)
        )
    )
    return contiguous_events(
        df, candidate, "summer_sudden_heat", min_event_hours, max_gap_hours
    )


def detect_cold_wave_events(
    df: pd.DataFrame, min_event_hours: int, max_gap_hours: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    winter = df["month_eastern"].isin([12, 1, 2, 3])
    winter_df = df.loc[winter].copy()
    cold_drop_threshold = min(-5.0, winter_df["temp_change_24h_c"].quantile(0.10))
    hdd_jump_threshold = max(4.0, winter_df["hdd_change_24h"].quantile(0.90))
    load_jump_threshold = winter_df["load_change_24h_mw"].quantile(0.75)
    candidate = winter & (
        (df["temp_change_24h_c"] <= cold_drop_threshold)
        & (
            (df["hdd_change_24h"] >= hdd_jump_threshold)
            | (df["load_change_24h_mw"] >= load_jump_threshold)
        )
    )
    return contiguous_events(
        df, candidate, "winter_cold_wave", min_event_hours, max_gap_hours
    )


def analyze_summer(
    df: pd.DataFrame, min_event_hours: int, max_gap_hours: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summer = df["month_eastern"].isin([6, 7, 8])
    summer_df = df.loc[summer].copy()
    summer_lmp_p90 = summer_df["lmp_clean"].quantile(0.90)
    summer_lmp_p95 = summer_df["lmp_clean"].quantile(0.95)
    summer_cdd_p90 = summer_df["cdd_clean"].quantile(0.90)
    summer_load_p90 = summer_df["load_mw_clean"].quantile(0.90)

    yearly = (
        summer_df.groupby("year_eastern")
        .agg(
            hours=("lmp_clean", "size"),
            mean_lmp=("lmp_clean", "mean"),
            median_lmp=("lmp_clean", "median"),
            p95_lmp=("lmp_clean", lambda s: s.quantile(0.95)),
            max_lmp=("lmp_clean", "max"),
            mean_load_mw=("load_mw_clean", "mean"),
            max_load_mw=("load_mw_clean", "max"),
            mean_temp_c=("temp_c_clean", "mean"),
            max_temp_c=("temp_c_clean", "max"),
            mean_cdd=("cdd_clean", "mean"),
            max_cdd=("cdd_clean", "max"),
            high_price_hours=("lmp_clean", lambda s: int((s >= summer_lmp_p95).sum())),
        )
        .reset_index()
    )

    hourly_profile = (
        summer_df.groupby(["year_eastern", "hour_eastern"])
        .agg(
            mean_lmp=("lmp_clean", "mean"),
            p95_lmp=("lmp_clean", lambda s: s.quantile(0.95)),
            mean_load_mw=("load_mw_clean", "mean"),
            mean_temp_c=("temp_c_clean", "mean"),
            mean_cdd=("cdd_clean", "mean"),
        )
        .reset_index()
    )

    heat_candidate = summer & (
        (
            (df["cdd_clean"] >= summer_cdd_p90)
            | (df["load_mw_clean"] >= summer_load_p90)
        )
        & (df["lmp_clean"] >= summer_lmp_p90)
    )
    heat_events, heat_hours = contiguous_events(
        df, heat_candidate, "summer_heat_price", min_event_hours, max_gap_hours
    )
    return yearly, hourly_profile, heat_events, heat_hours


def major_winter_storm_event_study(
    hourly: pd.DataFrame, winter_events: pd.DataFrame
) -> pd.DataFrame:
    if winter_events.empty:
        return pd.DataFrame()

    major_threshold = winter_events["max_lmp"].quantile(0.90)
    major_events = winter_events.loc[
        winter_events["max_lmp"] >= major_threshold
    ].sort_values("max_lmp", ascending=False)
    rows = []
    for _, event in major_events.iterrows():
        start = pd.Timestamp(event["start_utc"])
        end = pd.Timestamp(event["end_utc"])
        windows = {
            "pre_72h": (start - pd.Timedelta(hours=72), start - pd.Timedelta(hours=1)),
            "event": (start, end),
            "post_72h": (end + pd.Timedelta(hours=1), end + pd.Timedelta(hours=72)),
        }
        for phase, (phase_start, phase_end) in windows.items():
            phase_df = hourly.loc[
                (hourly.index >= phase_start) & (hourly.index <= phase_end)
            ].copy()
            if phase_df.empty:
                continue
            rows.append(
                {
                    "event_id": event["event_id"],
                    "phase": phase,
                    "phase_start_utc": phase_start,
                    "phase_end_utc": phase_end,
                    "hours": len(phase_df),
                    "mean_lmp": phase_df["lmp_clean"].mean(),
                    "median_lmp": phase_df["lmp_clean"].median(),
                    "max_lmp": phase_df["lmp_clean"].max(),
                    "mean_load_mw": phase_df["load_mw_clean"].mean(),
                    "max_load_mw": phase_df["load_mw_clean"].max(),
                    "mean_temp_c": phase_df["temp_c_clean"].mean(),
                    "min_temp_c": phase_df["temp_c_clean"].min(),
                    "mean_hdd": phase_df["hdd_clean"].mean(),
                    "max_hdd": phase_df["hdd_clean"].max(),
                    "max_wind_speed_ms": phase_df["wind_speed_ms_clean"].max(),
                    "precip_1h_total_mm": phase_df["precip_1h_mm_clean"].sum(),
                }
            )

    study = pd.DataFrame(rows)
    if study.empty:
        return study
    event_means = study.pivot(index="event_id", columns="phase", values="mean_lmp")
    if {"event", "pre_72h"}.issubset(event_means.columns):
        uplift = (event_means["event"] - event_means["pre_72h"]).rename(
            "event_mean_lmp_minus_pre_72h"
        )
        study = study.merge(uplift.reset_index(), on="event_id", how="left")
    return study


def regression_row(
    model: str,
    group: pd.DataFrame,
    pred_col: str,
    extra: dict[str, object],
) -> dict[str, object]:
    err = group[pred_col] - group["y_true"]
    sse = float(np.sum(err**2))
    sst = float(np.sum((group["y_true"] - group["y_true"].mean()) ** 2))
    row = {
        "model": model,
        "n": len(group),
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err**2)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - sse / sst) if sst > 0 else np.nan,
        "bias": float(np.mean(err)),
    }
    row.update(extra)
    return row


def prediction_metrics_by_context(
    reports_dir: Path,
    winter_hours: pd.DataFrame,
    summer_heat_hours: pd.DataFrame,
    sudden_heat_hours: pd.DataFrame,
    cold_wave_hours: pd.DataFrame,
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    pred_path = reports_dir / "forecast_predictions_24h.csv"
    if not pred_path.exists():
        return pd.DataFrame()

    preds = pd.read_csv(pred_path)
    preds["target_ts"] = pd.to_datetime(preds["target_ts"], utc=True)
    preds["origin_ts"] = pd.to_datetime(preds["origin_ts"], utc=True)
    preds = preds.set_index("target_ts")

    winter_event_map = pd.Series(dtype=object)
    if not winter_hours.empty:
        winter_tmp = winter_hours.copy()
        winter_tmp["timestamp_utc"] = pd.to_datetime(winter_tmp["timestamp_utc"], utc=True)
        winter_event_map = winter_tmp.drop_duplicates("timestamp_utc").set_index(
            "timestamp_utc"
        )["event_id"]

    heat_event_map = pd.Series(dtype=object)
    if not summer_heat_hours.empty:
        heat_tmp = summer_heat_hours.copy()
        heat_tmp["timestamp_utc"] = pd.to_datetime(heat_tmp["timestamp_utc"], utc=True)
        heat_event_map = heat_tmp.drop_duplicates("timestamp_utc").set_index(
            "timestamp_utc"
        )["event_id"]

    sudden_heat_map = pd.Series(dtype=object)
    if not sudden_heat_hours.empty:
        sudden_tmp = sudden_heat_hours.copy()
        sudden_tmp["timestamp_utc"] = pd.to_datetime(sudden_tmp["timestamp_utc"], utc=True)
        sudden_heat_map = sudden_tmp.drop_duplicates("timestamp_utc").set_index(
            "timestamp_utc"
        )["event_id"]

    cold_wave_map = pd.Series(dtype=object)
    if not cold_wave_hours.empty:
        cold_tmp = cold_wave_hours.copy()
        cold_tmp["timestamp_utc"] = pd.to_datetime(cold_tmp["timestamp_utc"], utc=True)
        cold_wave_map = cold_tmp.drop_duplicates("timestamp_utc").set_index(
            "timestamp_utc"
        )["event_id"]

    local = preds.index.tz_convert(LOCAL_TZ)
    preds["is_summer"] = local.month.isin([6, 7, 8])
    preds["is_winter"] = local.month.isin([12, 1, 2, 3])
    preds["winter_event_id"] = winter_event_map.reindex(preds.index).to_numpy()
    preds["summer_heat_event_id"] = heat_event_map.reindex(preds.index).to_numpy()
    preds["sudden_heat_event_id"] = sudden_heat_map.reindex(preds.index).to_numpy()
    preds["cold_wave_event_id"] = cold_wave_map.reindex(preds.index).to_numpy()
    preds["context"] = "other"
    preds.loc[preds["is_summer"], "context"] = "summer_all"
    preds.loc[preds["is_winter"], "context"] = "winter_all"
    preds.loc[preds["winter_event_id"].notna(), "context"] = "winter_storm"
    preds.loc[preds["cold_wave_event_id"].notna(), "context"] = "winter_cold_wave"
    preds.loc[preds["sudden_heat_event_id"].notna(), "context"] = "summer_sudden_heat"
    preds.loc[preds["summer_heat_event_id"].notna(), "context"] = "summer_heat_price_event"

    rows = []
    model_cols = [col for col in MODEL_COLS if col in preds.columns]
    for model_col in model_cols:
        tmp = preds.dropna(subset=["y_true", model_col])
        for (split, context), group in tmp.groupby(["split", "context"]):
            rows.append(
                regression_row(
                    model_col,
                    group,
                    model_col,
                    {"split": split, "context": context},
                )
            )
    return pd.DataFrame(rows)


def seasonal_year_prediction_metrics(
    reports_dir: Path, hourly: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_path = reports_dir / "forecast_predictions_24h.csv"
    if not pred_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    preds = pd.read_csv(pred_path)
    preds["target_ts"] = pd.to_datetime(preds["target_ts"], utc=True)
    local = preds["target_ts"].dt.tz_convert(LOCAL_TZ)
    preds["season"] = np.select(
        [local.dt.month.isin([6, 7, 8]), local.dt.month.isin([12, 1, 2, 3])],
        ["summer", "winter"],
        default="other",
    )
    preds["season_year"] = np.where(
        (preds["season"] == "winter") & (local.dt.month == 12),
        local.dt.year + 1,
        local.dt.year,
    )
    preds = preds.loc[preds["season"].isin(["summer", "winter"])].copy()

    train_threshold = float(
        hourly.loc[
            hourly.index <= pd.Timestamp("2022-12-31 23:00:00+00:00"), "lmp_clean"
        ].quantile(0.90)
    )

    metric_rows = []
    auc_rows = []
    curve_rows = []
    model_cols = [col for col in MODEL_COLS if col in preds.columns]
    for model_col in model_cols:
        tmp = preds.dropna(subset=["y_true", model_col])
        for (split, season, season_year), group in tmp.groupby(
            ["split", "season", "season_year"]
        ):
            metric_rows.append(
                regression_row(
                    model_col,
                    group,
                    model_col,
                    {
                        "split": split,
                        "season": season,
                        "season_year": int(season_year),
                    },
                )
            )
            y_binary = (group["y_true"] >= train_threshold).astype(int)
            positives = int(y_binary.sum())
            negatives = int(len(y_binary) - positives)
            auc_value = np.nan
            if positives > 0 and negatives > 0:
                auc_value = float(roc_auc_score(y_binary, group[model_col]))
                fpr, tpr, thresholds = roc_curve(y_binary, group[model_col])
                for point, (fpr_value, tpr_value, threshold_value) in enumerate(
                    zip(fpr, tpr, thresholds)
                ):
                    curve_rows.append(
                        {
                            "model": model_col,
                            "split": split,
                            "season": season,
                            "season_year": int(season_year),
                            "point": point,
                            "fpr": float(fpr_value),
                            "tpr": float(tpr_value),
                            "threshold": float(threshold_value),
                        }
                    )
            auc_rows.append(
                {
                    "model": model_col,
                    "split": split,
                    "season": season,
                    "season_year": int(season_year),
                    "threshold_lmp": train_threshold,
                    "n": len(group),
                    "positives": positives,
                    "negatives": negatives,
                    "auc": auc_value,
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(auc_rows), pd.DataFrame(curve_rows)


def write_seasonal_roc_plots(
    reports_dir: Path,
    seasonal_curve_points: pd.DataFrame,
    model: str = "dmdc_residual_24h",
) -> None:
    if seasonal_curve_points.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    for season in ["summer", "winter"]:
        plot_df = seasonal_curve_points[
            (seasonal_curve_points["season"] == season)
            & (seasonal_curve_points["model"] == model)
        ].copy()
        if plot_df.empty:
            continue
        plt.figure(figsize=(8, 6))
        for (split, season_year), group in plot_df.groupby(["split", "season_year"]):
            label = f"{split} {int(season_year)}"
            plt.plot(group["fpr"], group["tpr"], linewidth=1.5, label=label)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{season.title()} high-price ROC by year ({model})")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(reports_dir / f"roc_curve_{season}_by_year.png", dpi=160)
        plt.close()


def write_pattern_summary(
    reports_dir: Path,
    winter_events: pd.DataFrame,
    summer_yearly: pd.DataFrame,
    winter_yearly: pd.DataFrame,
    sudden_heat_events: pd.DataFrame,
    cold_wave_events: pd.DataFrame,
    major_storm_study: pd.DataFrame,
    event_metrics: pd.DataFrame,
) -> None:
    lines = ["# Winter Storm and Summer Price Pattern Summary", ""]
    lines.append("## Winter storm events")
    lines.append(f"- Detected events: {len(winter_events)}")
    if not winter_events.empty:
        top = winter_events.sort_values("max_lmp", ascending=False).head(10)
        lines.append("- Top winter storm events by max LMP:")
        for _, row in top.iterrows():
            lines.append(
                f"  - {row['event_id']}: {row['start_eastern']} to {row['end_eastern']}, "
                f"max_lmp={row['max_lmp']:.2f}, min_temp_c={row['min_temp_c']:.2f}, "
                f"max_wind_ms={row['max_wind_speed_ms']:.2f}"
            )

    lines.extend(["", "## Summer yearly pattern"])
    if not summer_yearly.empty:
        for _, row in summer_yearly.iterrows():
            lines.append(
                f"- {int(row['year_eastern'])}: mean_lmp={row['mean_lmp']:.2f}, "
                f"p95_lmp={row['p95_lmp']:.2f}, max_lmp={row['max_lmp']:.2f}, "
                f"max_temp_c={row['max_temp_c']:.2f}, high_price_hours={int(row['high_price_hours'])}"
            )

    lines.extend(["", "## Winter yearly pattern"])
    if not winter_yearly.empty:
        for _, row in winter_yearly.iterrows():
            lines.append(
                f"- winter {int(row['winter_year_eastern'])}: mean_lmp={row['mean_lmp']:.2f}, "
                f"p95_lmp={row['p95_lmp']:.2f}, max_lmp={row['max_lmp']:.2f}, "
                f"min_temp_c={row['min_temp_c']:.2f}, high_price_hours={int(row['high_price_hours'])}"
            )

    lines.extend(["", "## Sudden weather events"])
    lines.append(f"- Summer sudden-heat events: {len(sudden_heat_events)}")
    lines.append(f"- Winter cold-wave events: {len(cold_wave_events)}")
    if not major_storm_study.empty:
        uplift = (
            major_storm_study[["event_id", "event_mean_lmp_minus_pre_72h"]]
            .dropna()
            .drop_duplicates()
            .sort_values("event_mean_lmp_minus_pre_72h", ascending=False)
            .head(5)
        )
        lines.append("- Largest major-storm mean LMP uplifts vs pre-72h:")
        for _, row in uplift.iterrows():
            lines.append(
                f"  - {row['event_id']}: uplift={row['event_mean_lmp_minus_pre_72h']:.2f}"
            )

    lines.extend(["", "## Forecast metrics by context"])
    if event_metrics.empty:
        lines.append("- Forecast prediction file was not available, so context metrics were skipped.")
    else:
        best = event_metrics.sort_values(["split", "context", "mae"]).groupby(
            ["split", "context"]
        ).head(1)
        for _, row in best.iterrows():
            lines.append(
                f"- {row['split']} / {row['context']}: best_mae_model={row['model']}, "
                f"mae={row['mae']:.2f}, mse={row['mse']:.2f}, "
                f"rmse={row['rmse']:.2f}, r2={row['r2']:.3f}, n={int(row['n'])}"
            )

    (reports_dir / "pattern_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    hourly = load_hourly(args.processed_dir)
    winter_events, winter_hours = detect_winter_storms(
        hourly, args.min_event_hours, args.max_gap_hours
    )
    winter_yearly = analyze_winter(hourly)
    summer_yearly, summer_hourly, heat_events, heat_hours = analyze_summer(
        hourly, args.min_event_hours, args.max_gap_hours
    )
    sudden_heat_events, sudden_heat_hours = detect_sudden_heat_events(
        hourly, args.min_event_hours, args.max_gap_hours
    )
    cold_wave_events, cold_wave_hours = detect_cold_wave_events(
        hourly, args.min_event_hours, args.max_gap_hours
    )
    major_storm_study = major_winter_storm_event_study(hourly, winter_events)
    event_metrics = prediction_metrics_by_context(
        args.reports_dir,
        winter_hours,
        heat_hours,
        sudden_heat_hours,
        cold_wave_hours,
        hourly,
    )
    seasonal_metrics, seasonal_auc, seasonal_curve_points = seasonal_year_prediction_metrics(
        args.reports_dir, hourly
    )

    winter_events.to_csv(args.reports_dir / "winter_storm_events.csv", index=False)
    winter_hours.to_csv(args.reports_dir / "winter_storm_hours.csv", index=False)
    winter_yearly.to_csv(args.reports_dir / "winter_yearly_summary.csv", index=False)
    summer_yearly.to_csv(args.reports_dir / "summer_yearly_summary.csv", index=False)
    summer_hourly.to_csv(args.reports_dir / "summer_hourly_profile_by_year.csv", index=False)
    heat_events.to_csv(args.reports_dir / "summer_heat_price_events.csv", index=False)
    heat_hours.to_csv(args.reports_dir / "summer_heat_price_hours.csv", index=False)
    sudden_heat_events.to_csv(args.reports_dir / "summer_sudden_heat_events.csv", index=False)
    sudden_heat_hours.to_csv(args.reports_dir / "summer_sudden_heat_hours.csv", index=False)
    cold_wave_events.to_csv(args.reports_dir / "winter_cold_wave_events.csv", index=False)
    cold_wave_hours.to_csv(args.reports_dir / "winter_cold_wave_hours.csv", index=False)
    major_storm_study.to_csv(args.reports_dir / "major_winter_storm_event_study.csv", index=False)
    event_metrics.to_csv(args.reports_dir / "event_forecast_metrics.csv", index=False)
    seasonal_metrics.to_csv(args.reports_dir / "seasonal_year_forecast_metrics.csv", index=False)
    seasonal_auc.to_csv(args.reports_dir / "seasonal_year_roc_auc.csv", index=False)
    seasonal_curve_points.to_csv(
        args.reports_dir / "seasonal_year_roc_curve_points.csv", index=False
    )
    write_seasonal_roc_plots(args.reports_dir, seasonal_curve_points)
    write_pattern_summary(
        args.reports_dir,
        winter_events,
        summer_yearly,
        winter_yearly,
        sudden_heat_events,
        cold_wave_events,
        major_storm_study,
        event_metrics,
    )

    print("Finished winter/summer pattern analysis")
    print(f"Winter storm events: {len(winter_events)}")
    print(f"Summer heat-price events: {len(heat_events)}")
    print(f"Summer sudden-heat events: {len(sudden_heat_events)}")
    print(f"Winter cold-wave events: {len(cold_wave_events)}")
    if not event_metrics.empty:
        print(event_metrics.sort_values(['split', 'context', 'mae']).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
