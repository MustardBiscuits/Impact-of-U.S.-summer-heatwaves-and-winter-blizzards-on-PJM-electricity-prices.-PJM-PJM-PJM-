from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "seasonal_naive_24h",
    "seasonal_naive_168h",
    "dmdc",
    "dmdc_residual_24h",
]
MODEL_LABELS = {
    "seasonal_naive_24h": "Seasonal naive 24h",
    "seasonal_naive_168h": "Seasonal naive 168h",
    "dmdc": "DMDc",
    "dmdc_residual_24h": "Seasonal + DMDc residual",
}
SPLIT_COLORS = {
    "train": "#2A6F97",
    "validation": "#F4A261",
    "test": "#2A9D8F",
}
MODEL_COLORS = {
    "seasonal_naive_24h": "#356D8A",
    "seasonal_naive_168h": "#8A8F38",
    "dmdc": "#C65D3A",
    "dmdc_residual_24h": "#2A9D8F",
}


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    work_dir = script_path.parents[1]
    parser = argparse.ArgumentParser(description="Generate report figures.")
    parser.add_argument("--processed-dir", type=Path, default=work_dir / "data" / "processed")
    parser.add_argument("--reports-dir", type=Path, default=work_dir / "reports")
    parser.add_argument("--focus-model", default="dmdc_residual_24h")
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
        }
    )


def load_data(processed_dir: Path, reports_dir: Path) -> dict[str, pd.DataFrame]:
    data = {
        "hourly": pd.read_parquet(processed_dir / "pjm_weather_hourly_clean.parquet"),
        "supervised": pd.read_parquet(processed_dir / "pjm_weather_supervised_24h.parquet"),
        "metrics_summary": pd.read_csv(reports_dir / "forecast_metrics_summary.csv"),
        "metrics_horizon": pd.read_csv(reports_dir / "forecast_metrics_by_horizon.csv"),
        "predictions": pd.read_csv(reports_dir / "forecast_predictions_24h.csv"),
        "roc_points": pd.read_csv(reports_dir / "roc_curve_points.csv"),
        "event_metrics": pd.read_csv(reports_dir / "event_forecast_metrics.csv"),
    }
    for key in ["hourly", "supervised"]:
        data[key]["timestamp_utc"] = pd.to_datetime(data[key]["timestamp_utc"], utc=True)
    data["predictions"]["origin_ts"] = pd.to_datetime(data["predictions"]["origin_ts"], utc=True)
    data["predictions"]["target_ts"] = pd.to_datetime(data["predictions"]["target_ts"], utc=True)
    return data


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_split(data: dict[str, pd.DataFrame], reports_dir: Path) -> None:
    supervised = data["supervised"].copy()
    split_summary = (
        supervised.groupby("split")
        .agg(
            start=("timestamp_utc", "min"),
            end=("timestamp_utc", "max"),
            rows=("timestamp_utc", "size"),
        )
        .reindex(["train", "validation", "test"])
        .dropna()
        .reset_index()
    )
    split_summary["start_num"] = mdates.date2num(split_summary["start"].dt.tz_convert(None))
    split_summary["end_num"] = mdates.date2num(split_summary["end"].dt.tz_convert(None))
    split_summary["width"] = split_summary["end_num"] - split_summary["start_num"]

    fig, ax = plt.subplots(figsize=(11, 3.8))
    y_positions = np.arange(len(split_summary))
    for y, row in zip(y_positions, split_summary.itertuples(index=False)):
        ax.barh(
            y,
            row.width,
            left=row.start_num,
            height=0.48,
            color=SPLIT_COLORS.get(row.split, "#777777"),
        )
        ax.text(
            row.start_num + row.width / 2,
            y,
            f"{row.split}: {int(row.rows):,} rows",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(split_summary["split"])
    ax.invert_yaxis()
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("Train-validation-test time split")
    ax.set_xlabel("Forecast origin timestamp")
    save_fig(fig, reports_dir / "fig_train_validation_test_split.png")


def plot_lmp_load_temp(data: dict[str, pd.DataFrame], reports_dir: Path) -> None:
    hourly = data["hourly"].copy()
    hourly = hourly.set_index("timestamp_utc").sort_index()
    daily = hourly[["lmp_clean", "load_mw_clean", "temp_c_clean"]].resample("D").mean()
    roll = daily.rolling(7, min_periods=1).mean()

    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    specs = [
        ("lmp_clean", "LMP ($/MWh)", "#264653"),
        ("load_mw_clean", "Load (MW)", "#2A9D8F"),
        ("temp_c_clean", "Temperature (C)", "#E76F51"),
    ]
    for ax, (col, label, color) in zip(axes, specs):
        ax.plot(daily.index, daily[col], color=color, alpha=0.18, linewidth=0.7)
        ax.plot(roll.index, roll[col], color=color, linewidth=1.6, label="7-day mean")
        ax.set_ylabel(label)
        ax.legend(loc="upper left")
    axes[0].set_title("PJM DOM LMP, load, and temperature time series")
    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save_fig(fig, reports_dir / "fig_lmp_load_temperature_timeseries.png")


def plot_overall_model_comparison(data: dict[str, pd.DataFrame], reports_dir: Path) -> None:
    metrics = data["metrics_summary"].copy()
    metrics["model"] = pd.Categorical(metrics["model"], categories=MODEL_ORDER, ordered=True)
    metrics = metrics.sort_values(["split", "model"])
    metric_specs = [("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R2")]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (metric, label) in zip(axes, metric_specs):
        pivot = metrics.pivot(index="model", columns="split", values=metric).reindex(MODEL_ORDER)
        x = np.arange(len(pivot.index))
        width = 0.36
        splits = [s for s in ["validation", "test"] if s in pivot.columns]
        offsets = np.linspace(-width / 2, width / 2, len(splits)) if len(splits) > 1 else [0]
        for split, offset in zip(splits, offsets):
            ax.bar(
                x + offset,
                pivot[split],
                width=width,
                label=split,
                color=SPLIT_COLORS.get(split, "#777777"),
                alpha=0.9,
            )
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in pivot.index], rotation=35, ha="right")
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.legend()
    fig.suptitle("Overall model comparison: MAE, RMSE, and R2", y=1.03)
    save_fig(fig, reports_dir / "fig_overall_model_comparison_mae_rmse_r2.png")


def plot_horizon_error(data: dict[str, pd.DataFrame], reports_dir: Path) -> None:
    horizon = data["metrics_horizon"].copy()
    horizon = horizon[horizon["split"] == "test"].copy()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for model in MODEL_ORDER:
        model_df = horizon[horizon["model"] == model].sort_values("horizon")
        if model_df.empty:
            continue
        label = MODEL_LABELS.get(model, model)
        color = MODEL_COLORS.get(model)
        axes[0].plot(model_df["horizon"], model_df["mae"], marker="o", linewidth=1.6, label=label, color=color)
        axes[1].plot(model_df["horizon"], model_df["rmse"], marker="o", linewidth=1.6, label=label, color=color)
    axes[0].set_ylabel("MAE")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xlabel("Forecast horizon (hours ahead)")
    axes[1].set_xticks(range(1, 25))
    axes[0].set_title("Error by forecast horizon 1-24h (test split)")
    axes[0].legend(ncol=2)
    save_fig(fig, reports_dir / "fig_error_by_forecast_horizon_1_24h.png")


def plot_actual_vs_predicted(
    data: dict[str, pd.DataFrame], reports_dir: Path, focus_model: str
) -> None:
    preds = data["predictions"].copy()
    preds = preds[(preds["split"] == "test") & preds[focus_model].notna()].sort_values("target_ts")
    preds["target_day"] = preds["target_ts"].dt.floor("D")
    daily = preds.groupby("target_day")[["y_true", focus_model]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(daily["target_day"], daily["y_true"], color="#222222", linewidth=1.8, label="Actual LMP")
    ax.plot(
        daily["target_day"],
        daily[focus_model],
        color=MODEL_COLORS.get(focus_model, "#2A9D8F"),
        linewidth=1.6,
        label=f"Predicted: {MODEL_LABELS.get(focus_model, focus_model)}",
    )
    ax.set_title("Actual vs predicted LMP (test split, daily mean)")
    ax.set_ylabel("LMP ($/MWh)")
    ax.set_xlabel("Target date")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend()
    save_fig(fig, reports_dir / "fig_actual_vs_predicted_lmp_test_daily.png")

    high_start = pd.Timestamp("2025-06-01", tz="UTC")
    high_end = pd.Timestamp("2025-08-26", tz="UTC")
    window = preds[(preds["target_ts"] >= high_start) & (preds["target_ts"] <= high_end)].copy()
    if not window.empty:
        fig, ax = plt.subplots(figsize=(12, 4.8))
        ax.plot(window["target_ts"], window["y_true"], color="#222222", linewidth=1.0, label="Actual LMP")
        ax.plot(
            window["target_ts"],
            window[focus_model],
            color=MODEL_COLORS.get(focus_model, "#2A9D8F"),
            linewidth=1.0,
            alpha=0.9,
            label=f"Predicted: {MODEL_LABELS.get(focus_model, focus_model)}",
        )
        ax.set_title("Actual vs predicted LMP during summer 2025 (hourly)")
        ax.set_ylabel("LMP ($/MWh)")
        ax.set_xlabel("Target timestamp")
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.legend()
        save_fig(fig, reports_dir / "fig_actual_vs_predicted_lmp_summer_2025_hourly.png")


def plot_roc(data: dict[str, pd.DataFrame], reports_dir: Path) -> None:
    roc = data["roc_points"].copy()
    roc = roc[(roc["split"] == "test") & (roc["context"] == "all")]
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    for model in MODEL_ORDER:
        model_df = roc[roc["model"] == model]
        if model_df.empty:
            continue
        ax.plot(
            model_df["fpr"],
            model_df["tpr"],
            linewidth=1.8,
            color=MODEL_COLORS.get(model),
            label=MODEL_LABELS.get(model, model),
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1)
    ax.set_title("ROC curve for high-price detection (test split)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    save_fig(fig, reports_dir / "fig_roc_curve_high_price_detection.png")


def plot_event_metrics(data: dict[str, pd.DataFrame], reports_dir: Path) -> None:
    event_metrics = data["event_metrics"].copy()
    contexts = [
        "summer_heat_price_event",
        "summer_sudden_heat",
        "winter_cold_wave",
        "winter_storm",
    ]
    event_metrics = event_metrics[
        (event_metrics["split"] == "test")
        & event_metrics["context"].isin(contexts)
        & event_metrics["model"].isin(MODEL_ORDER)
    ].copy()
    if event_metrics.empty:
        return

    context_labels = {
        "summer_heat_price_event": "Summer heat-price",
        "summer_sudden_heat": "Sudden summer heat",
        "winter_cold_wave": "Winter cold wave",
        "winter_storm": "Winter storm",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharex=True)
    for ax, metric, ylabel in [(axes[0], "mae", "MAE"), (axes[1], "rmse", "RMSE")]:
        pivot = (
            event_metrics.pivot(index="context", columns="model", values=metric)
            .reindex(contexts)
            .reindex(columns=MODEL_ORDER)
        )
        x = np.arange(len(pivot.index))
        width = 0.18
        offsets = np.linspace(-1.5 * width, 1.5 * width, len(MODEL_ORDER))
        for model, offset in zip(MODEL_ORDER, offsets):
            if model not in pivot.columns:
                continue
            ax.bar(
                x + offset,
                pivot[model],
                width=width,
                color=MODEL_COLORS.get(model),
                label=MODEL_LABELS.get(model, model),
            )
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([context_labels.get(c, c) for c in pivot.index], rotation=25, ha="right")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Event-based MAE/RMSE comparison (test split)", y=1.03)
    save_fig(fig, reports_dir / "fig_event_based_mae_rmse_comparison.png")


def main() -> None:
    args = parse_args()
    set_style()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.processed_dir, args.reports_dir)

    plot_split(data, args.reports_dir)
    plot_lmp_load_temp(data, args.reports_dir)
    plot_overall_model_comparison(data, args.reports_dir)
    plot_horizon_error(data, args.reports_dir)
    plot_actual_vs_predicted(data, args.reports_dir, args.focus_model)
    plot_roc(data, args.reports_dir)
    plot_event_metrics(data, args.reports_dir)

    print("Generated report figures in", args.reports_dir)


if __name__ == "__main__":
    main()
