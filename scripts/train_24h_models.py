from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


UTC = "UTC"
HORIZON = 24


STATE_COLS = [
    "lmp_clean",
    "load_mw_clean",
    "temp_c_clean",
    "dewpoint_c_clean",
    "wind_speed_ms_clean",
    "cdd_clean",
    "hdd_clean",
    "precip_1h_mm_clean",
]

RESID_STATE_COLS = [
    "lmp_resid_24h",
    "load_resid_24h",
    "temp_resid_24h",
    "hdd_resid_24h",
    "cdd_resid_24h",
]

CONTROL_COLS = [
    "sin_hour",
    "cos_hour",
    "sin_dow",
    "cos_dow",
    "sin_month",
    "cos_month",
    "is_weekend",
    "is_summer",
    "is_winter",
]

SARIMAX_EXOG_COLS = CONTROL_COLS + [
    "temp_c_clean",
    "hdd_clean",
    "cdd_clean",
    "load_mw_clean",
]


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    work_dir = script_path.parents[1]
    parser = argparse.ArgumentParser(
        description="Train/evaluate 24h PJM price baselines and DMDc."
    )
    parser.add_argument("--processed-dir", type=Path, default=work_dir / "data" / "processed")
    parser.add_argument("--reports-dir", type=Path, default=work_dir / "reports")
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--origin-step-hours", type=int, default=24)
    parser.add_argument("--run-sarimax", action="store_true")
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--cv-origin-step-hours", type=int, default=24)
    parser.add_argument("--high-price-quantile", type=float, default=0.90)
    parser.add_argument("--sarimax-train-tail-hours", type=int, default=12000)
    parser.add_argument("--sarimax-origin-step-hours", type=int, default=24)
    parser.add_argument("--max-sarimax-origins", type=int, default=90)
    return parser.parse_args()


@dataclass
class Standardizer:
    mean: pd.Series
    std: pd.Series

    @classmethod
    def fit(cls, df: pd.DataFrame, cols: list[str]) -> "Standardizer":
        mean = df[cols].mean()
        std = df[cols].std(ddof=0).replace(0, 1.0).fillna(1.0)
        return cls(mean=mean, std=std)

    def transform(self, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        return ((df[cols] - self.mean[cols]) / self.std[cols]).to_numpy(dtype=float)

    def transform_row(self, row: pd.Series, cols: list[str]) -> np.ndarray:
        return ((row[cols] - self.mean[cols]) / self.std[cols]).to_numpy(dtype=float)

    def inverse_state(self, arr: np.ndarray, cols: list[str]) -> np.ndarray:
        return arr * self.std[cols].to_numpy(dtype=float) + self.mean[cols].to_numpy(dtype=float)


@dataclass
class DMDcModel:
    a_matrix: np.ndarray
    b_matrix: np.ndarray
    state_scaler: Standardizer
    control_scaler: Standardizer
    state_cols: list[str]
    control_cols: list[str]

    @classmethod
    def fit(
        cls,
        hourly_train: pd.DataFrame,
        state_cols: list[str],
        control_cols: list[str],
        ridge: float = 1e-6,
    ) -> "DMDcModel":
        fit_df = hourly_train[state_cols + control_cols].dropna().copy()
        state_scaler = Standardizer.fit(fit_df, state_cols)
        control_scaler = Standardizer.fit(fit_df, control_cols)
        x = state_scaler.transform(fit_df, state_cols)
        u = control_scaler.transform(fit_df, control_cols)

        x0 = x[:-1].T
        x1 = x[1:].T
        u0 = u[:-1].T
        omega = np.vstack([x0, u0])

        # Ridge-stabilized least squares for X_next = [A B] [X; U].
        lhs = x1 @ omega.T
        rhs = omega @ omega.T + ridge * np.eye(omega.shape[0])
        ab = lhs @ np.linalg.pinv(rhs)
        n_state = len(state_cols)
        return cls(
            a_matrix=ab[:, :n_state],
            b_matrix=ab[:, n_state:],
            state_scaler=state_scaler,
            control_scaler=control_scaler,
            state_cols=state_cols,
            control_cols=control_cols,
        )

    def forecast_one_origin(
        self,
        origin_row: pd.Series,
        future_controls: pd.DataFrame,
        horizon: int,
    ) -> list[float]:
        x = self.state_scaler.transform_row(origin_row, self.state_cols)
        preds: list[float] = []
        for step in range(horizon):
            u = self.control_scaler.transform_row(
                future_controls.iloc[step], self.control_cols
            )
            x = self.a_matrix @ x + self.b_matrix @ u
            x_unscaled = self.state_scaler.inverse_state(x, self.state_cols)
            preds.append(float(x_unscaled[0]))
        return preds


def load_data(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly = pd.read_parquet(processed_dir / "pjm_weather_hourly_clean.parquet")
    supervised = pd.read_parquet(processed_dir / "pjm_weather_supervised_24h.parquet")
    hourly["timestamp_utc"] = pd.to_datetime(hourly["timestamp_utc"], utc=True)
    supervised["timestamp_utc"] = pd.to_datetime(supervised["timestamp_utc"], utc=True)
    hourly = hourly.set_index("timestamp_utc").sort_index()
    supervised = supervised.set_index("timestamp_utc").sort_index()

    residual_map = {
        "lmp_resid_24h": "lmp_clean",
        "load_resid_24h": "load_mw_clean",
        "temp_resid_24h": "temp_c_clean",
        "hdd_resid_24h": "hdd_clean",
        "cdd_resid_24h": "cdd_clean",
    }
    for resid_col, source_col in residual_map.items():
        hourly[resid_col] = hourly[source_col] - hourly[source_col].shift(24)
        supervised[resid_col] = hourly[resid_col].reindex(supervised.index)
    return hourly, supervised


def select_origins(supervised: pd.DataFrame, step_hours: int) -> pd.DataFrame:
    eval_df = supervised.loc[supervised["split"].isin(["validation", "test"])].copy()
    return downsample_origins(eval_df, step_hours)


def downsample_origins(origins: pd.DataFrame, step_hours: int) -> pd.DataFrame:
    if step_hours <= 1:
        return origins
    origin_hours = ((origins.index - origins.index.min()) / pd.Timedelta(hours=1)).astype(int)
    return origins.loc[origin_hours % step_hours == 0]


def make_predictions(
    hourly: pd.DataFrame,
    origins: pd.DataFrame,
    dmdc: DMDcModel,
    residual_dmdc: DMDcModel,
    horizon: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    lmp_map = hourly["lmp_clean"]

    for origin_ts, origin_row in origins.iterrows():
        future_times = pd.date_range(
            origin_ts + pd.Timedelta(hours=1),
            periods=horizon,
            freq="h",
            tz=UTC,
        )
        if not future_times.isin(hourly.index).all():
            continue

        future_controls = hourly.loc[future_times, dmdc.control_cols]
        dmdc_preds = dmdc.forecast_one_origin(origin_row, future_controls, horizon)
        residual_preds = residual_dmdc.forecast_one_origin(
            origin_row, future_controls, horizon
        )

        for h, target_ts in enumerate(future_times, start=1):
            y_true = origin_row[f"target_lmp_h{h:02d}"]
            daily_ref = target_ts - pd.Timedelta(hours=24)
            weekly_ref = target_ts - pd.Timedelta(hours=168)
            seasonal_24h = lmp_map.get(daily_ref, np.nan)
            dmdc_resid_24h = (
                seasonal_24h + residual_preds[h - 1]
                if pd.notna(seasonal_24h)
                else np.nan
            )
            rows.append(
                {
                    "origin_ts": origin_ts,
                    "target_ts": target_ts,
                    "split": origin_row["split"],
                    "horizon": h,
                    "y_true": y_true,
                    "seasonal_naive_24h": seasonal_24h,
                    "seasonal_naive_168h": lmp_map.get(weekly_ref, np.nan),
                    "dmdc": dmdc_preds[h - 1],
                    "dmdc_residual_24h": dmdc_resid_24h,
                }
            )

    return pd.DataFrame(rows)


def add_sarimax_predictions(
    predictions: pd.DataFrame,
    hourly: pd.DataFrame,
    origins: pd.DataFrame,
    horizon: int,
    train_tail_hours: int,
    sarimax_origin_step_hours: int,
    max_sarimax_origins: int,
) -> pd.DataFrame:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception as exc:  # pragma: no cover
        print(f"SARIMAX skipped; statsmodels import failed: {exc}")
        predictions["sarimax"] = np.nan
        return predictions

    train_hourly = hourly.loc[hourly.index <= pd.Timestamp("2022-12-31 23:00:00+00:00")]
    train_hourly = train_hourly.tail(train_tail_hours)
    model = SARIMAX(
        train_hourly["lmp_clean"],
        exog=train_hourly[SARIMAX_EXOG_COLS],
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 24),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=75)

    sarimax_origin_df = select_origins(origins, sarimax_origin_step_hours).head(max_sarimax_origins)
    sarimax_rows: list[dict[str, object]] = []

    for origin_ts, _origin_row in sarimax_origin_df.iterrows():
        history = hourly.loc[:origin_ts].tail(train_tail_hours)
        future_times = pd.date_range(
            origin_ts + pd.Timedelta(hours=1), periods=horizon, freq="h", tz=UTC
        )
        if not future_times.isin(hourly.index).all():
            continue

        origin_model = SARIMAX(
            history["lmp_clean"],
            exog=history[SARIMAX_EXOG_COLS],
            order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 24),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        filtered = origin_model.filter(fitted.params)
        fcst = filtered.forecast(steps=horizon, exog=hourly.loc[future_times, SARIMAX_EXOG_COLS])
        for h, (target_ts, pred) in enumerate(zip(future_times, fcst), start=1):
            sarimax_rows.append(
                {
                    "origin_ts": origin_ts,
                    "target_ts": target_ts,
                    "horizon": h,
                    "sarimax": float(pred),
                }
            )

    sarimax_df = pd.DataFrame(sarimax_rows)
    if sarimax_df.empty:
        predictions["sarimax"] = np.nan
        return predictions

    merged = predictions.merge(
        sarimax_df, on=["origin_ts", "target_ts", "horizon"], how="left"
    )
    return merged


def regression_metric_row(
    model: str,
    group: pd.DataFrame,
    pred_col: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    err = group[pred_col] - group["y_true"]
    sse = float(np.sum(err**2))
    sst = float(np.sum((group["y_true"] - group["y_true"].mean()) ** 2))
    row: dict[str, object] = {
        "model": model,
        "n": len(group),
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err**2)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - sse / sst) if sst > 0 else np.nan,
        "bias": float(np.mean(err)),
    }
    if extra:
        row.update(extra)
    return row


def metric_table(predictions: pd.DataFrame, model_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for model_col in model_cols:
        if model_col not in predictions.columns:
            continue
        model_data = predictions.dropna(subset=["y_true", model_col]).copy()
        if model_data.empty:
            continue
        for (split, horizon), group in model_data.groupby(["split", "horizon"]):
            rows.append(
                regression_metric_row(
                    model_col,
                    group,
                    model_col,
                    {"split": split, "horizon": horizon},
                )
            )
    by_horizon = pd.DataFrame(rows)

    summary_rows = []
    for model_col in model_cols:
        if model_col not in predictions.columns:
            continue
        model_data = predictions.dropna(subset=["y_true", model_col]).copy()
        if model_data.empty:
            continue
        for split, group in model_data.groupby("split"):
            summary_rows.append(
                regression_metric_row(
                    model_col,
                    group,
                    model_col,
                    {"split": split},
                )
            )
    summary = pd.DataFrame(summary_rows)
    return by_horizon, summary


def roc_auc_tables(
    predictions: pd.DataFrame,
    model_cols: list[str],
    high_price_threshold: float,
    context_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    point_rows = []
    grouped = (
        predictions.groupby(["split", context_col])
        if context_col and context_col in predictions.columns
        else predictions.groupby(["split"])
    )

    for group_key, split_group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        split_name = group_key[0]
        context_name = group_key[1] if len(group_key) > 1 else "all"
        y_binary = (split_group["y_true"] >= high_price_threshold).astype(int)
        positives = int(y_binary.sum())
        negatives = int(len(y_binary) - positives)
        if positives == 0 or negatives == 0:
            continue

        for model_col in model_cols:
            if model_col not in split_group.columns:
                continue
            model_data = split_group.dropna(subset=["y_true", model_col]).copy()
            if model_data.empty:
                continue
            y_binary = (model_data["y_true"] >= high_price_threshold).astype(int)
            positives = int(y_binary.sum())
            negatives = int(len(y_binary) - positives)
            if positives == 0 or negatives == 0:
                continue
            fpr, tpr, thresholds = roc_curve(y_binary, model_data[model_col])
            auc_value = float(auc(fpr, tpr))
            rows.append(
                {
                    "model": model_col,
                    "split": split_name,
                    "context": context_name,
                    "threshold_lmp": high_price_threshold,
                    "n": len(model_data),
                    "positives": positives,
                    "negatives": negatives,
                    "auc": auc_value,
                }
            )
            for idx, (fpr_value, tpr_value, threshold_value) in enumerate(
                zip(fpr, tpr, thresholds)
            ):
                point_rows.append(
                    {
                        "model": model_col,
                        "split": split_name,
                        "context": context_name,
                        "point": idx,
                        "fpr": float(fpr_value),
                        "tpr": float(tpr_value),
                        "threshold": float(threshold_value),
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(point_rows)


def write_roc_plot(roc_points: pd.DataFrame, output_path: Path, split: str = "test") -> None:
    if roc_points.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plot_df = roc_points[
        (roc_points["split"] == split) & (roc_points["context"] == "all")
    ].copy()
    if plot_df.empty:
        return

    plt.figure(figsize=(7, 6))
    for model, group in plot_df.groupby("model"):
        plt.plot(group["fpr"], group["tpr"], label=model)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"High-price ROC curve ({split})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def time_series_cv_predictions(
    hourly: pd.DataFrame,
    supervised: pd.DataFrame,
    horizon: int,
    origin_step_hours: int,
) -> pd.DataFrame:
    folds = [
        {
            "fold": "cv_2020",
            "train_end": pd.Timestamp("2019-12-31 23:00:00+00:00"),
            "validation_start": pd.Timestamp("2020-01-01 00:00:00+00:00"),
            "validation_end": pd.Timestamp("2020-12-31 23:00:00+00:00"),
        },
        {
            "fold": "cv_2021",
            "train_end": pd.Timestamp("2020-12-31 23:00:00+00:00"),
            "validation_start": pd.Timestamp("2021-01-01 00:00:00+00:00"),
            "validation_end": pd.Timestamp("2021-12-31 23:00:00+00:00"),
        },
        {
            "fold": "cv_2022",
            "train_end": pd.Timestamp("2021-12-31 23:00:00+00:00"),
            "validation_start": pd.Timestamp("2022-01-01 00:00:00+00:00"),
            "validation_end": pd.Timestamp("2022-12-31 23:00:00+00:00"),
        },
    ]
    fold_predictions = []
    for fold in folds:
        train_hourly = hourly.loc[hourly.index <= fold["train_end"]]
        fold_origins = supervised.loc[
            (supervised.index >= fold["validation_start"])
            & (supervised.index <= fold["validation_end"])
        ].copy()
        if train_hourly.empty or fold_origins.empty:
            continue
        fold_origins["split"] = fold["fold"]
        fold_origins = downsample_origins(fold_origins, origin_step_hours)
        dmdc = DMDcModel.fit(train_hourly, STATE_COLS, CONTROL_COLS)
        residual_dmdc = DMDcModel.fit(train_hourly, RESID_STATE_COLS, CONTROL_COLS)
        preds = make_predictions(hourly, fold_origins, dmdc, residual_dmdc, horizon)
        preds["fold"] = fold["fold"]
        preds["cv_train_end"] = fold["train_end"]
        fold_predictions.append(preds)
    return pd.concat(fold_predictions, ignore_index=True) if fold_predictions else pd.DataFrame()


def write_loss_function_notes(
    reports_dir: Path,
    holdout_summary: pd.DataFrame,
    cv_summary: pd.DataFrame,
    high_price_threshold: float,
) -> None:
    lines = [
        "# Loss Function and Metric Notes",
        "",
        "## Regression losses",
        "- MAE is robust to occasional price spikes and is easier to interpret in $/MWh.",
        "- MSE and RMSE penalize large spike misses more heavily, so they are useful for winter storm and summer scarcity periods.",
        "- R2 measures explained variance relative to predicting the sample mean. Negative R2 means the model is worse than that mean baseline for that evaluated group.",
        "- Bias is the mean signed error; negative bias means under-forecasting.",
        "",
        "## High-price ROC/AUC setup",
        f"- The high-price class is defined as observed LMP >= {high_price_threshold:.4f}.",
        "- Forecasted price is used as the classification score. This evaluates ranking ability for high-price hours, not calibrated probability.",
        "",
    ]
    if not holdout_summary.empty:
        best_holdout = holdout_summary.sort_values(["split", "mae"]).groupby("split").head(1)
        lines.append("## Best holdout MAE by split")
        for _, row in best_holdout.iterrows():
            lines.append(
                f"- {row['split']}: {row['model']} MAE={row['mae']:.4f}, "
                f"MSE={row['mse']:.4f}, R2={row['r2']:.4f}"
            )
        lines.append("")
    if not cv_summary.empty:
        best_cv = cv_summary.sort_values(["split", "mae"]).groupby("split").head(1)
        lines.append("## Best expanding-window CV MAE by fold")
        for _, row in best_cv.iterrows():
            lines.append(
                f"- {row['split']}: {row['model']} MAE={row['mae']:.4f}, "
                f"MSE={row['mse']:.4f}, R2={row['r2']:.4f}"
            )

    (reports_dir / "loss_function_analysis.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    hourly, supervised = load_data(args.processed_dir)
    origins = select_origins(supervised, args.origin_step_hours)

    train_hourly = hourly.loc[hourly.index <= pd.Timestamp("2022-12-31 23:00:00+00:00")]
    dmdc = DMDcModel.fit(train_hourly, STATE_COLS, CONTROL_COLS)
    residual_dmdc = DMDcModel.fit(train_hourly, RESID_STATE_COLS, CONTROL_COLS)
    predictions = make_predictions(hourly, origins, dmdc, residual_dmdc, args.horizon)

    model_cols = [
        "seasonal_naive_24h",
        "seasonal_naive_168h",
        "dmdc",
        "dmdc_residual_24h",
    ]
    if args.run_sarimax:
        predictions = add_sarimax_predictions(
            predictions,
            hourly,
            origins,
            args.horizon,
            args.sarimax_train_tail_hours,
            args.sarimax_origin_step_hours,
            args.max_sarimax_origins,
        )
        model_cols.append("sarimax")

    predictions.to_csv(args.reports_dir / "forecast_predictions_24h.csv", index=False)
    by_horizon, summary = metric_table(predictions, model_cols)
    by_horizon.to_csv(args.reports_dir / "forecast_metrics_by_horizon.csv", index=False)
    summary.to_csv(args.reports_dir / "forecast_metrics_summary.csv", index=False)

    high_price_threshold = float(
        train_hourly["lmp_clean"].quantile(args.high_price_quantile)
    )
    roc_summary, roc_points = roc_auc_tables(
        predictions, model_cols, high_price_threshold
    )
    roc_summary.to_csv(args.reports_dir / "roc_auc_summary.csv", index=False)
    roc_points.to_csv(args.reports_dir / "roc_curve_points.csv", index=False)
    write_roc_plot(roc_points, args.reports_dir / "roc_curve_test.png", split="test")

    cv_summary = pd.DataFrame()
    if not args.skip_cv:
        cv_predictions = time_series_cv_predictions(
            hourly,
            supervised,
            args.horizon,
            args.cv_origin_step_hours,
        )
        cv_predictions.to_csv(
            args.reports_dir / "time_series_cv_predictions_24h.csv", index=False
        )
        if not cv_predictions.empty:
            cv_by_horizon, cv_summary = metric_table(cv_predictions, model_cols)
            cv_by_horizon.to_csv(
                args.reports_dir / "time_series_cv_metrics_by_horizon.csv",
                index=False,
            )
            cv_summary.to_csv(
                args.reports_dir / "time_series_cv_metrics_summary.csv",
                index=False,
            )
            cv_roc_summary, cv_roc_points = roc_auc_tables(
                cv_predictions, model_cols, high_price_threshold
            )
            cv_roc_summary.to_csv(
                args.reports_dir / "time_series_cv_roc_auc_summary.csv",
                index=False,
            )
            cv_roc_points.to_csv(
                args.reports_dir / "time_series_cv_roc_curve_points.csv",
                index=False,
            )

    write_loss_function_notes(
        args.reports_dir, summary, cv_summary, high_price_threshold
    )

    print("Finished 24h forecast evaluation")
    print(f"Forecast rows: {len(predictions):,}")
    print(summary.to_string(index=False))
    if not roc_summary.empty:
        print("ROC/AUC summary")
        print(roc_summary.to_string(index=False))
    if not cv_summary.empty:
        print("Expanding-window CV summary")
        print(cv_summary.to_string(index=False))


if __name__ == "__main__":
    main()
