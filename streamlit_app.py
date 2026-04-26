from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"

st.set_page_config(
    layout="wide",
    page_title="PJM Weather-Driven Electricity Price Forecasting",
    page_icon="⚡",
)


MODEL_PRIORITY = [
    "dmdc_residual_24h",
    "dmdc",
    "seasonal_naive_24h",
    "seasonal_naive_168h",
    "sarimax",
]

MODEL_LABELS = {
    "dmdc_residual_24h": "Residual DMDc",
    "dmdc": "DMDc",
    "seasonal_naive_24h": "Seasonal Naive 24h",
    "seasonal_naive_168h": "Seasonal Naive 168h",
    "sarimax": "SARIMAX",
}

NON_MODEL_COLUMNS = {
    "origin_ts",
    "target_ts",
    "timestamp_utc",
    "split",
    "horizon",
    "y_true",
    "actual_lmp",
    "predicted_lmp",
    "error",
    "abs_error",
    "model",
    "n",
    "mae",
    "mse",
    "rmse",
    "r2",
    "bias",
    "context",
    "event_type",
    "event_name",
    "threshold_lmp",
    "positives",
    "negatives",
    "auc",
    "point",
    "fpr",
    "tpr",
    "threshold",
}


st.markdown(
    """
    <style>
    :root {
        --page: #F6FAFF;
        --page-2: #EEF7FF;
        --card: rgba(255, 255, 255, 0.92);
        --card-strong: rgba(255, 255, 255, 0.98);
        --border: rgba(37, 99, 235, 0.14);
        --border-strong: rgba(0, 163, 255, 0.28);
        --text: #0F172A;
        --secondary: #334155;
        --muted: #64748B;
        --blue: #2563EB;
        --electric: #00A3FF;
        --cyan: #22D3EE;
        --soft-cyan: #E0F7FF;
        --violet: #6366F1;
        --success-bg: #ECFDF5;
        --success-text: #065F46;
        --info-bg: #EFF6FF;
        --info-text: #1E3A8A;
        --warning-bg: #FFFBEB;
        --warning-text: #92400E;
        --shadow: 0 18px 48px rgba(37, 99, 235, 0.09);
    }
    header[data-testid="stHeader"] {
        background: transparent;
        height: 0rem;
    }
    div[data-testid="stToolbar"],
    div[data-testid="stStatusWidget"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    div[data-testid="stDecoration"],
    #MainMenu,
    footer {
        display: none;
        visibility: hidden;
    }
    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(0, 163, 255, 0.18), transparent 30rem),
            radial-gradient(circle at 88% 5%, rgba(34, 211, 238, 0.16), transparent 28rem),
            radial-gradient(circle at 50% 100%, rgba(99, 102, 241, 0.10), transparent 36rem),
            linear-gradient(135deg, #F6FAFF 0%, #EEF7FF 48%, #F8FBFF 100%);
        color: var(--text);
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background:
            linear-gradient(rgba(37, 99, 235, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(37, 99, 235, 0.025) 1px, transparent 1px);
        background-size: 58px 58px;
        mask-image: linear-gradient(to bottom, rgba(255, 255, 255, 0.70), transparent 76%);
        z-index: 0;
    }
    .block-container {
        position: relative;
        z-index: 1;
        padding-top: 1.35rem;
        padding-bottom: 2.6rem;
        max-width: 1560px;
    }
    .stMarkdown,
    .stMarkdown p,
    .stMarkdown li,
    label,
    div[data-testid="stCaptionContainer"] {
        color: var(--secondary) !important;
    }
    h1, h2, h3 {
        color: var(--text);
        letter-spacing: -0.025em;
    }
    h2, h3 {
        margin-top: 0.55rem;
        margin-bottom: 0.55rem;
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 1.45rem 1.65rem;
        border: 1px solid var(--border);
        border-radius: 24px;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(255, 255, 255, 0.84)),
            radial-gradient(circle at 8% 0%, rgba(0, 163, 255, 0.10), transparent 36rem);
        box-shadow: var(--shadow);
        backdrop-filter: blur(18px);
        margin-bottom: 1.1rem;
    }
    .hero::before {
        content: "";
        position: absolute;
        top: 0;
        left: 1.65rem;
        right: 1.65rem;
        height: 3px;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--blue), var(--electric), var(--cyan));
        opacity: 0.78;
    }
    .hero::after {
        content: "";
        position: absolute;
        top: -9rem;
        right: -4rem;
        width: 24rem;
        height: 24rem;
        border-radius: 999px;
        background: radial-gradient(circle, rgba(0, 163, 255, 0.12), transparent 68%);
    }
    .hero-title {
        position: relative;
        font-size: clamp(2.15rem, 3.1vw, 3.45rem);
        line-height: 1.03;
        margin: 0 0 0.42rem 0;
        font-weight: 850;
        letter-spacing: -0.055em;
        color: var(--text);
        max-width: 100%;
    }
    .hero-subtitle {
        position: relative;
        color: var(--secondary);
        font-size: 1.02rem;
        line-height: 1.52;
        max-width: 920px;
        margin: 0;
    }
    .metric-card {
        position: relative;
        overflow: hidden;
        padding: 0.95rem 1.0rem;
        border-radius: 18px;
        border: 1px solid var(--border);
        background:
            linear-gradient(150deg, rgba(255, 255, 255, 0.98), rgba(255, 255, 255, 0.86)),
            radial-gradient(circle at 10% 0%, rgba(0, 163, 255, 0.08), transparent 56%);
        box-shadow: 0 14px 34px rgba(37, 99, 235, 0.08);
        min-height: 108px;
        transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: var(--border-strong);
        box-shadow: 0 18px 42px rgba(37, 99, 235, 0.13);
    }
    .metric-label {
        color: var(--muted);
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        margin-bottom: 0.45rem;
    }
    .metric-value {
        color: var(--text);
        font-size: 1.52rem;
        font-weight: 820;
        line-height: 1.15;
        word-break: break-word;
    }
    .metric-note {
        color: var(--muted);
        font-size: 0.82rem;
        margin-top: 0.45rem;
    }
    .section-card {
        padding: 1.0rem 1.1rem;
        border-radius: 20px;
        border: 1px solid var(--border);
        background: var(--card);
        box-shadow: 0 14px 34px rgba(37, 99, 235, 0.08);
        margin: 0.55rem 0 0.95rem 0;
        color: var(--secondary);
    }
    .insight-box {
        padding: 0.95rem 1.05rem;
        border-radius: 18px;
        border: 1px solid rgba(34, 211, 238, 0.34);
        background: linear-gradient(145deg, rgba(224, 247, 255, 0.94), rgba(255, 255, 255, 0.92));
        color: #164E63;
        box-shadow: 0 12px 30px rgba(34, 211, 238, 0.10);
    }
    .warning-box {
        padding: 0.95rem 1.05rem;
        border-radius: 18px;
        border: 1px solid rgba(217, 119, 6, 0.20);
        background: var(--warning-bg);
        color: var(--warning-text);
        box-shadow: 0 12px 30px rgba(217, 119, 6, 0.08);
    }
    .pipeline-step {
        padding: 0.92rem;
        min-height: 88px;
        border-radius: 18px;
        border: 1px solid var(--border);
        background: var(--card-strong);
        text-align: center;
        box-shadow: 0 12px 28px rgba(37, 99, 235, 0.07);
        transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
    }
    .pipeline-step:hover {
        transform: translateY(-2px);
        border-color: var(--border-strong);
        box-shadow: 0 16px 36px rgba(37, 99, 235, 0.12);
    }
    .pipeline-step strong {
        color: var(--text);
        display: block;
        margin-bottom: 0.25rem;
    }
    .pipeline-step span {
        color: var(--muted);
        font-size: 0.82rem;
    }
    div[data-testid="stTabs"] button {
        color: var(--muted);
        font-size: 0.88rem;
        font-weight: 650;
        padding-left: 0.45rem;
        padding-right: 0.45rem;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--blue);
        border-bottom-color: var(--electric);
        font-weight: 800;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.22rem;
        border-bottom: 1px solid rgba(37, 99, 235, 0.12);
        margin-bottom: 0.75rem;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 16px;
        overflow: hidden;
        background: #FFFFFF;
        box-shadow: 0 12px 28px rgba(37, 99, 235, 0.08);
    }
    div[data-testid="stPlotlyChart"] {
        border: 1px solid var(--border);
        border-radius: 18px;
        background: #FFFFFF;
        box-shadow: 0 14px 34px rgba(37, 99, 235, 0.08);
        padding: 0.2rem;
    }
    div.stButton > button {
        min-height: 2.72rem;
        border-radius: 999px;
        border: none;
        background: linear-gradient(135deg, #2563EB 0%, #00A3FF 100%);
        color: #FFFFFF !important;
        font-weight: 800;
        padding: 0.56rem 1.05rem;
        box-shadow: 0 12px 26px rgba(37, 99, 235, 0.22);
    }
    div.stButton > button p,
    div.stButton > button span {
        color: #FFFFFF !important;
        font-weight: 800;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #0284C7 100%);
        box-shadow: 0 14px 30px rgba(37, 99, 235, 0.30);
        color: #FFFFFF !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    input,
    input[type="text"] {
        background-color: #FFFFFF !important;
        color: var(--text) !important;
        border-color: rgba(37, 99, 235, 0.18) !important;
        border-radius: 14px !important;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.06);
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="input"] input {
        color: var(--text) !important;
    }
    div[data-baseweb="select"]:focus-within,
    div[data-baseweb="input"]:focus-within {
        box-shadow: 0 0 0 3px rgba(0, 163, 255, 0.14);
        border-radius: 14px;
    }
    .stSlider [data-baseweb="slider"] div {
        color: var(--blue);
    }
    div[data-testid="stAlert"] {
        border-radius: 16px;
        border: 1px solid rgba(37, 99, 235, 0.14);
        background: rgba(255, 255, 255, 0.92);
        color: var(--text) !important;
        box-shadow: 0 12px 28px rgba(37, 99, 235, 0.08);
    }
    div[data-testid="stAlert"] * {
        color: var(--text) !important;
    }
    div[data-testid="stCodeBlock"] {
        border: 1px solid rgba(37, 99, 235, 0.16);
        border-radius: 14px;
        overflow: hidden;
        background: #EFF6FF !important;
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.08);
    }
    div[data-testid="stCodeBlock"] pre,
    div[data-testid="stCodeBlock"] code,
    code {
        background: #EFF6FF !important;
        color: #1E3A8A !important;
        border-radius: 8px;
    }
    .clean-table-wrap {
        width: 100%;
        overflow: auto;
        border: 1px solid var(--border);
        border-radius: 16px;
        background: #FFFFFF;
        box-shadow: 0 12px 28px rgba(37, 99, 235, 0.08);
        margin: 0.45rem 0 0.9rem 0;
    }
    .clean-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        color: var(--text);
        font-size: 0.88rem;
        line-height: 1.45;
    }
    .clean-table th {
        position: sticky;
        top: 0;
        z-index: 1;
        padding: 0.72rem 0.8rem;
        text-align: left;
        background: linear-gradient(180deg, #F8FBFF 0%, #EEF7FF 100%);
        color: var(--secondary);
        border-bottom: 1px solid rgba(37, 99, 235, 0.14);
        font-weight: 800;
        white-space: nowrap;
    }
    .clean-table td {
        padding: 0.65rem 0.8rem;
        border-bottom: 1px solid rgba(226, 232, 240, 0.88);
        color: var(--text);
        white-space: nowrap;
    }
    .clean-table tr:nth-child(even) td {
        background: #F8FBFF;
    }
    .clean-table tr:hover td {
        background: #EFF6FF;
    }
    .code-pill {
        display: block;
        padding: 0.72rem 0.85rem;
        border: 1px solid rgba(37, 99, 235, 0.18);
        border-radius: 14px;
        background: linear-gradient(135deg, #EFF6FF 0%, #E0F7FF 100%);
        color: #1E3A8A;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.88rem;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.07);
        overflow-wrap: anywhere;
    }
    .status-box {
        padding: 0.86rem 1rem;
        border-radius: 16px;
        border: 1px solid var(--border);
        margin: 0.45rem 0;
        font-weight: 650;
    }
    .status-success {
        background: var(--success-bg);
        border-color: rgba(5, 150, 105, 0.20);
        color: var(--success-text);
    }
    .status-info {
        background: var(--info-bg);
        border-color: rgba(37, 99, 235, 0.18);
        color: var(--info-text);
    }
    .status-warning {
        background: var(--warning-bg);
        border-color: rgba(217, 119, 6, 0.22);
        color: var(--warning-text);
    }
    .method-hero {
        padding: 1.25rem 1.35rem;
        border: 1px solid var(--border);
        border-radius: 22px;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(239, 246, 255, 0.86)),
            radial-gradient(circle at 0% 0%, rgba(0, 163, 255, 0.10), transparent 30rem);
        box-shadow: var(--shadow);
        margin-bottom: 0.9rem;
    }
    .method-hero-title {
        margin: 0 0 0.25rem 0;
        color: var(--text);
        font-size: 1.65rem;
        font-weight: 850;
        letter-spacing: -0.035em;
    }
    .method-hero-subtitle {
        margin: 0;
        color: var(--secondary);
        font-size: 0.98rem;
        line-height: 1.5;
    }
    .method-card {
        min-height: 180px;
        padding: 1rem 1.05rem;
        border: 1px solid var(--border);
        border-radius: 18px;
        background: var(--card-strong);
        box-shadow: 0 12px 28px rgba(37, 99, 235, 0.07);
        margin-bottom: 0.85rem;
    }
    .method-label {
        color: var(--blue);
        font-size: 0.72rem;
        font-weight: 850;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .method-title {
        color: var(--text);
        font-size: 1.02rem;
        font-weight: 820;
        margin-bottom: 0.5rem;
    }
    .method-card ul {
        margin: 0;
        padding-left: 1.05rem;
    }
    .method-card li {
        color: var(--secondary);
        margin-bottom: 0.28rem;
        line-height: 1.42;
    }
    .formula-card {
        padding: 1rem 1.1rem;
        border: 1px solid var(--border);
        border-radius: 20px;
        background: var(--card-strong);
        box-shadow: 0 12px 28px rgba(37, 99, 235, 0.07);
        margin-top: 0.3rem;
    }
    @media (min-width: 1180px) {
        .hero-title {
            white-space: nowrap;
        }
    }
    @media (max-width: 1100px) {
        .hero-title {
            font-size: clamp(2.0rem, 5vw, 3.2rem);
            white-space: normal;
        }
        div[data-testid="stTabs"] button {
            font-size: 0.78rem;
            padding-left: 0.25rem;
            padding-right: 0.25rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(filename: str) -> pd.DataFrame:
    """Load a report CSV. Missing files return an empty DataFrame."""
    path = REPORTS_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    for col in ["origin_ts", "target_ts", "timestamp_utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def missing_warning(filename: str, description: str) -> None:
    st.warning(f"Missing `{filename}`. {description}")


def pretty_model_name(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " ").title())


def ordered_models(models: Iterable[str]) -> list[str]:
    present = list(dict.fromkeys(models))
    ordered = [m for m in MODEL_PRIORITY if m in present]
    ordered.extend(sorted(m for m in present if m not in ordered))
    return ordered


def detect_model_columns(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    candidates = [
        col for col in df.columns
        if col in numeric_cols and col not in NON_MODEL_COLUMNS
    ]
    return ordered_models(candidates)


def detect_metric_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in ["mae", "rmse", "r2", "bias"] if col in df.columns]


def format_number(value: object, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return str(value)


def metric_card(label: str, value: object, note: str | None = None, digits: int = 2) -> None:
    display_value = escape(format_number(value, digits=digits))
    display_label = escape(label)
    display_note = f"<div class='metric-note'>{escape(note)}</div>" if note else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{display_label}</div>
            <div class="metric-value">{display_value}</div>
            {display_note}
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_box(text: str) -> None:
    st.markdown(f"<div class='insight-box'>{text}</div>", unsafe_allow_html=True)


def warning_box(text: str) -> None:
    st.markdown(f"<div class='warning-box'>{text}</div>", unsafe_allow_html=True)


def render_clean_table(df: pd.DataFrame, max_rows: int | None = None, height: int | None = None) -> None:
    display_df = df.copy()
    truncated = False
    if max_rows is not None and len(display_df) > max_rows:
        display_df = display_df.head(max_rows)
        truncated = True
    max_height = f"max-height: {height}px;" if height is not None else ""
    table_html = display_df.to_html(index=False, classes="clean-table", border=0, escape=True)
    note = ""
    if truncated:
        note = (
            f"<div style='padding:0.55rem 0.8rem;color:#64748B;font-size:0.82rem;'>"
            f"Showing first {max_rows:,} of {len(df):,} rows.</div>"
        )
    st.markdown(
        f"<div class='clean-table-wrap' style='{max_height}'>{table_html}{note}</div>",
        unsafe_allow_html=True,
    )


def code_pill(text: str) -> None:
    st.markdown(f"<div class='code-pill'>{escape(text)}</div>", unsafe_allow_html=True)


def status_success_box(text: str) -> None:
    st.markdown(f"<div class='status-box status-success'>{escape(text)}</div>", unsafe_allow_html=True)


def status_info_box(text: str) -> None:
    st.markdown(f"<div class='status-box status-info'>{escape(text)}</div>", unsafe_allow_html=True)


def status_warning_box(text: str) -> None:
    st.markdown(f"<div class='status-box status-warning'>{escape(text)}</div>", unsafe_allow_html=True)


def safe_regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if frame.empty:
        return {"mae": np.nan, "rmse": np.nan, "bias": np.nan, "max_abs_error": np.nan}
    error = frame["predicted"] - frame["actual"]
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(error.mean()),
        "max_abs_error": float(error.abs().max()),
    }


def best_metric_row(metrics: pd.DataFrame) -> tuple[pd.Series | None, str]:
    if metrics.empty or not {"model", "mae"}.issubset(metrics.columns):
        return None, "No model metric file was available."
    if "split" in metrics.columns and (metrics["split"] == "test").any():
        subset = metrics[metrics["split"] == "test"].copy()
        note = "Selected from the test split by lowest MAE."
    else:
        subset = metrics.copy()
        note = "Test split was not found, so all available rows were used."
    if subset.empty:
        return None, note
    return subset.sort_values("mae", ascending=True).iloc[0], note


def split_options(df: pd.DataFrame) -> list[str]:
    if df.empty or "split" not in df.columns:
        return ["all available"]
    preferred = [s for s in ["test", "validation", "train"] if s in set(df["split"].dropna())]
    remaining = sorted(s for s in df["split"].dropna().unique() if s not in preferred)
    return ["all available", *preferred, *remaining]


def apply_split_filter(df: pd.DataFrame, selected_split: str) -> pd.DataFrame:
    if selected_split == "all available" or "split" not in df.columns:
        return df.copy()
    return df[df["split"] == selected_split].copy()


def plot_template() -> str:
    return "plotly_white"


def style_plotly_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#0F172A", "family": "Inter, Arial, sans-serif"},
        margin={"l": 40, "r": 30, "t": 70, "b": 45},
        colorway=["#2563EB", "#00A3FF", "#22D3EE", "#5EEAD4", "#6366F1", "#059669"],
        legend={
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "rgba(37,99,235,0.12)",
            "borderwidth": 1,
            "font": {"color": "#334155"},
        },
        hoverlabel={
            "bgcolor": "white",
            "bordercolor": "#00A3FF",
            "font": {"color": "#0F172A"},
        },
    )
    fig.update_xaxes(
        gridcolor="#E2E8F0",
        zerolinecolor="#CBD5E1",
        linecolor="#CBD5E1",
        tickfont={"color": "#334155"},
        title_font={"color": "#334155"},
    )
    fig.update_yaxes(
        gridcolor="#E2E8F0",
        zerolinecolor="#CBD5E1",
        linecolor="#CBD5E1",
        tickfont={"color": "#334155"},
        title_font={"color": "#334155"},
    )
    return fig


def add_model_pretty(df: pd.DataFrame, model_col: str = "model") -> pd.DataFrame:
    out = df.copy()
    if model_col in out.columns:
        out["model_pretty"] = out[model_col].map(pretty_model_name)
    return out


def render_project_overview(metrics_summary: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">PJM Weather-Driven Electricity Price Forecasting</div>
            <p class="hero-subtitle">
                Interactive analysis of summer heatwaves, winter blizzards, and
                24-hour-ahead PJM electricity price forecasts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    best, note = best_metric_row(metrics_summary)
    cols = st.columns(4)
    if best is not None:
        with cols[0]:
            metric_card("Best available model", pretty_model_name(str(best["model"])), note, digits=2)
        with cols[1]:
            metric_card("Test MAE", best.get("mae"), "$/MWh" if best.get("split", "test") == "test" else "Selected row", digits=2)
        with cols[2]:
            metric_card("Test RMSE", best.get("rmse"), "$/MWh" if best.get("split", "test") == "test" else "Selected row", digits=2)
        with cols[3]:
            metric_card("Test R²", best.get("r2"), "Coefficient of determination", digits=3)
    else:
        for col, label in zip(cols, ["Best available model", "Test MAE", "Test RMSE", "Test R²"]):
            with col:
                metric_card(label, "N/A", "Run the pipeline to create forecast_metrics_summary.csv")

    st.markdown(
        """
        <div class="section-card">
        This project studies how extreme weather conditions affect electricity price behavior.
        The app visualizes model performance, forecast errors, high-price detection, and
        event-specific performance. It is a historical analysis and forecast replay dashboard,
        not a live market trading system.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Project pipeline")
    pipeline = [
        ("Data", "PJM LMP, load, weather"),
        ("Feature Engineering", "Lags, time features, event windows"),
        ("Forecast Models", "Seasonal naive, DMDc, residual DMDc"),
        ("Event Evaluation", "Heatwaves, cold waves, blizzards"),
        ("Interactive Demo", "Replay 24-hour forecasts"),
    ]
    cols = st.columns(len(pipeline))
    for col, (title, subtitle) in zip(cols, pipeline):
        with col:
            st.markdown(
                f"<div class='pipeline-step'><strong>{escape(title)}</strong><span>{escape(subtitle)}</span></div>",
                unsafe_allow_html=True,
            )


def render_methodology(metrics_summary: pd.DataFrame, predictions: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="method-hero">
            <div class="method-hero-title">Methodology</div>
            <p class="method-hero-subtitle">
                A structured 24-hour-ahead forecasting workflow combining PJM price data,
                weather stress features, and model diagnostics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    available = []
    if not metrics_summary.empty and "model" in metrics_summary.columns:
        available = ordered_models(metrics_summary["model"].dropna().unique())
    elif not predictions.empty:
        available = detect_model_columns(predictions)
    available_models = ", ".join(pretty_model_name(model) for model in available) if available else "Model outputs were not found yet."

    cards = [
        (
            "01 Data",
            "Data and target",
            [
                "Target: PJM Dominion day-ahead LMP.",
                "Inputs include historical PJM load and observed weather features.",
                "Forecasting task: 24-hour-ahead LMP prediction.",
            ],
        ),
        (
            "02 Weather",
            "Weather stress features",
            [
                "Temperature, wind, precipitation, HDD, and CDD features.",
                "Summer heat-price periods and sudden heat events.",
                "Winter cold waves, winter storms, and blizzard-style stress periods.",
            ],
        ),
        (
            "03 Forecast",
            "Forecasting setup",
            [
                "Each forecast origin produces 24 target horizons.",
                "Validation/test periods are chronological to avoid leakage.",
                "Expanding-window cross-validation is used where available.",
            ],
        ),
        (
            "04 Models",
            "Model family",
            [
                f"Detected outputs: {available_models}",
                "Seasonal naive baselines preserve simple operational benchmarks.",
                "DMDc and residual-corrected DMDc are used when present in outputs.",
            ],
        ),
        (
            "05 Metrics",
            "Evaluation metrics",
            [
                "MAE, RMSE, Bias, and R2 evaluate regression quality.",
                "ROC/AUC evaluates high-price detection behavior.",
                "Event metrics isolate performance during weather stress windows.",
            ],
        ),
    ]

    for row_start in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, (label, title, bullets) in zip(cols, cards[row_start:row_start + 3]):
            bullet_html = "".join(f"<li>{escape(item)}</li>" for item in bullets)
            with col:
                st.markdown(
                    f"""
                    <div class="method-card">
                        <div class="method-label">{escape(label)}</div>
                        <div class="method-title">{escape(title)}</div>
                        <ul>{bullet_html}</ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown(
        """
        <div class="formula-card">
            <div class="method-label">Metric formulas</div>
            <div class="method-title">Regression diagnostics used throughout the dashboard</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    formula_cols = st.columns(2)
    with formula_cols[0]:
        st.latex(r"MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|")
        st.latex(r"Bias = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)")
    with formula_cols[1]:
        st.latex(r"RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}")
        st.latex(r"R^2 = 1 - \frac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y})^2}")
    insight_box(
        "ROC/AUC is used as a high-price detection diagnostic by treating forecasted LMP as a ranking score."
    )


def render_high_price_detection(roc_auc_summary: pd.DataFrame, roc_curve_points: pd.DataFrame) -> None:
    st.subheader("High-price detection ROC/AUC")
    if roc_auc_summary.empty and roc_curve_points.empty:
        missing_warning(
            "roc_auc_summary.csv / roc_curve_points.csv",
            "High-price detection diagnostics are unavailable.",
        )
        return

    filter_base = roc_auc_summary if not roc_auc_summary.empty else roc_curve_points
    split_choices = split_options(filter_base)
    default_split_index = split_choices.index("test") if "test" in split_choices else 0

    context_col = next(
        (col for col in ["context", "event_type", "event_name"] if col in filter_base.columns),
        None,
    )
    context_choices = ["all available"]
    if context_col:
        context_choices.extend(sorted(str(v) for v in filter_base[context_col].dropna().unique()))

    controls = st.columns([1.0, 1.0, 2.0])
    with controls[0]:
        selected_split = st.selectbox(
            "ROC split",
            split_choices,
            index=default_split_index,
            key="roc_split",
        )
    with controls[1]:
        selected_context = st.selectbox(
            "ROC context",
            context_choices,
            key="roc_context",
        )

    def filter_roc_frame(df: pd.DataFrame) -> pd.DataFrame:
        out = apply_split_filter(df, selected_split)
        local_context_col = next(
            (col for col in ["context", "event_type", "event_name"] if col in out.columns),
            None,
        )
        if local_context_col and selected_context != "all available":
            out = out[out[local_context_col].astype(str) == selected_context]
        return out.copy()

    if not roc_auc_summary.empty:
        if {"model", "auc"}.issubset(roc_auc_summary.columns):
            auc_df = filter_roc_frame(roc_auc_summary)
            auc_df = auc_df[auc_df["auc"].notna()].copy()
            auc_df = add_model_pretty(auc_df)
            if auc_df.empty:
                st.warning("No AUC values are available for the selected ROC controls.")
            else:
                color_col = (
                    "context"
                    if "context" in auc_df.columns and selected_context == "all available"
                    else "model_pretty"
                )
                auc_fig = px.bar(
                    auc_df.sort_values("auc", ascending=False),
                    x="model_pretty",
                    y="auc",
                    color=color_col,
                    barmode="group",
                    template=plot_template(),
                    labels={"model_pretty": "Model", "auc": "AUC"},
                    title="High-price detection AUC by model",
                )
                auc_fig.update_layout(showlegend=color_col != "model_pretty", xaxis_tickangle=-25)
                auc_fig.update_yaxes(range=[0, 1])
                st.plotly_chart(style_plotly_fig(auc_fig), use_container_width=True)

                display_cols = [
                    col for col in ["model", "split", "context", "threshold_lmp", "n", "positives", "negatives", "auc"]
                    if col in auc_df.columns
                ]
                render_clean_table(auc_df[display_cols], max_rows=500, height=360)
        else:
            st.warning("`roc_auc_summary.csv` exists but does not contain model and auc columns.")
    else:
        missing_warning("roc_auc_summary.csv", "AUC summary is unavailable.")

    if not roc_curve_points.empty:
        if {"model", "fpr", "tpr"}.issubset(roc_curve_points.columns):
            roc_df = filter_roc_frame(roc_curve_points)
            model_options = ordered_models(roc_df["model"].dropna().unique()) if "model" in roc_df.columns else []
            if model_options:
                selected_models = st.multiselect(
                    "ROC curve models",
                    model_options,
                    default=model_options[: min(4, len(model_options))],
                    format_func=pretty_model_name,
                    key="roc_models",
                )
                roc_df = roc_df[roc_df["model"].isin(selected_models)].copy()
            roc_df = add_model_pretty(roc_df)
            if roc_df.empty:
                st.warning("No ROC curve points are available for the selected controls.")
            else:
                curve_fig = px.line(
                    roc_df.sort_values(["model", "fpr", "tpr"]),
                    x="fpr",
                    y="tpr",
                    color="model_pretty",
                    template=plot_template(),
                    labels={"fpr": "False positive rate", "tpr": "True positive rate", "model_pretty": "Model"},
                    title="ROC curve for high-price detection",
                )
                curve_fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Random baseline",
                        line={"dash": "dash", "color": "rgba(148,163,184,0.75)"},
                    )
                )
                curve_fig.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1])
                st.plotly_chart(style_plotly_fig(curve_fig), use_container_width=True)
        else:
            st.warning("`roc_curve_points.csv` exists but does not contain model/fpr/tpr columns.")
    else:
        missing_warning("roc_curve_points.csv", "ROC curve points are unavailable.")


def render_model_results(
    metrics_summary: pd.DataFrame,
    metrics_by_horizon: pd.DataFrame,
    roc_auc_summary: pd.DataFrame,
    roc_curve_points: pd.DataFrame,
) -> None:
    st.header("Model Results")
    if metrics_summary.empty:
        missing_warning("forecast_metrics_summary.csv", "Model comparison is unavailable.")
        return
    required = {"model", "mae"}
    if not required.issubset(metrics_summary.columns):
        st.warning("`forecast_metrics_summary.csv` does not contain the required model/mae columns.")
        return

    controls = st.columns([1.1, 1.1, 2.4])
    with controls[0]:
        selected_split = st.selectbox("Split", split_options(metrics_summary), key="model_results_split")
    metric_cols = detect_metric_columns(metrics_summary)
    with controls[1]:
        selected_metric = st.selectbox("Metric", metric_cols or ["mae"], key="model_results_metric")

    plot_df = apply_split_filter(metrics_summary, selected_split)
    plot_df = plot_df[plot_df[selected_metric].notna()].copy()
    plot_df = add_model_pretty(plot_df)
    ascending = selected_metric != "r2"
    sorted_df = plot_df.sort_values(selected_metric, ascending=ascending)

    if sorted_df.empty:
        st.warning("No model metrics are available for the selected controls.")
    else:
        fig = px.bar(
            sorted_df,
            x="model_pretty",
            y=selected_metric,
            color="split" if "split" in sorted_df.columns and selected_split == "all available" else "model_pretty",
            barmode="group",
            template=plot_template(),
            labels={"model_pretty": "Model", selected_metric: selected_metric.upper()},
            title=f"Model comparison by {selected_metric.upper()}",
        )
        fig.update_layout(showlegend=selected_split == "all available", xaxis_tickangle=-25)
        st.plotly_chart(style_plotly_fig(fig), use_container_width=True)
        render_clean_table(sorted_df.drop(columns=["model_pretty"], errors="ignore"), height=360)

        best_mae = plot_df.dropna(subset=["mae"]).sort_values("mae").head(1)
        insight_lines = []
        if not best_mae.empty:
            row = best_mae.iloc[0]
            insight_lines.append(
                f"<strong>Best by MAE:</strong> {escape(pretty_model_name(str(row['model'])))} "
                f"with MAE {format_number(row['mae'])}."
            )
        residual = plot_df[plot_df["model"] == "dmdc_residual_24h"]
        naive = plot_df[plot_df["model"] == "seasonal_naive_24h"]
        if not residual.empty and not naive.empty and "mae" in plot_df.columns:
            residual_mae = float(residual["mae"].mean())
            naive_mae = float(naive["mae"].mean())
            diff = naive_mae - residual_mae
            direction = "improves on" if diff > 0 else "does not improve on"
            insight_lines.append(
                f"<strong>Residual DMDc comparison:</strong> it {direction} Seasonal Naive 24h "
                f"by {format_number(abs(diff))} MAE points under the selected view."
            )

        horizon_df = apply_split_filter(metrics_by_horizon, selected_split)
        if not horizon_df.empty and {"horizon", "mae", "model"}.issubset(horizon_df.columns):
            worst = horizon_df.dropna(subset=["mae"]).sort_values("mae", ascending=False).head(1)
            if not worst.empty:
                w = worst.iloc[0]
                insight_lines.append(
                    f"<strong>Largest horizon MAE:</strong> horizon {int(w['horizon'])} "
                    f"for {escape(pretty_model_name(str(w['model'])))}."
                )
        if insight_lines:
            insight_box("<br>".join(insight_lines))

    st.subheader("Forecast horizon diagnostics")
    if metrics_by_horizon.empty:
        missing_warning("forecast_metrics_by_horizon.csv", "Horizon-level diagnostics are unavailable.")
        render_high_price_detection(roc_auc_summary, roc_curve_points)
        return
    if "model" not in metrics_by_horizon.columns or "horizon" not in metrics_by_horizon.columns:
        st.warning("`forecast_metrics_by_horizon.csv` does not contain model and horizon columns.")
        render_high_price_detection(roc_auc_summary, roc_curve_points)
        return

    horizon_models = ordered_models(metrics_by_horizon["model"].dropna().unique())
    selected_model = st.selectbox(
        "Horizon model",
        horizon_models,
        format_func=pretty_model_name,
        key="horizon_model",
    )
    horizon_df = metrics_by_horizon[metrics_by_horizon["model"] == selected_model].copy()
    horizon_df = apply_split_filter(horizon_df, selected_split)
    if horizon_df.empty:
        st.warning("No horizon metrics are available for this model and split.")
        render_high_price_detection(roc_auc_summary, roc_curve_points)
        return

    fig = go.Figure()
    split_values = horizon_df["split"].dropna().unique() if "split" in horizon_df.columns else ["all"]
    for split in split_values:
        subset = horizon_df if split == "all" else horizon_df[horizon_df["split"] == split]
        label_suffix = "" if selected_split != "all available" else f" ({split})"
        if "mae" in subset.columns:
            fig.add_trace(go.Scatter(x=subset["horizon"], y=subset["mae"], mode="lines+markers", name=f"MAE{label_suffix}"))
        if "rmse" in subset.columns:
            fig.add_trace(go.Scatter(x=subset["horizon"], y=subset["rmse"], mode="lines+markers", name=f"RMSE{label_suffix}"))
    fig.update_layout(
        template=plot_template(),
        title=f"MAE and RMSE by horizon: {pretty_model_name(selected_model)}",
        xaxis_title="Forecast horizon",
        yaxis_title="Error",
    )
    st.plotly_chart(style_plotly_fig(fig), use_container_width=True)

    if "bias" in horizon_df.columns:
        bias_fig = px.line(
            horizon_df,
            x="horizon",
            y="bias",
            color="split" if selected_split == "all available" and "split" in horizon_df.columns else None,
            markers=True,
            template=plot_template(),
            title=f"Bias by horizon: {pretty_model_name(selected_model)}",
        )
        bias_fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(style_plotly_fig(bias_fig), use_container_width=True)

    render_high_price_detection(roc_auc_summary, roc_curve_points)


def prepare_prediction_subset(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    out = predictions.copy()
    if "y_true" in out.columns and model in out.columns:
        out["actual_lmp"] = out["y_true"]
        out["predicted_lmp"] = out[model]
        out["error"] = out["predicted_lmp"] - out["actual_lmp"]
        out["abs_error"] = out["error"].abs()
    return out


def render_forecast_explorer(predictions: pd.DataFrame) -> None:
    st.header("Forecast Explorer")
    if predictions.empty:
        missing_warning("forecast_predictions_24h.csv", "Forecast explorer is unavailable.")
        return
    if not {"horizon", "y_true"}.issubset(predictions.columns):
        st.warning("Forecast predictions must contain `horizon` and `y_true` columns.")
        return
    model_cols = detect_model_columns(predictions)
    if not model_cols:
        st.warning("No model prediction columns were detected.")
        return

    controls = st.columns([1.0, 1.2, 1.2, 2.2])
    with controls[0]:
        selected_split = st.selectbox("Split", split_options(predictions), key="explorer_split")
    with controls[1]:
        selected_model = st.selectbox("Model", model_cols, format_func=pretty_model_name, key="explorer_model")
    horizon_min = int(predictions["horizon"].min())
    horizon_max = int(predictions["horizon"].max())
    with controls[2]:
        selected_horizon = st.slider("Horizon", min_value=horizon_min, max_value=horizon_max, value=min(24, horizon_max))

    filtered = apply_split_filter(predictions, selected_split)
    filtered = filtered[filtered["horizon"] == selected_horizon].copy()
    if "target_ts" in filtered.columns and filtered["target_ts"].notna().any():
        min_date = filtered["target_ts"].min().date()
        max_date = filtered["target_ts"].max().date()
        with controls[3]:
            selected_range = st.date_input(
                "Target date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start = pd.Timestamp(selected_range[0], tz="UTC")
            end = pd.Timestamp(selected_range[1], tz="UTC") + pd.Timedelta(days=1)
            filtered = filtered[(filtered["target_ts"] >= start) & (filtered["target_ts"] < end)]
    else:
        st.info("`target_ts` is unavailable, so date filtering is disabled.")

    filtered = prepare_prediction_subset(filtered, selected_model).dropna(subset=["actual_lmp", "predicted_lmp"])
    if filtered.empty:
        st.warning("No forecast rows are available for the selected controls.")
        return

    metrics = safe_regression_metrics(filtered["actual_lmp"], filtered["predicted_lmp"])
    cols = st.columns(4)
    with cols[0]:
        metric_card("MAE", metrics["mae"], "$/MWh")
    with cols[1]:
        metric_card("RMSE", metrics["rmse"], "$/MWh")
    with cols[2]:
        metric_card("Bias", metrics["bias"], "Predicted - actual")
    with cols[3]:
        metric_card("Max absolute error", metrics["max_abs_error"], "$/MWh")

    if "target_ts" in filtered.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered["target_ts"], y=filtered["actual_lmp"], mode="lines", name="Actual LMP"))
        fig.add_trace(go.Scatter(x=filtered["target_ts"], y=filtered["predicted_lmp"], mode="lines", name=pretty_model_name(selected_model)))
        fig.update_layout(template=plot_template(), title="Actual vs predicted LMP", xaxis_title="Target timestamp", yaxis_title="LMP ($/MWh)")
        st.plotly_chart(style_plotly_fig(fig), use_container_width=True)

        err_fig = px.line(
            filtered,
            x="target_ts",
            y="error",
            template=plot_template(),
            title="Forecast error over time",
            labels={"error": "Predicted - actual"},
        )
        err_fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(style_plotly_fig(err_fig), use_container_width=True)
    else:
        st.warning("Cannot plot time-series charts because `target_ts` is missing.")

    hist_fig = px.histogram(
        filtered,
        x="error",
        nbins=60,
        template=plot_template(),
        title="Residual distribution",
        labels={"error": "Predicted - actual"},
    )
    st.plotly_chart(style_plotly_fig(hist_fig), use_container_width=True)

    preview_cols = [col for col in ["origin_ts", "target_ts", "horizon", "actual_lmp", "predicted_lmp", "error", "abs_error"] if col in filtered.columns]
    render_clean_table(filtered[preview_cols], max_rows=500, height=420)


def render_weather_event_lab(event_metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    st.header("Weather Event Lab")
    warning_box(
        "This lab helps identify whether extreme or high-stress conditions cause systematic under-prediction."
    )

    if event_metrics.empty:
        missing_warning("event_forecast_metrics.csv", "Event-based performance tables are unavailable.")
    else:
        st.subheader("Event-based model performance")
        event_col = next((col for col in ["event_type", "event_name", "context"] if col in event_metrics.columns), None)
        metric_options = [col for col in ["mae", "rmse"] if col in event_metrics.columns]
        if event_col and metric_options and "model" in event_metrics.columns:
            controls = st.columns([1.2, 1.2, 1.8])
            with controls[0]:
                selected_metric = st.selectbox("Event metric", metric_options, key="event_metric")
            with controls[1]:
                selected_split = st.selectbox("Event split", split_options(event_metrics), key="event_split")
            event_df = apply_split_filter(event_metrics, selected_split)
            if event_col in ["event_type", "event_name"]:
                with controls[2]:
                    selected_events = st.multiselect(
                        "Event type",
                        sorted(event_df[event_col].dropna().unique()),
                        default=sorted(event_df[event_col].dropna().unique()),
                    )
                    event_df = event_df[event_df[event_col].isin(selected_events)]
            event_df = add_model_pretty(event_df)
            fig = px.bar(
                event_df,
                x=event_col,
                y=selected_metric,
                color="model_pretty",
                barmode="group",
                template=plot_template(),
                title=f"{selected_metric.upper()} by event type and model",
                labels={event_col: "Event type", "model_pretty": "Model"},
            )
            fig.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(style_plotly_fig(fig), use_container_width=True)
        else:
            st.info("Event metrics were found, but event/model/metric columns were incomplete.")
        render_clean_table(event_metrics, max_rows=500, height=420)

    st.subheader("Weather Stress Lens")
    st.caption(
        "Exploratory historical lens. If explicit event labels are not present in forecast rows, "
        "the app uses price-stress proxies rather than meteorological event labels."
    )

    if predictions.empty:
        missing_warning("forecast_predictions_24h.csv", "The stress lens needs forecast rows.")
        return
    model_cols = detect_model_columns(predictions)
    if not model_cols or "y_true" not in predictions.columns:
        st.warning("Forecast rows need `y_true` and at least one model prediction column.")
        return

    stress_controls = st.columns([1.4, 1.4, 1.4])
    with stress_controls[0]:
        scenario = st.selectbox(
            "Scenario",
            ["Heatwave scenario", "Winter blizzard scenario", "Normal weather baseline"],
        )
    with stress_controls[1]:
        selected_model = st.selectbox("Stress lens model", model_cols, format_func=pretty_model_name)
    with stress_controls[2]:
        selected_split = st.selectbox("Stress lens split", split_options(predictions), key="stress_split")

    lens = apply_split_filter(predictions, selected_split).copy()
    lens = prepare_prediction_subset(lens, selected_model).dropna(subset=["actual_lmp", "predicted_lmp"])
    if lens.empty:
        st.warning("No prediction rows are available for the selected stress lens controls.")
        return

    if "target_ts" in lens.columns:
        months = lens["target_ts"].dt.month
    else:
        months = pd.Series(np.nan, index=lens.index)

    high_threshold = float(lens["actual_lmp"].quantile(0.90))
    normal_low = float(lens["actual_lmp"].quantile(0.40))
    normal_high = float(lens["actual_lmp"].quantile(0.60))
    if scenario == "Heatwave scenario":
        subset = lens[(months.isin([6, 7, 8])) & (lens["actual_lmp"] >= high_threshold)]
        proxy_note = "Summer top-10% actual LMP rows are used as a heatwave price-stress proxy."
    elif scenario == "Winter blizzard scenario":
        subset = lens[(months.isin([12, 1, 2, 3])) & (lens["actual_lmp"] >= high_threshold)]
        proxy_note = "Winter top-10% actual LMP rows are used as a blizzard price-stress proxy."
    else:
        subset = lens[(lens["actual_lmp"] >= normal_low) & (lens["actual_lmp"] <= normal_high)]
        proxy_note = "Middle-price rows are used as a normal baseline proxy."
    if subset.empty:
        st.warning("No rows matched this scenario. Try another split or model.")
        return

    subset_error = subset["predicted_lmp"] - subset["actual_lmp"]
    actual_high = subset["actual_lmp"] >= high_threshold
    miss_rate = np.nan
    if int(actual_high.sum()) > 0:
        miss_rate = float(((subset["predicted_lmp"] < high_threshold) & actual_high).sum() / actual_high.sum())

    cols = st.columns(4)
    with cols[0]:
        metric_card("Average actual LMP", subset["actual_lmp"].mean(), "$/MWh")
    with cols[1]:
        metric_card("Average predicted LMP", subset["predicted_lmp"].mean(), "$/MWh")
    with cols[2]:
        metric_card("Average forecast error", subset_error.mean(), "Predicted - actual")
    with cols[3]:
        metric_card("High-price miss rate", miss_rate * 100 if pd.notna(miss_rate) else np.nan, "% of actual high-price rows")

    warning_box(proxy_note + " This is not a physical weather simulator.")


def replay_status_messages(mae: float, bias: float, actual_std: float, actual_peak: float, predicted_peak: float) -> None:
    if pd.notna(actual_std) and actual_std > 0 and mae < actual_std:
        status_success_box("Good forecast replay: the 24-hour MAE is low relative to the actual LMP standard deviation.")
    else:
        status_info_box("Forecast replay is challenging for this origin: the MAE is large relative to the daily variability.")
    if pd.notna(bias) and bias < 0:
        status_warning_box("The model under-predicted the 24-hour average LMP.")
    elif pd.notna(bias):
        status_info_box("The model over-predicted the 24-hour average LMP.")
    if pd.notna(actual_peak) and pd.notna(predicted_peak) and actual_peak > predicted_peak * 1.2:
        status_warning_box("Possible missed price spike: actual peak LMP is much higher than predicted peak LMP.")


def render_replay_demo(predictions: pd.DataFrame) -> None:
    st.header("Historical Forecast Replay Demo")
    st.caption("A historical replay of one 24-hour forecast origin. This is not real-time forecasting.")
    if predictions.empty:
        missing_warning("forecast_predictions_24h.csv", "Historical replay is unavailable.")
        return
    if not {"origin_ts", "horizon", "y_true"}.issubset(predictions.columns):
        st.warning("Replay requires `origin_ts`, `horizon`, and `y_true` columns.")
        return
    model_cols = detect_model_columns(predictions)
    if not model_cols:
        st.warning("No model prediction columns were detected.")
        return

    controls = st.columns([1.2, 1.4, 1.1, 2.2])
    with controls[0]:
        selected_split = st.selectbox("Replay split", split_options(predictions), key="replay_split")
    replay_df = apply_split_filter(predictions, selected_split).copy()
    with controls[1]:
        selected_model = st.selectbox("Replay model", model_cols, format_func=pretty_model_name, key="replay_model")
    with controls[2]:
        st.caption("Replay action")
        random_clicked = st.button("Random forecast origin", type="primary")

    origins = replay_df["origin_ts"].dropna().drop_duplicates().sort_values().to_list()
    if not origins:
        st.warning("No forecast origins are available for the selected split.")
        return

    state_key = f"replay_origin_{selected_split}"
    if random_clicked or state_key not in st.session_state:
        st.session_state[state_key] = pd.Timestamp(np.random.choice(origins))
    selected_origin = pd.Timestamp(st.session_state[state_key])
    if selected_origin not in set(origins):
        selected_origin = pd.Timestamp(origins[0])
        st.session_state[state_key] = selected_origin

    with controls[3]:
        st.caption("Selected forecast origin")
        code_pill(str(selected_origin))

    sample = replay_df[replay_df["origin_ts"] == selected_origin].copy()
    sample = prepare_prediction_subset(sample, selected_model).dropna(subset=["actual_lmp", "predicted_lmp"])
    sample = sample.sort_values("horizon")
    if sample.empty:
        st.warning("The selected origin does not have usable prediction rows for this model.")
        return

    metrics = safe_regression_metrics(sample["actual_lmp"], sample["predicted_lmp"])
    worst_row = sample.loc[sample["abs_error"].idxmax()]
    actual_peak = float(sample["actual_lmp"].max())
    predicted_peak = float(sample["predicted_lmp"].max())

    st.subheader("Selected sample")
    sample_cols = st.columns(4)
    with sample_cols[0]:
        metric_card("Forecast origin time", str(selected_origin), "Historical replay", digits=2)
    with sample_cols[1]:
        metric_card("Split", selected_split, "Evaluation partition", digits=2)
    with sample_cols[2]:
        metric_card("Model", pretty_model_name(selected_model), "Loaded from reports", digits=2)
    with sample_cols[3]:
        metric_card("Forecast horizons", len(sample), "Rows in this replay", digits=0)

    st.subheader("24-hour prediction summary")
    summary_cols = st.columns(6)
    with summary_cols[0]:
        metric_card("24h MAE", metrics["mae"], "$/MWh")
    with summary_cols[1]:
        metric_card("24h RMSE", metrics["rmse"], "$/MWh")
    with summary_cols[2]:
        metric_card("Bias", metrics["bias"], "Predicted - actual")
    with summary_cols[3]:
        metric_card("Worst horizon", int(worst_row["horizon"]), f"Abs error {format_number(worst_row['abs_error'])}", digits=0)
    with summary_cols[4]:
        metric_card("Actual peak LMP", actual_peak, "$/MWh")
    with summary_cols[5]:
        metric_card("Predicted peak LMP", predicted_peak, "$/MWh")

    replay_status_messages(
        metrics["mae"],
        metrics["bias"],
        float(sample["actual_lmp"].std()),
        actual_peak,
        predicted_peak,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sample["horizon"], y=sample["actual_lmp"], mode="lines+markers", name="Actual LMP"))
    fig.add_trace(go.Scatter(x=sample["horizon"], y=sample["predicted_lmp"], mode="lines+markers", name=pretty_model_name(selected_model)))
    fig.update_layout(template=plot_template(), title="24-hour actual vs predicted LMP curve", xaxis_title="Forecast horizon", yaxis_title="LMP ($/MWh)")
    st.plotly_chart(style_plotly_fig(fig), use_container_width=True)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        err_fig = px.bar(
            sample,
            x="horizon",
            y="error",
            template=plot_template(),
            title="Forecast error by horizon",
            labels={"error": "Predicted - actual"},
        )
        err_fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(style_plotly_fig(err_fig), use_container_width=True)
    with chart_cols[1]:
        sample["cumulative_abs_error"] = sample["abs_error"].cumsum()
        cum_fig = px.line(
            sample,
            x="horizon",
            y="cumulative_abs_error",
            markers=True,
            template=plot_template(),
            title="Cumulative absolute error by horizon",
        )
        st.plotly_chart(style_plotly_fig(cum_fig), use_container_width=True)

    replay_cols = [col for col in ["horizon", "target_ts", "actual_lmp", "predicted_lmp", "error", "abs_error"] if col in sample.columns]
    render_clean_table(sample[replay_cols], height=360)


def render_submission_information() -> None:
    st.header("Submission Information")
    st.markdown(
        """
        <div class="section-card">
        <strong>Repository:</strong> Impact-of-U.S.-summer-heatwaves-and-winter-blizzards-on-PJM-electricity-prices<br>
        <strong>App purpose:</strong> demonstrate a historical PJM electricity price forecasting workflow and
        event-focused model diagnostics.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Files read by the app")
        st.write(
            "- `reports/forecast_predictions_24h.csv`\n"
            "- `reports/forecast_metrics_summary.csv`\n"
            "- `reports/forecast_metrics_by_horizon.csv`\n"
            "- `reports/roc_auc_summary.csv`\n"
            "- `reports/roc_curve_points.csv`\n"
            "- `reports/event_forecast_metrics.csv`"
        )
        st.subheader("What the app demonstrates")
        st.write(
            "- 24-hour-ahead LMP forecast evaluation\n"
            "- Model comparison across MAE/RMSE/R2/Bias\n"
            "- Interactive forecast inspection\n"
            "- High-price detection via ROC/AUC\n"
            "- Event-specific performance under weather stress"
        )
    with col2:
        st.subheader("Limitations")
        st.write(
            "- This app is a historical dashboard and replay demo.\n"
            "- It does not yet ingest real-time PJM data or live weather forecasts.\n"
            "- A production version would need forecast-origin weather forecasts, scheduled data updates, saved model artifacts, and monitoring."
        )
        st.subheader("Future improvements")
        st.write(
            "- Add live PJM API or downloaded daily PJM data\n"
            "- Add weather forecast API\n"
            "- Save trained DMDc/residual models using joblib\n"
            "- Add a true 24-hour forecast interface\n"
            "- Add model monitoring and drift detection"
        )


forecast_predictions = load_csv("forecast_predictions_24h.csv")
forecast_metrics_summary = load_csv("forecast_metrics_summary.csv")
forecast_metrics_by_horizon = load_csv("forecast_metrics_by_horizon.csv")
roc_auc_summary = load_csv("roc_auc_summary.csv")
roc_curve_points = load_csv("roc_curve_points.csv")
event_forecast_metrics = load_csv("event_forecast_metrics.csv")

tabs = st.tabs(
    [
        "Project Overview",
        "Methodology",
        "Model Results",
        "Forecast Explorer",
        "Weather Event Lab",
        "Historical Forecast Replay Demo",
        "Submission Information",
    ]
)

with tabs[0]:
    render_project_overview(forecast_metrics_summary)

with tabs[1]:
    render_methodology(forecast_metrics_summary, forecast_predictions)

with tabs[2]:
    render_model_results(
        forecast_metrics_summary,
        forecast_metrics_by_horizon,
        roc_auc_summary,
        roc_curve_points,
    )

with tabs[3]:
    render_forecast_explorer(forecast_predictions)

with tabs[4]:
    render_weather_event_lab(event_forecast_metrics, forecast_predictions)

with tabs[5]:
    render_replay_demo(forecast_predictions)

with tabs[6]:
    render_submission_information()
