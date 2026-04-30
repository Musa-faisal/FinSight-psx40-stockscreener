import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.screener.engine import ScreenerEngine
from src.ingestion.kaggle_csv_loader import (
    load_combined_csv,
    load_per_ticker_csv,
    load_all_from_kaggle_dir,
    KAGGLE_DIR,
)
from src.ingestion.data_validator import validate_ohlcv
from src.database.db_manager import upsert_prices, db_summary
from config.settings import SMA_SHORT, SMA_LONG, SMA_200, RSI_PERIOD
from src.reports.stock_report import build_stock_report, report_to_dataframe, get_report_warning_messages
from src.portfolio.portfolio_builder import build_portfolio, summarize_portfolio
from src.portfolio.backtest import backtest_portfolio, backtest_to_dataframe

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PSX40 · Phase 10 Screener",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ═══════════════════════════════════════════════════════════════════

DARK_BG      = "#07090f"
PANEL_BG     = "#0d1117"
PANEL_BORDER = "#1c2333"
ACCENT       = "#00d4aa"
ACCENT2      = "#3b82f6"
PURPLE       = "#a78bfa"
RED          = "#f43f5e"
YELLOW       = "#fbbf24"
GREEN        = "#34d399"
TEXT_PRIMARY = "#e2e8f0"
TEXT_MUTED   = "#4b5563"
TEXT_DIM     = "#1e293b"

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d1117",
    font=dict(family="'IBM Plex Mono', monospace", color="#64748b", size=11),
    margin=dict(l=0, r=0, t=32, b=0),
    legend=dict(
        orientation="h", x=0, y=1.08,
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=10, color="#94a3b8"),
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#1c2333", bordercolor=ACCENT,
        font=dict(color="#e2e8f0", size=11),
    ),
)


def apply_plotly_layout(fig: go.Figure, height: int = 340, margin: dict | None = None, **kwargs) -> go.Figure:
    layout = dict(PLOTLY_BASE)
    layout["height"] = height
    if margin is not None:
        layout["margin"] = margin
    layout.update(kwargs)
    fig.update_layout(**layout)
    return fig


st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Syne:wght@400;600;700;800&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
    font-family: 'IBM Plex Mono', monospace;
}}
.main .block-container {{ padding: 1.5rem 2rem 3rem; max-width: 100%; }}

#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}

::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {DARK_BG}; }}
::-webkit-scrollbar-thumb {{ background: #1c2333; border-radius: 2px; }}
::-webkit-scrollbar-thumb:hover {{ background: {ACCENT}; }}

.page-header {{
    display: flex; align-items: baseline; gap: 0.75rem; margin-bottom: 0.2rem;
}}
.page-title {{
    font-family: 'Syne', sans-serif;
    font-size: 1.85rem; font-weight: 800;
    color: {TEXT_PRIMARY}; letter-spacing: -0.02em; line-height: 1;
}}
.page-badge {{
    font-size: 0.62rem; font-weight: 600; color: {ACCENT};
    letter-spacing: 0.25em; text-transform: uppercase;
    border: 1px solid {ACCENT}; padding: 2px 7px; border-radius: 3px;
}}
.page-sub {{
    font-size: 0.67rem; color: {TEXT_MUTED};
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 1.25rem;
}}
.disclaimer {{
    background: #1a0f00; border: 1px solid #92400e44;
    border-left: 3px solid {YELLOW}; border-radius: 4px;
    padding: 0.45rem 1rem; font-size: 0.67rem; color: #92400e;
    margin-bottom: 1.25rem; letter-spacing: 0.03em;
}}

.kpi-row {{
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 0.65rem; margin-bottom: 1.5rem;
}}
.kpi {{
    background: {PANEL_BG}; border: 1px solid {PANEL_BORDER};
    border-radius: 6px; padding: 0.9rem 1.1rem;
    position: relative; overflow: hidden; transition: border-color 0.2s;
}}
.kpi::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; background: var(--c, {ACCENT});
}}
.kpi:hover {{ border-color: var(--c, {ACCENT}); }}
.kpi-lbl {{
    font-size: 0.58rem; color: {TEXT_MUTED};
    text-transform: uppercase; letter-spacing: 0.18em; margin-bottom: 0.45rem;
}}
.kpi-val {{
    font-family: 'Syne', sans-serif; font-size: 1.55rem;
    font-weight: 700; color: var(--c, {ACCENT}); line-height: 1;
}}
.kpi-hint {{ font-size: 0.6rem; color: {TEXT_MUTED}; margin-top: 0.2rem; }}

.sec {{
    font-size: 0.58rem; text-transform: uppercase;
    letter-spacing: 0.22em; color: {TEXT_MUTED};
    border-bottom: 1px solid {PANEL_BORDER};
    padding-bottom: 0.35rem; margin: 1.25rem 0 0.65rem;
}}

section[data-testid="stSidebar"] {{
    background: #090c14; border-right: 1px solid {PANEL_BORDER};
}}
section[data-testid="stSidebar"] .block-container {{ padding: 1.25rem 1rem; }}
.sb-brand {{
    font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700;
    color: {ACCENT}; letter-spacing: 0.05em; margin-bottom: 0.1rem;
}}
.sb-sec {{
    font-size: 0.57rem; text-transform: uppercase;
    letter-spacing: 0.2em; color: {TEXT_MUTED}; margin: 1rem 0 0.35rem;
}}

div[data-testid="stSelectbox"] > div,
div[data-testid="stMultiSelect"] > div {{
    background: {PANEL_BG} !important;
    border-color: {PANEL_BORDER} !important;
    color: {TEXT_PRIMARY} !important;
    font-size: 0.76rem !important;
}}
div[data-testid="stDataFrame"] {{
    border: 1px solid {PANEL_BORDER}; border-radius: 6px; overflow: hidden;
}}
div[data-testid="stTabs"] [data-testid="stTab"] {{
    font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase;
}}
button[kind="primary"] {{
    background: {ACCENT} !important; color: #000 !important;
    font-weight: 600 !important; font-size: 0.72rem !important;
    border: none !important; border-radius: 4px !important;
}}
div[data-testid="stMetric"] {{
    background: {PANEL_BG}; border: 1px solid {PANEL_BORDER};
    border-radius: 6px; padding: 0.65rem 0.85rem;
}}
div[data-testid="stMetric"] label {{
    font-size: 0.58rem !important; color: {TEXT_MUTED} !important;
    text-transform: uppercase; letter-spacing: 0.15em;
}}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    font-family: 'Syne', sans-serif; font-size: 1.2rem !important;
    color: {ACCENT} !important;
}}
.pill {{
    display: inline-block; padding: 2px 8px; border-radius: 3px;
    font-size: 0.6rem; font-weight: 600; letter-spacing: 0.06em;
}}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# ENGINE
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="⬡  Initialising engine...")
def load_engine() -> ScreenerEngine:
    e = ScreenerEngine()
    e.run()
    return e


def reload_engine() -> ScreenerEngine:
    st.cache_resource.clear()
    e = ScreenerEngine()
    e.run(force_reseed=False)
    return e


engine      = load_engine()
screener_df = engine.get_screener_df()


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

VERDICT_COLORS = {
    "Strong Buy": ACCENT,
    "Buy":        ACCENT2,
    "Hold":       YELLOW,
    "Weak":       "#f97316",
    "Avoid":      RED,
}


def vc(label: str) -> str:
    return VERDICT_COLORS.get(label, TEXT_MUTED)


def pct(val) -> str:
    try:
        return f"{float(val)*100:+.1f}%"
    except Exception:
        return "—"


def pct_no_sign(val) -> str:
    try:
        return f"{float(val)*100:.1f}%"
    except Exception:
        return "—"


def fmt(val, decimals=2, suffix=""):
    try:
        return f"{float(val):.{decimals}f}{suffix}"
    except Exception:
        return "—"


def safe_float(val, decimals=2):
    try:
        return round(float(val), decimals)
    except Exception:
        return None


def plotly_theme(fig: go.Figure, height: int = 340) -> go.Figure:
    apply_plotly_layout(fig, height=height)
    fig.update_xaxes(showgrid=False, zeroline=False,
                     tickfont=dict(size=10, color="#475569"), linecolor="#1c2333")
    fig.update_yaxes(gridcolor="#1c2333", zeroline=False,
                     tickfont=dict(size=10, color="#475569"), linecolor="#1c2333")
    return fig


def kpi_card(label, value, hint="", color=ACCENT) -> str:
    return (
        f'<div class="kpi" style="--c:{color}">'
        f'<div class="kpi-lbl">{label}</div>'
        f'<div class="kpi-val">{value}</div>'
        f'<div class="kpi-hint">{hint}</div>'
        f'</div>'
    )


def _pct_fmt(val, signed=True) -> str:
    """Format a decimal as percentage string."""
    try:
        f = float(val)
        return f"{f*100:+.2f}%" if signed else f"{f*100:.2f}%"
    except Exception:
        return "N/A"


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sb-brand">⬡ PSX40</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.57rem;color:#374151;letter-spacing:0.14em;'
        'text-transform:uppercase;margin-bottom:0.75rem;">Phase 10 · Quant Screener</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown('<div class="sb-sec">Preset Screener</div>', unsafe_allow_html=True)
    preset_options  = ["Custom Filters"] + engine.get_preset_names()
    selected_preset = st.selectbox(
        "Preset", options=preset_options, index=0, label_visibility="collapsed",
    )
    st.divider()

    st.markdown('<div class="sb-sec">Sector</div>', unsafe_allow_html=True)
    all_sectors = engine.get_sectors() if hasattr(engine, "get_sectors") else []
    sel_sectors = st.multiselect(
        "Sector", options=all_sectors, default=[],
        placeholder="All sectors", label_visibility="collapsed",
    )

    st.markdown('<div class="sb-sec">Final Verdict</div>', unsafe_allow_html=True)
    verdict_opts = ["Strong Buy", "Buy", "Hold", "Weak", "Avoid"]
    sel_verdicts = st.multiselect(
        "Verdict", options=verdict_opts, default=[],
        placeholder="All verdicts", label_visibility="collapsed",
    )

    st.markdown('<div class="sb-sec">Composite Score</div>', unsafe_allow_html=True)
    score_range = st.slider("Score", 0, 100, (0, 100), step=5, label_visibility="collapsed")

    st.markdown('<div class="sb-sec">Max Risk Score</div>', unsafe_allow_html=True)
    max_risk = st.slider("Risk", 0, 100, 100, step=5, label_visibility="collapsed")

    st.markdown('<div class="sb-sec">Min Dividend Yield (%)</div>', unsafe_allow_html=True)
    min_div = st.slider("Div Yield", 0.0, 15.0, 0.0, step=0.5, label_visibility="collapsed")

    st.markdown('<div class="sb-sec">Max P/E Ratio</div>', unsafe_allow_html=True)
    max_pe = st.number_input("Max PE", min_value=0.0, max_value=500.0,
                              value=500.0, step=5.0, label_visibility="collapsed")

    st.divider()
    st.markdown('<div class="sb-sec">Detail Ticker</div>', unsafe_allow_html=True)
    chart_tickers = engine.get_tickers() if hasattr(engine, "get_tickers") else []
    chart_ticker  = st.selectbox(
        "Chart", options=chart_tickers, index=0 if chart_tickers else None,
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(
        '<div style="font-size:0.57rem;color:#1f2937;line-height:1.7;">'
        'Data may be synthetic.<br>Not financial advice.</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
# FILTER LOGIC
# ═══════════════════════════════════════════════════════════════════

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    if sel_sectors:
        out = out[out["sector"].isin(sel_sectors)]

    if sel_verdicts:
        vcol = "final_verdict" if "final_verdict" in out.columns else "verdict"
        out  = out[out[vcol].isin(sel_verdicts)]

    if "composite_score" in out.columns:
        out = out[out["composite_score"].between(*score_range, inclusive="both")]

    if "risk_score" in out.columns:
        out = out[out["risk_score"].fillna(0) <= max_risk]

    if "dividend_yield" in out.columns and min_div > 0:
        out = out[out["dividend_yield"].fillna(0) >= min_div / 100]

    if "pe_ratio" in out.columns and max_pe < 500:
        out = out[out["pe_ratio"].isna() | (out["pe_ratio"] <= max_pe)]

    return out.reset_index(drop=True)


if selected_preset == "Custom Filters":
    base_df     = screener_df
    preset_empty = False
else:
    base_df      = engine.get_preset(selected_preset)
    preset_empty = base_df.empty

filtered_df = apply_filters(base_df)
verdict_col = "final_verdict" if "final_verdict" in screener_df.columns else "verdict"


# ═══════════════════════════════════════════════════════════════════
# PAGE HEADER
# ═══════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="page-header">'
    '<span class="page-title">PSX40 Stock Screener</span>'
    '<span class="page-badge">Phase 10</span>'
    '</div>'
    '<div class="page-sub">Pakistan Stock Exchange · Quant Factor Model · '
    'Technical + Fundamental + Sector + Portfolio</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="disclaimer">⚠ Data may be synthetic or CSV-imported. '
    'For demonstration only — not financial advice.</div>',
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════
# KPI CARDS
# ═══════════════════════════════════════════════════════════════════

if not screener_df.empty:
    total_stocks = len(screener_df)
    avg_comp     = round(screener_df["composite_score"].mean(), 1) \
                   if "composite_score" in screener_df.columns else "—"

    if "composite_score" in screener_df.columns:
        top_idx = screener_df["composite_score"].idxmax()
        top_tk  = screener_df.loc[top_idx, "ticker"]
        top_sc  = round(screener_df.loc[top_idx, "composite_score"], 1)
    else:
        top_tk, top_sc = "—", "—"

    avg_risk   = round(screener_df["risk_score"].mean(), 1) \
                 if "risk_score" in screener_df.columns else "—"
    fund_cols  = ["pe_ratio", "pb_ratio", "roe", "debt_to_equity", "dividend_yield"]
    fund_cols  = [c for c in fund_cols if c in screener_df.columns]
    complete_f = int((screener_df[fund_cols].notna().all(axis=1)).sum()) if fund_cols else 0
else:
    total_stocks = avg_comp = top_tk = top_sc = avg_risk = complete_f = "—"

st.markdown(
    '<div class="kpi-row">'
    + kpi_card("Total Stocks",   str(total_stocks), "in universe",           ACCENT)
    + kpi_card("Avg Composite",  str(avg_comp),     "composite score",       ACCENT2)
    + kpi_card("Top by Score",   str(top_tk),       f"score {top_sc}",       PURPLE)
    + kpi_card("Avg Risk Score", str(avg_risk),     "lower = less risky",    GREEN)
    + kpi_card("Complete Funds", str(complete_f),   "full fundamental data", YELLOW)
    + '</div>',
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════

tab_screen, tab_sector, tab_detail, tab_portfolio, tab_import = st.tabs([
    "⬡  Screener", "◫  Sector Benchmark", "▣  Ticker Detail",
    "▦  Portfolio", "⊕  Import",
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — SCREENER TABLE
# ═══════════════════════════════════════════════════════════════════

with tab_screen:

    title_text = (
        f"Screener Results — {len(filtered_df)} stocks matched"
        if selected_preset == "Custom Filters"
        else f"Preset: {selected_preset} — {len(filtered_df)} stocks matched"
    )
    st.markdown(f'<div class="sec">{title_text}</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.info(
            "No stocks currently match this preset."
            if (selected_preset != "Custom Filters" and preset_empty)
            else "No stocks match the current filters."
        )
    else:
        want_cols = [
            "ticker", "name", "sector", "latest_close",
            "technical_score", "fundamental_score", "risk_score", "composite_score",
            verdict_col,
            "pe_ratio", "pb_ratio", "roe", "debt_to_equity", "dividend_yield",
            "pe_vs_sector", "pb_vs_sector", "roe_vs_sector",
            "dividend_yield_vs_sector", "composite_score_vs_sector", "sector_value_label",
        ]
        show_cols   = [c for c in want_cols if c in filtered_df.columns]
        display_tbl = filtered_df[show_cols].copy()

        col_cfg = {
            "latest_close":      st.column_config.NumberColumn("Close ₨",    format="₨%.2f"),
            "technical_score":   st.column_config.ProgressColumn("Tech",      min_value=0, max_value=100, format="%.1f"),
            "fundamental_score": st.column_config.ProgressColumn("Fund",      min_value=0, max_value=100, format="%.1f"),
            "risk_score":        st.column_config.ProgressColumn("Risk",      min_value=0, max_value=100, format="%.1f"),
            "composite_score":   st.column_config.ProgressColumn("Composite", min_value=0, max_value=100, format="%.1f"),
            verdict_col:         st.column_config.TextColumn("Verdict"),
            "pe_ratio":          st.column_config.NumberColumn("P/E",         format="%.1f"),
            "pb_ratio":          st.column_config.NumberColumn("P/B",         format="%.2f"),
            "roe":               st.column_config.NumberColumn("ROE",         format="%.1%%"),
            "debt_to_equity":    st.column_config.NumberColumn("D/E",         format="%.2f"),
            "dividend_yield":    st.column_config.NumberColumn("Div Yield",   format="%.2%%"),
            "pe_vs_sector":      st.column_config.NumberColumn("P/E vs Sect", format="%+.1%%"),
            "pb_vs_sector":      st.column_config.NumberColumn("P/B vs Sect", format="%+.1%%"),
            "roe_vs_sector":     st.column_config.NumberColumn("ROE vs Sect", format="%+.1%%"),
            "dividend_yield_vs_sector": st.column_config.NumberColumn("Div vs Sect", format="%+.1%%"),
            "composite_score_vs_sector": st.column_config.NumberColumn("Score vs Sect", format="%+.1%%"),
            "sector_value_label": st.column_config.TextColumn("Sector Label"),
        }

        st.dataframe(display_tbl, use_container_width=True, hide_index=True,
                     height=440, column_config=col_cfg)

        csv_bytes = display_tbl.to_csv(index=False).encode("utf-8")
        st.download_button("⬇  Download CSV", data=csv_bytes,
                           file_name="psx40_screener.csv", mime="text/csv")

        st.markdown('<div class="sec">Verdict Distribution</div>', unsafe_allow_html=True)
        if verdict_col in filtered_df.columns:
            vdist = (
                filtered_df[verdict_col]
                .value_counts()
                .reindex(verdict_opts, fill_value=0)
                .reset_index()
            )
            vdist.columns = ["Verdict", "Count"]
            colors = [vc(v) for v in vdist["Verdict"]]
            fig_vd = go.Figure(go.Bar(
                x=vdist["Verdict"], y=vdist["Count"],
                marker_color=colors, marker_line_width=0,
                text=vdist["Count"], textposition="outside",
                textfont=dict(size=11, color="#94a3b8"),
            ))
            apply_plotly_layout(fig_vd, height=220, showlegend=False, bargap=0.35,
                                margin=dict(l=0, r=0, t=20, b=0))
            fig_vd.update_xaxes(showgrid=False)
            fig_vd.update_yaxes(gridcolor="#1c2333")
            st.plotly_chart(fig_vd, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — SECTOR BENCHMARK
# ═══════════════════════════════════════════════════════════════════

with tab_sector:

    st.markdown('<div class="sec">Sector Benchmark Summary</div>', unsafe_allow_html=True)

    sector_df = engine.get_sector_benchmarks() \
        if hasattr(engine, "get_sector_benchmarks") else pd.DataFrame()

    if sector_df.empty:
        st.info("No sector benchmark data available yet.")
    else:
        sector_show_cols = [
            "sector", "sector_stock_count", "sector_avg_pe", "sector_median_pe",
            "sector_avg_pb", "sector_avg_roe", "sector_avg_dividend_yield",
            "sector_avg_composite_score", "sector_avg_risk_score",
        ]
        sector_show_cols = [c for c in sector_show_cols if c in sector_df.columns]
        sector_tbl = sector_df[sector_show_cols].copy()

        st.dataframe(
            sector_tbl, use_container_width=True, hide_index=True, height=320,
            column_config={
                "sector":                     st.column_config.TextColumn("Sector"),
                "sector_stock_count":         st.column_config.NumberColumn("Stocks",       format="%d"),
                "sector_avg_pe":              st.column_config.NumberColumn("Avg P/E",      format="%.2f"),
                "sector_median_pe":           st.column_config.NumberColumn("Median P/E",   format="%.2f"),
                "sector_avg_pb":              st.column_config.NumberColumn("Avg P/B",      format="%.2f"),
                "sector_avg_roe":             st.column_config.NumberColumn("Avg ROE",      format="%.2%%"),
                "sector_avg_dividend_yield":  st.column_config.NumberColumn("Avg Div Yield", format="%.2%%"),
                "sector_avg_composite_score": st.column_config.NumberColumn("Avg Composite", format="%.1f"),
                "sector_avg_risk_score":      st.column_config.NumberColumn("Avg Risk",     format="%.1f"),
            },
        )

        st.markdown('<div class="sec">Sector Average Composite Score</div>', unsafe_allow_html=True)
        if "sector_avg_composite_score" in sector_df.columns:
            chart_df = sector_df.dropna(subset=["sector_avg_composite_score"]).copy()
            if not chart_df.empty:
                chart_df = chart_df.sort_values("sector_avg_composite_score", ascending=False)
                fig_sec = go.Figure(go.Bar(
                    x=chart_df["sector"],
                    y=chart_df["sector_avg_composite_score"],
                    marker_color=ACCENT2, marker_line_width=0,
                    text=chart_df["sector_avg_composite_score"].round(1),
                    textposition="outside",
                    textfont=dict(size=10, color="#94a3b8"),
                ))
                apply_plotly_layout(fig_sec, height=260, showlegend=False,
                                    margin=dict(l=0, r=20, t=20, b=0))
                fig_sec.update_xaxes(showgrid=False)
                fig_sec.update_yaxes(range=[0, 110], gridcolor="#1c2333")
                st.plotly_chart(fig_sec, use_container_width=True)

        st.markdown('<div class="sec">How to Read Sector Labels</div>', unsafe_allow_html=True)
        st.markdown("""
- **Sector Value Leader**: stronger combined relative value/profitability/yield/score than peers.
- **Attractive vs Sector**: modestly better than sector average.
- **In Line with Sector**: broadly near sector averages.
- **Weak vs Sector / Expensive vs Sector**: valuation or quality weaker than peers.
- **Insufficient sector data**: fewer than 2 valid peer rows for safe benchmarking.
        """)


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — TICKER DETAIL
# ═══════════════════════════════════════════════════════════════════

with tab_detail:

    if not chart_ticker:
        st.info("Select a ticker from the sidebar.")
    else:
        st.markdown(f'<div class="sec">{chart_ticker} — Ticker Detail</div>',
                    unsafe_allow_html=True)

        price_df = engine.get_price_df(chart_ticker) \
            if hasattr(engine, "get_price_df") else pd.DataFrame()
        row_df   = screener_df[screener_df["ticker"] == chart_ticker]
        r        = row_df.iloc[0] if not row_df.empty else None

        if r is not None:
            comp   = safe_float(r.get("composite_score", 0)) or 0
            tech   = safe_float(r.get("technical_score",  0)) or 0
            fund   = safe_float(r.get("fundamental_score", 0)) or 0
            risk   = safe_float(r.get("risk_score",  0)) or 0
            verd   = r.get(verdict_col, "Hold")
            b1, b2, b3, b4, b5 = st.columns(5)
            with b1: st.metric("Composite",  f"{comp:.1f}")
            with b2: st.metric("Technical",  f"{tech:.1f}")
            with b3: st.metric("Fundamental", f"{fund:.1f}")
            with b4: st.metric("Risk",        f"{risk:.1f}")
            with b5: st.metric("Verdict",     verd)

        if price_df.empty:
            st.warning(f"No price history for {chart_ticker}.")
        else:
            col_price, col_rsi = st.columns([3, 1])

            with col_price:
                fig_c = go.Figure()
                fig_c.add_trace(go.Candlestick(
                    x=price_df["date"],
                    open=price_df["open"], high=price_df["high"],
                    low=price_df["low"],   close=price_df["close"],
                    name=chart_ticker,
                    increasing=dict(line=dict(color=ACCENT, width=1),
                                    fillcolor="rgba(0,212,170,0.33)"),
                    decreasing=dict(line=dict(color=RED, width=1),
                                    fillcolor="rgba(244,63,94,0.33)"),
                ))
                for sma_w, color, dash in [
                    (SMA_SHORT, YELLOW, "solid"),
                    (SMA_LONG,  PURPLE, "dot"),
                    (SMA_200,   RED,    "dash"),
                ]:
                    col_n = f"sma_{sma_w}"
                    if col_n in price_df.columns:
                        fig_c.add_trace(go.Scatter(
                            x=price_df["date"], y=price_df[col_n],
                            name=f"SMA {sma_w}",
                            line=dict(color=color, width=1.2, dash=dash),
                        ))
                apply_plotly_layout(
                    fig_c, height=320,
                    title=dict(text=f"Price + SMAs — {chart_ticker}",
                               font=dict(size=11, color="#64748b")),
                    xaxis_rangeslider_visible=False,
                )
                st.plotly_chart(fig_c, use_container_width=True)

            with col_rsi:
                rsi_col_n = f"rsi_{RSI_PERIOD}"
                if rsi_col_n in price_df.columns:
                    latest_rsi = (price_df[rsi_col_n].dropna().iloc[-1]
                                  if not price_df[rsi_col_n].dropna().empty else 50)
                    rsi_color  = RED if latest_rsi > 70 else ACCENT if latest_rsi < 30 else YELLOW
                    fig_gauge  = go.Figure(go.Indicator(
                        mode="gauge+number", value=latest_rsi,
                        title=dict(text=f"RSI {RSI_PERIOD}",
                                   font=dict(color="#64748b", size=11)),
                        number=dict(font=dict(color=rsi_color, size=26,
                                              family="IBM Plex Mono")),
                        gauge=dict(
                            axis=dict(range=[0, 100],
                                      tickfont=dict(color="#475569", size=9)),
                            bar=dict(color=rsi_color, thickness=0.18),
                            bgcolor="#0d1117", bordercolor="#1c2333",
                            steps=[
                                dict(range=[0,  30], color="rgba(0,212,170,0.07)"),
                                dict(range=[30, 70], color="rgba(28,35,51,0.27)"),
                                dict(range=[70,100], color="rgba(244,63,94,0.07)"),
                            ],
                            threshold=dict(line=dict(color="#475569", width=1),
                                           thickness=0.6, value=50),
                        ),
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                        height=170, margin=dict(l=10, r=10, t=40, b=10),
                        font=dict(family="IBM Plex Mono", color="#64748b"),
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    rsi_lbl = ("🔴 Overbought" if latest_rsi > 70
                               else "🟢 Oversold" if latest_rsi < 30 else "🟡 Neutral")
                    st.markdown(
                        f'<div style="text-align:center;font-size:0.7rem;'
                        f'color:{rsi_color};letter-spacing:0.1em;">{rsi_lbl}</div>',
                        unsafe_allow_html=True,
                    )

            rsi_col_n = f"rsi_{RSI_PERIOD}"
            if rsi_col_n in price_df.columns:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=price_df["date"], y=price_df[rsi_col_n],
                    line=dict(color=ACCENT2, width=1.5),
                    fill="tozeroy", fillcolor="rgba(59,130,246,0.06)",
                    name=f"RSI {RSI_PERIOD}",
                ))
                for lvl, col, lbl in [(70, RED, "OB"), (50, "#374151", "Mid"), (30, ACCENT, "OS")]:
                    fig_rsi.add_hline(y=lvl, line_dash="dot", line_color=col, line_width=1,
                                      annotation_text=lbl,
                                      annotation_font=dict(size=9, color=col))
                apply_plotly_layout(
                    fig_rsi, height=180,
                    title=dict(text=f"RSI {RSI_PERIOD}",
                               font=dict(size=11, color="#64748b")),
                )
                fig_rsi.update_yaxes(range=[0, 100], gridcolor="#1c2333")
                st.plotly_chart(fig_rsi, use_container_width=True)

        if r is not None:
            st.markdown('<div class="sec">Fundamental Ratio Summary</div>',
                        unsafe_allow_html=True)
            f1, f2, f3, f4, f5 = st.columns(5)
            with f1:
                pe = r.get("pe_ratio"); st.metric("P/E Ratio", fmt(pe, 2) if pe else "—")
            with f2:
                pb = r.get("pb_ratio"); st.metric("P/B Ratio", fmt(pb, 2) if pb else "—")
            with f3:
                roe = r.get("roe"); st.metric("ROE", pct(roe) if roe else "—")
            with f4:
                de = r.get("debt_to_equity"); st.metric("D/E Ratio", fmt(de, 2) if de else "—")
            with f5:
                dy = r.get("dividend_yield"); st.metric("Div Yield", pct(dy) if dy else "—")

            st.markdown('<div class="sec">Sector Benchmark Snapshot</div>',
                        unsafe_allow_html=True)
            s1, s2, s3, s4, s5 = st.columns(5)
            with s1: st.metric("P/E vs Sector",   pct(r.get("pe_vs_sector")))
            with s2: st.metric("P/B vs Sector",   pct(r.get("pb_vs_sector")))
            with s3: st.metric("ROE vs Sector",   pct(r.get("roe_vs_sector")))
            with s4: st.metric("Score vs Sector", pct(r.get("composite_score_vs_sector")))
            with s5: st.metric("Sector Label",    str(r.get("sector_value_label", "—")))

            st.markdown('<div class="sec">Score Breakdown</div>', unsafe_allow_html=True)
            group_labels = ["Technical", "Fundamental", "Risk (inv)"]
            group_scores = [
                safe_float(r.get("technical_score",   0)) or 0,
                safe_float(r.get("fundamental_score", 0)) or 0,
                100 - (safe_float(r.get("risk_score", 0)) or 0),
            ]
            col_r, col_b = st.columns([1, 1])
            with col_r:
                fig_rad = go.Figure(go.Scatterpolar(
                    r=group_scores + [group_scores[0]],
                    theta=group_labels + [group_labels[0]],
                    fill="toself",
                    fillcolor="rgba(0,212,170,0.09)",
                    line=dict(color=ACCENT, width=2),
                    marker=dict(color=ACCENT, size=5),
                ))
                fig_rad.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    polar=dict(
                        bgcolor="#0d1117",
                        radialaxis=dict(visible=True, range=[0, 100],
                                        tickfont=dict(size=8, color="#475569"),
                                        gridcolor="#1c2333", linecolor="#1c2333"),
                        angularaxis=dict(tickfont=dict(size=10, color="#94a3b8"),
                                         linecolor="#1c2333", gridcolor="#1c2333"),
                    ),
                    height=260, margin=dict(l=30, r=30, t=30, b=30),
                    showlegend=False, font=dict(family="IBM Plex Mono"),
                )
                st.plotly_chart(fig_rad, use_container_width=True)
            with col_b:
                fig_bar = go.Figure(go.Bar(
                    x=group_labels, y=group_scores,
                    marker_color=[ACCENT2, PURPLE, GREEN], marker_line_width=0,
                    text=[f"{s:.1f}" for s in group_scores],
                    textposition="outside", textfont=dict(size=10, color="#94a3b8"),
                ))
                fig_bar.add_hline(y=50, line_dash="dot", line_color="#374151",
                                  line_width=1, annotation_text="Neutral",
                                  annotation_font=dict(size=9, color="#374151"))
                apply_plotly_layout(fig_bar, height=260, showlegend=False,
                                    bargap=0.35, margin=dict(l=0, r=40, t=32, b=0))
                fig_bar.update_yaxes(range=[0, 120], gridcolor="#1c2333")
                st.plotly_chart(fig_bar, use_container_width=True)

        # ── Analyst Report ───────────────────────────────────────────
        st.markdown('<div class="sec">Analyst Report</div>', unsafe_allow_html=True)
        try:
            report   = build_stock_report(screener_df, chart_ticker)
            id_info  = report["identity"]
            verdict_info = report["verdict"]

            r1, r2, r3, r4 = st.columns(4)
            with r1: st.metric("Company", id_info.get("name", chart_ticker))
            with r2: st.metric("Sector",  id_info.get("sector", "—"))
            with r3:
                close_val = id_info.get("latest_close")
                st.metric("Latest Close", f"₨{close_val:,.2f}" if close_val else "—")
            with r4:
                emoji = verdict_info.get("verdict_emoji", "")
                label = verdict_info.get("final_verdict", "—")
                st.metric("Verdict", f"{emoji} {label}" if emoji else label)

            summary = report.get("summary", "")
            if summary:
                st.markdown(
                    f'<div style="background:{PANEL_BG};border:1px solid {PANEL_BORDER};'
                    f'border-radius:6px;padding:1rem 1.25rem;margin:0.75rem 0;'
                    f'font-size:0.82rem;line-height:1.7;color:{TEXT_PRIMARY};">'
                    f'{summary}</div>',
                    unsafe_allow_html=True,
                )

            for w in report.get("warnings", []):
                st.warning(w)

            st.markdown(
                f'<div style="font-size:0.58rem;text-transform:uppercase;'
                f'letter-spacing:0.18em;color:{TEXT_MUTED};margin:0.5rem 0 0.3rem;">'
                'Scores</div>', unsafe_allow_html=True,
            )
            score_data = report.get("scores", {})
            s1, s2, s3, s4 = st.columns(4)
            with s1: st.metric("Technical",   f"{score_data.get('technical_score','—')}")
            with s2: st.metric("Fundamental", f"{score_data.get('fundamental_score','—')}")
            with s3: st.metric("Risk",        f"{score_data.get('risk_score','—')}")
            with s4: st.metric("Composite",   f"{score_data.get('composite_score','—')}")

            # Fundamentals
            st.markdown(
                f'<div style="font-size:0.58rem;text-transform:uppercase;'
                f'letter-spacing:0.18em;color:{TEXT_MUTED};margin:0.75rem 0 0.3rem;">'
                'Fundamental Ratios</div>', unsafe_allow_html=True,
            )
            fund_data = report.get("fundamentals", {})
            fund_rows = []
            for k, v in fund_data.items():
                if v is not None:
                    if k in ("roe", "net_profit_margin", "dividend_yield", "payout_ratio"):
                        dv = f"{v*100:.2f}%" if isinstance(v, float) else str(v)
                    else:
                        dv = f"{v:.2f}" if isinstance(v, float) else str(v)
                else:
                    dv = "N/A"
                fund_rows.append({"Metric": k.replace("_", " ").title(), "Value": dv})
            if fund_rows:
                st.dataframe(pd.DataFrame(fund_rows), use_container_width=True,
                             hide_index=True, height=260)

            # Sector comparison
            st.markdown(
                f'<div style="font-size:0.58rem;text-transform:uppercase;'
                f'letter-spacing:0.18em;color:{TEXT_MUTED};margin:0.75rem 0 0.3rem;">'
                'Sector Comparison</div>', unsafe_allow_html=True,
            )
            sec_data = report.get("sector_comparison", {})
            sec_rows = []
            for k, v in sec_data.items():
                if k == "sector_value_label":
                    continue
                if v is not None:
                    dv = (f"{v*100:+.1f}%" if "vs_sector" in k
                          else f"{v*100:.2f}%" if k in ("sector_avg_roe","sector_avg_dividend_yield")
                          else f"{v:.2f}")
                else:
                    dv = "N/A"
                sec_rows.append({"Metric": k.replace("_", " ").title(), "Value": dv})
            if sec_rows:
                st.dataframe(pd.DataFrame(sec_rows), use_container_width=True,
                             hide_index=True, height=320)
            label = sec_data.get("sector_value_label")
            if label:
                st.markdown(
                    f'<div style="font-size:0.72rem;color:{ACCENT};margin-top:0.3rem;">'
                    f'Sector Label: <b>{label}</b></div>', unsafe_allow_html=True,
                )

            # Technical snapshot
            st.markdown(
                f'<div style="font-size:0.58rem;text-transform:uppercase;'
                f'letter-spacing:0.18em;color:{TEXT_MUTED};margin:0.75rem 0 0.3rem;">'
                'Technical Snapshot</div>', unsafe_allow_html=True,
            )
            tech_data = report.get("technicals", {})
            tech_rows = []
            for k, v in tech_data.items():
                if v is not None:
                    if k in ("return_1m", "return_3m", "return_6m"):
                        dv = f"{v*100:+.2f}%"
                    elif k == "rsi_14":
                        dv = f"{v:.1f}"
                    elif k in ("macd_hist","breakout_ratio","volatility_30d",
                                "volatility","volume_surge","volume_ratio"):
                        dv = f"{v:.3f}"
                    elif k == "avg_volume_20d":
                        dv = f"{int(v):,}"
                    else:
                        dv = f"{v:.2f}"
                else:
                    dv = "N/A"
                tech_rows.append({"Metric": k.replace("_", " ").upper(), "Value": dv})
            if tech_rows:
                st.dataframe(pd.DataFrame(tech_rows), use_container_width=True,
                             hide_index=True, height=360)

            rationale = verdict_info.get("verdict_rationale")
            if rationale:
                st.markdown(
                    f'<div style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:0.5rem;">'
                    f'<b>Rationale:</b> {rationale}</div>', unsafe_allow_html=True,
                )

        except Exception as exc:
            st.warning(f"Could not generate analyst report for {chart_ticker}: {exc}")


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — PORTFOLIO BUILDER  (Phase 10)
# ═══════════════════════════════════════════════════════════════════

with tab_portfolio:

    st.markdown('<div class="sec">Portfolio Builder</div>', unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────────
    pc1, pc2, pc3 = st.columns(3)

    with pc1:
        port_top_n = st.slider(
            "Number of holdings", min_value=3, max_value=30,
            value=10, step=1,
        )
        port_method = st.selectbox(
            "Weighting method",
            options=["Equal Weight", "Inverse Volatility"],
            index=0,
        )

    with pc2:
        port_min_score = st.number_input(
            "Min composite score", min_value=0.0, max_value=100.0,
            value=50.0, step=5.0,
        )
        port_max_risk = st.number_input(
            "Max risk score", min_value=0.0, max_value=100.0,
            value=70.0, step=5.0,
        )

    with pc3:
        port_sectors = st.multiselect(
            "Restrict to sectors",
            options=all_sectors,
            default=[],
            placeholder="All sectors",
        )

    weighting_key = "equal" if port_method == "Equal Weight" else "inverse_volatility"

    # ── Build portfolio ───────────────────────────────────────────
    try:
        portfolio_df = build_portfolio(
            screener_df,
            top_n=port_top_n,
            weighting=weighting_key,
            min_composite=float(port_min_score),
            max_risk=float(port_max_risk),
            sectors=port_sectors if port_sectors else None,
        )
    except Exception as exc:
        st.error(f"Portfolio build failed: {exc}")
        portfolio_df = pd.DataFrame()

    if portfolio_df.empty:
        st.info(
            "No stocks matched the current portfolio filters. "
            "Try lowering the minimum composite score or increasing the max risk score."
        )
    else:
        # ── Portfolio table ───────────────────────────────────────
        st.markdown('<div class="sec">Portfolio Holdings</div>', unsafe_allow_html=True)

        port_display_cols = [
            "ticker", "name", "sector", "weight",
            "composite_score", "risk_score", "dividend_yield",
        ]
        verd_col_p = "final_verdict" if "final_verdict" in portfolio_df.columns else "verdict"
        if verd_col_p in portfolio_df.columns:
            port_display_cols.append(verd_col_p)

        port_display_cols = [c for c in port_display_cols if c in portfolio_df.columns]
        port_tbl = portfolio_df[port_display_cols].copy()

        st.dataframe(
            port_tbl,
            use_container_width=True,
            hide_index=True,
            height=min(80 + len(port_tbl) * 38, 480),
            column_config={
                "weight":          st.column_config.NumberColumn("Weight",    format="%.2%%"),
                "composite_score": st.column_config.ProgressColumn("Composite", min_value=0, max_value=100, format="%.1f"),
                "risk_score":      st.column_config.ProgressColumn("Risk",    min_value=0, max_value=100, format="%.1f"),
                "dividend_yield":  st.column_config.NumberColumn("Div Yield", format="%.2%%"),
                verd_col_p:        st.column_config.TextColumn("Verdict"),
            },
        )

        # ── Portfolio summary metrics ─────────────────────────────
        st.markdown('<div class="sec">Portfolio Summary</div>', unsafe_allow_html=True)

        summary = summarize_portfolio(portfolio_df)

        pm1, pm2, pm3, pm4, pm5, pm6 = st.columns(6)
        with pm1:
            st.metric("Holdings", str(summary.get("holdings", "—")))
        with pm2:
            st.metric("Weight Sum",
                      f"{summary.get('weight_sum', 0):.4f}")
        with pm3:
            avg_c = summary.get("avg_composite_score")
            st.metric("Avg Composite", f"{avg_c:.1f}" if avg_c else "—")
        with pm4:
            avg_r = summary.get("avg_risk_score")
            st.metric("Avg Risk", f"{avg_r:.1f}" if avg_r else "—")
        with pm5:
            avg_d = summary.get("avg_dividend_yield")
            st.metric("Avg Div Yield",
                      f"{avg_d:.2f}%" if avg_d else "—")
        with pm6:
            st.metric("Top Holding", str(summary.get("top_holding", "—")))

        # Sector allocation bar chart
        sector_alloc = summary.get("sector_allocation", {})
        if sector_alloc:
            st.markdown('<div class="sec">Sector Allocation</div>',
                        unsafe_allow_html=True)
            sa_df = (
                pd.DataFrame(list(sector_alloc.items()),
                             columns=["Sector", "Weight"])
                .sort_values("Weight", ascending=False)
            )
            fig_sa = go.Figure(go.Bar(
                x=sa_df["Sector"],
                y=sa_df["Weight"] * 100,
                marker_color=ACCENT2, marker_line_width=0,
                text=[f"{w*100:.1f}%" for w in sa_df["Weight"]],
                textposition="outside",
                textfont=dict(size=10, color="#94a3b8"),
            ))
            apply_plotly_layout(fig_sa, height=220, showlegend=False,
                                bargap=0.3, margin=dict(l=0, r=0, t=20, b=0))
            fig_sa.update_xaxes(showgrid=False)
            fig_sa.update_yaxes(gridcolor="#1c2333",
                                 title_text="Weight %",
                                 title_font=dict(size=9, color=TEXT_MUTED))
            st.plotly_chart(fig_sa, use_container_width=True)

        # ── Backtest section ──────────────────────────────────────
        st.markdown('<div class="sec">Backtest</div>', unsafe_allow_html=True)

        run_bt = st.button("▶  Run Backtest", type="primary")

        if run_bt:
            with st.spinner("Running backtest..."):
                try:
                    bt_result = backtest_portfolio(engine, portfolio_df)
                except Exception as exc:
                    st.error(f"Backtest failed: {exc}")
                    bt_result = None

            if bt_result is not None:

                # Show warnings
                bt_warns = bt_result.get("warnings", [])
                if bt_warns:
                    for w in bt_warns:
                        st.warning(w)

                if bt_result.get("status") == "error":
                    st.error(
                        "Backtest could not produce results. "
                        "Check that price data is loaded for these tickers."
                    )
                else:
                    # ── Metric cards ──────────────────────────────
                    metrics = bt_result.get("metrics", {})

                    bm1, bm2, bm3, bm4, bm5, bm6 = st.columns(6)

                    cum_ret = metrics.get("cumulative_return")
                    ann_ret = metrics.get("annualized_return")
                    ann_vol = metrics.get("annualized_volatility")
                    sharpe  = metrics.get("sharpe_ratio")
                    mdd     = metrics.get("max_drawdown")
                    t_days  = metrics.get("trading_days", 0)

                    with bm1:
                        st.metric("Cumulative Return",
                                  _pct_fmt(cum_ret) if cum_ret is not None else "N/A")
                    with bm2:
                        st.metric("Ann. Return",
                                  _pct_fmt(ann_ret) if ann_ret is not None else "N/A")
                    with bm3:
                        st.metric("Ann. Volatility",
                                  _pct_fmt(ann_vol, signed=False) if ann_vol is not None else "N/A")
                    with bm4:
                        st.metric("Sharpe Ratio",
                                  f"{sharpe:.3f}" if sharpe is not None else "N/A")
                    with bm5:
                        st.metric("Max Drawdown",
                                  _pct_fmt(mdd) if mdd is not None else "N/A")
                    with bm6:
                        st.metric("Trading Days", str(t_days))

                    # ── Cumulative returns chart ───────────────────
                    cum_series = bt_result.get("cumulative_returns")

                    if cum_series is not None and not cum_series.empty:
                        st.markdown(
                            '<div class="sec">Cumulative Returns</div>',
                            unsafe_allow_html=True,
                        )
                        fig_bt = go.Figure()
                        fig_bt.add_trace(go.Scatter(
                            x=cum_series.index,
                            y=(cum_series - 1) * 100,
                            name="Portfolio",
                            line=dict(color=ACCENT, width=2),
                            fill="tozeroy",
                            fillcolor="rgba(0,212,170,0.06)",
                        ))
                        fig_bt.add_hline(
                            y=0, line_dash="dot",
                            line_color="#374151", line_width=1,
                        )
                        apply_plotly_layout(
                            fig_bt, height=300,
                            title=dict(
                                text="Cumulative Return % over Time",
                                font=dict(size=11, color="#64748b"),
                            ),
                            margin=dict(l=0, r=0, t=40, b=0),
                        )
                        fig_bt.update_xaxes(showgrid=False, zeroline=False,
                                             tickfont=dict(size=10, color="#475569"))
                        fig_bt.update_yaxes(gridcolor="#1c2333", zeroline=False,
                                             tickfont=dict(size=10, color="#475569"),
                                             ticksuffix="%")
                        st.plotly_chart(fig_bt, use_container_width=True)

                    # ── Full metrics table ────────────────────────
                    with st.expander("Full backtest metrics table"):
                        bt_df = backtest_to_dataframe(bt_result)
                        st.dataframe(bt_df, use_container_width=True,
                                     hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 5 — IMPORT
# ═══════════════════════════════════════════════════════════════════

with tab_import:

    st.markdown('<div class="sec">Import CSV Data</div>', unsafe_allow_html=True)

    with st.expander("📋 Accepted CSV Format"):
        st.markdown("""
**Required columns** (flexible names, case-insensitive):

| Canonical | Accepted aliases |
|-----------|-----------------|
| `ticker`  | Symbol, symbol |
| `date`    | Date, DATE, trade_date |
| `open`    | Open, OPEN |
| `high`    | High, HIGH |
| `low`     | Low, LOW |
| `close`   | Close, CLOSE, Adj Close |
| `volume`  | Volume, VOLUME |

Extra columns (`LDCP`, `CHANGE`, etc.) are silently ignored.
        """)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            preview_df = pd.read_csv(uploaded_file, nrows=10)
            st.markdown("**Preview — first 10 rows:**")
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
            uploaded_file.seek(0)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        headers        = list(preview_df.columns)
        has_ticker_col = any(h.lower() in ["ticker", "symbol"] for h in headers)
        st.info(f"Detected: **{'Combined (multi-ticker)' if has_ticker_col else 'Per-ticker'}**")

        if st.button("✅ Validate & Import", type="primary"):
            with st.spinner("Validating and importing..."):
                try:
                    temp_path = KAGGLE_DIR / uploaded_file.name
                    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
                    temp_path.write_bytes(uploaded_file.getvalue())

                    raw_df = (load_combined_csv(temp_path)
                              if has_ticker_col
                              else load_per_ticker_csv(temp_path))

                    clean_df, result = validate_ohlcv(
                        raw_df, source=uploaded_file.name, auto_clean=True
                    )

                    if result.passed:
                        st.success(f"✅ Validation passed — {result.rows_after:,} clean rows.")
                    else:
                        st.error("❌ Validation failed.")

                    with st.expander("Validation Details"):
                        st.text(result.summary())

                    if result.passed and not clean_df.empty:
                        upsert_prices(clean_df)
                        db = db_summary()
                        st.success(
                            f"✅ Import complete — database now has "
                            f"**{db['price_rows']:,}** rows across "
                            f"**{db.get('tickers_with_data', db.get('tickers_with_prices','?'))}** ticker(s)."
                        )
                        engine = reload_engine()
                        st.success("Engine refreshed — switch to ⬡ Screener tab.")
                    elif clean_df.empty:
                        st.warning("No valid rows after cleaning.")

                except Exception as exc:
                    st.error(f"❌ Import failed: {exc}")
    else:
        st.divider()
        st.markdown("**Or import all CSVs already in `data/raw/kaggle/`:**")
        kaggle_files = sorted(KAGGLE_DIR.glob("*.csv")) if KAGGLE_DIR.exists() else []
        if kaggle_files:
            for f in kaggle_files:
                st.write(f"  • `{f.name}`")
        else:
            st.info("No CSV files found in `data/raw/kaggle/`.")

        if st.button("📂 Import All from data/raw/kaggle/"):
            with st.spinner("Loading and importing..."):
                try:
                    raw_df = load_all_from_kaggle_dir()
                    if raw_df.empty:
                        st.warning("No data loaded.")
                    else:
                        clean_df, result = validate_ohlcv(
                            raw_df, source="kaggle_dir", auto_clean=True
                        )
                        with st.expander("Validation Details"):
                            st.text(result.summary())
                        if result.passed and not clean_df.empty:
                            upsert_prices(clean_df)
                            db = db_summary()
                            st.success(
                                f"✅ Imported {result.rows_after:,} rows — "
                                f"database now has {db['price_rows']:,} total rows."
                            )
                            engine = reload_engine()
                            st.success("Engine refreshed.")
                        else:
                            st.error("Validation failed or no clean rows.")
                except Exception as exc:
                    st.error(f"❌ Import failed: {exc}")


# ── Footer ───────────────────────────────────────────────────────────

st.divider()
st.markdown(
    f'<div style="text-align:center;font-size:0.57rem;color:{TEXT_DIM};'
    f'letter-spacing:0.18em;text-transform:uppercase;">'
    f'PSX40 STOCK SCREENER &nbsp;·&nbsp; PHASE 10 &nbsp;·&nbsp; '
    f'TECHNICAL + FUNDAMENTAL + SECTOR + PORTFOLIO &nbsp;·&nbsp; NOT FINANCIAL ADVICE'
    f'</div>',
    unsafe_allow_html=True,
)