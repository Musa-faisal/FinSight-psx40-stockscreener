import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from src.ingestion.kaggle_csv_loader import (
    load_combined_csv,
    load_per_ticker_csv,
    KAGGLE_DIR,
)
from src.ingestion.data_validator import validate_ohlcv
from src.database.db_manager import upsert_prices, db_summary
from config.settings import SMA_SHORT, SMA_LONG, SMA_200, RSI_PERIOD

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PSX40 Stock Screener",
    page_icon="📈",
    layout="wide",
)

# ── Engine ────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading PSX40 engine...")
def load_engine() -> ScreenerEngine:
    engine = ScreenerEngine()
    engine.run()
    return engine


def reload_engine() -> ScreenerEngine:
    st.cache_resource.clear()
    engine = ScreenerEngine()
    engine.run(force_reseed=False)
    return engine


engine = load_engine()

# ── Title ─────────────────────────────────────────────────────────────────────

st.title("📈 PSX40 Stock Screener")
st.warning(
    "⚠️ **Disclaimer:** Data may be synthetically generated or imported from CSV. "
    "This is not real-time PSX data and is not financial advice."
)

tab_dashboard, tab_factors, tab_import = st.tabs(
    ["📊 Dashboard", "🔬 Factor Detail", "📥 Import Data"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard
# ══════════════════════════════════════════════════════════════════════════════

with tab_dashboard:

    summary = engine.summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks Scored",  summary["total_tickers"])
    c2.metric("Sectors",        summary["sectors"])
    c3.metric("Avg Score",      summary["avg_score"])
    c4.metric("Top Ticker",     f"{summary['top_ticker']}  ({summary['top_score']})")

    st.divider()

    # ── Sidebar ───────────────────────────────────────────────────────────────

    st.sidebar.header("Filters")

    all_sectors      = engine.get_sectors()
    selected_sectors = st.sidebar.multiselect(
        "Sector", options=all_sectors, default=[], placeholder="All sectors"
    )

    verdict_options   = ["Strong Buy", "Buy", "Hold", "Weak", "Avoid"]
    selected_verdicts = st.sidebar.multiselect(
        "Verdict", options=verdict_options, default=[], placeholder="All verdicts"
    )

    score_range = st.sidebar.slider(
        "Composite Score Range", min_value=0, max_value=100, value=(0, 100), step=5
    )

    all_tickers     = engine.get_tickers()
    selected_ticker = st.sidebar.selectbox("Price Chart Ticker", options=all_tickers, index=0)

    # ── Screener table ────────────────────────────────────────────────────────

    filtered_df = engine.filter(
        sectors=selected_sectors   if selected_sectors   else None,
        verdicts=selected_verdicts if selected_verdicts  else None,
        min_score=float(score_range[0]),
        max_score=float(score_range[1]),
    )

    st.subheader(f"Screener Results — {len(filtered_df)} stocks")

    display_cols = [
        "ticker", "name", "sector", "latest_close",
        f"sma_{SMA_SHORT}", f"sma_{SMA_LONG}",
        f"rsi_{RSI_PERIOD}", "macd_hist",
        "breakout_ratio", "volatility_30d", "max_drawdown",
        "volume_surge",
        "trend_score", "momentum_score", "risk_score", "volume_score",
        "composite_score", "verdict_emoji", "verdict",
    ]
    display_cols = [c for c in display_cols if c in filtered_df.columns]

    st.dataframe(
        filtered_df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "composite_score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%.1f"
            ),
            "trend_score":    st.column_config.NumberColumn("Trend",    format="%.1f"),
            "momentum_score": st.column_config.NumberColumn("Momentum", format="%.1f"),
            "risk_score":     st.column_config.NumberColumn("Risk",     format="%.1f"),
            "volume_score":   st.column_config.NumberColumn("Volume",   format="%.1f"),
            "latest_close":   st.column_config.NumberColumn("Close ₨",  format="₨%.2f"),
            f"sma_{SMA_SHORT}": st.column_config.NumberColumn(f"SMA{SMA_SHORT}", format="%.2f"),
            f"sma_{SMA_LONG}":  st.column_config.NumberColumn(f"SMA{SMA_LONG}",  format="%.2f"),
            f"rsi_{RSI_PERIOD}": st.column_config.NumberColumn("RSI",   format="%.1f"),
            "macd_hist":      st.column_config.NumberColumn("MACD Hist", format="%.3f"),
            "breakout_ratio": st.column_config.NumberColumn("52W Pos",  format="%.1%%"),
            "volatility_30d": st.column_config.NumberColumn("Vol 30d",  format="%.1%%"),
            "max_drawdown":   st.column_config.NumberColumn("Max DD",   format="%.1%%"),
            "volume_surge":   st.column_config.NumberColumn("Vol Surge", format="%.2fx"),
            "verdict_emoji":  st.column_config.TextColumn(""),
        },
    )

    csv_bytes = filtered_df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Results as CSV", data=csv_bytes,
        file_name="psx40_results.csv", mime="text/csv"
    )

    st.divider()

    # ── Price chart ───────────────────────────────────────────────────────────

    st.subheader(f"Price Chart — {selected_ticker}")
    price_df = engine.get_price_df(selected_ticker)

    if price_df.empty:
        st.warning(f"No price data for {selected_ticker}.")
    else:
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.55, 0.22, 0.23],
            subplot_titles=(
                f"{selected_ticker} — Price & SMAs",
                "MACD",
                f"RSI {RSI_PERIOD}",
            ),
        )

        # Price + SMAs
        fig.add_trace(go.Scatter(
            x=price_df["date"], y=price_df["close"],
            name="Close", line=dict(color="#2563eb", width=1.8)
        ), row=1, col=1)

        for sma_w, color, dash in [
            (SMA_SHORT, "#f59e0b", "solid"),
            (SMA_LONG,  "#7c3aed", "dot"),
            (SMA_200,   "#dc2626", "dash"),
        ]:
            col_name = f"sma_{sma_w}"
            if col_name in price_df.columns:
                fig.add_trace(go.Scatter(
                    x=price_df["date"], y=price_df[col_name],
                    name=f"SMA {sma_w}",
                    line=dict(color=color, width=1.3, dash=dash),
                ), row=1, col=1)

        # MACD
        if "macd_line" in price_df.columns:
            fig.add_trace(go.Scatter(
                x=price_df["date"], y=price_df["macd_line"],
                name="MACD", line=dict(color="#0891b2", width=1.3)
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=price_df["date"], y=price_df["macd_signal"],
                name="Signal", line=dict(color="#f59e0b", width=1.3)
            ), row=2, col=1)
            fig.add_trace(go.Bar(
                x=price_df["date"], y=price_df["macd_hist"],
                name="Histogram",
                marker_color=price_df["macd_hist"].apply(
                    lambda v: "#00C853" if (v is not None and v >= 0) else "#D50000"
                ),
                opacity=0.6,
            ), row=2, col=1)

        # RSI
        rsi_col = f"rsi_{RSI_PERIOD}"
        if rsi_col in price_df.columns:
            fig.add_trace(go.Scatter(
                x=price_df["date"], y=price_df[rsi_col],
                name=f"RSI {RSI_PERIOD}",
                line=dict(color="#0891b2", width=1.3),
                fill="tozeroy", fillcolor="rgba(8,145,178,0.05)",
            ), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash",
                          line_color="rgba(220,38,38,0.6)", line_width=1,
                          annotation_text="70", annotation_font_size=9,
                          row=3, col=1)
            fig.add_hline(y=30, line_dash="dash",
                          line_color="rgba(22,163,74,0.6)", line_width=1,
                          annotation_text="30", annotation_font_size=9,
                          row=3, col=1)

        fig.update_layout(
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        xanchor="left", x=0),
            hovermode="x unified",
            xaxis_rangeslider_visible=False,
        )
        fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
        fig.update_xaxes(showgrid=False)

        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Factor Detail
# ══════════════════════════════════════════════════════════════════════════════

with tab_factors:

    st.subheader("🔬 Factor Score Breakdown")
    st.caption(
        "Composite score = Trend 35% + Momentum 30% + Risk 20% + Volume 15%. "
        "Valuation reserved for Phase 3."
    )

    screener_df = engine.get_screener_df()

    if screener_df.empty:
        st.info("No data available.")
    else:
        # Ticker picker
        factor_ticker = st.selectbox(
            "Select ticker", options=sorted(screener_df["ticker"].tolist()), key="factor_ticker"
        )

        row = screener_df[screener_df["ticker"] == factor_ticker]
        if row.empty:
            st.warning("No data for this ticker.")
        else:
            r = row.iloc[0]

            # ── Group score bar chart ─────────────────────────────────────────
            group_labels  = ["Trend", "Momentum", "Risk", "Volume"]
            group_scores  = [
                r.get("trend_score", 0),
                r.get("momentum_score", 0),
                r.get("risk_score", 0),
                r.get("volume_score", 0),
            ]
            group_weights = [35, 30, 20, 15]
            group_colors  = ["#2563eb", "#7c3aed", "#dc2626", "#0891b2"]

            fig_groups = go.Figure()
            fig_groups.add_trace(go.Bar(
                x=group_labels,
                y=group_scores,
                marker_color=group_colors,
                text=[f"{s:.1f}" for s in group_scores],
                textposition="outside",
                customdata=group_weights,
                hovertemplate="%{x}<br>Score: %{y:.1f}<br>Weight: %{customdata}%<extra></extra>",
            ))
            fig_groups.add_hline(y=50, line_dash="dot",
                                 line_color="gray", line_width=1,
                                 annotation_text="Neutral 50")
            fig_groups.update_layout(
                title=f"{factor_ticker} — Group Scores  |  Composite: {r.get('composite_score', '—')}  {r.get('verdict_emoji','')} {r.get('verdict','')}",
                yaxis=dict(range=[0, 110]),
                height=350,
                margin=dict(l=0, r=0, t=50, b=0),
                showlegend=False,
            )
            st.plotly_chart(fig_groups, use_container_width=True)

            # ── Sub-factor detail ─────────────────────────────────────────────
            col_trend, col_mom, col_risk, col_vol = st.columns(4)

            with col_trend:
                st.markdown("**Trend Factors**")
                trend_items = {
                    "Price > SMA20":   r.get("price_above_sma20"),
                    "SMA20 > SMA50":   r.get("sma20_above_sma50"),
                    "SMA50 > SMA200":  r.get("sma50_above_sma200"),
                    "1M Return":       f"{(r.get('return_1m') or 0)*100:.1f}%",
                    "3M Return":       f"{(r.get('return_3m') or 0)*100:.1f}%",
                    "6M Return":       f"{(r.get('return_6m') or 0)*100:.1f}%",
                }
                for k, v in trend_items.items():
                    st.metric(k, v)

            with col_mom:
                st.markdown("**Momentum Factors**")
                st.metric(f"RSI {RSI_PERIOD}", f"{r.get(f'rsi_{RSI_PERIOD}', '—')}")
                st.metric("MACD Line",    f"{r.get('macd_line',  '—')}")
                st.metric("MACD Hist",    f"{r.get('macd_hist',  '—')}")
                st.metric("52W Position", f"{(r.get('breakout_ratio') or 0)*100:.1f}%")

            with col_risk:
                st.markdown("**Risk Factors**")
                st.metric("Volatility 30d", f"{(r.get('volatility_30d') or 0)*100:.1f}%")
                st.metric("Max Drawdown",   f"{(r.get('max_drawdown')   or 0)*100:.1f}%")
                st.metric("Downside Dev",   f"{(r.get('downside_deviation') or 0)*100:.1f}%")

            with col_vol:
                st.markdown("**Volume Factors**")
                avg_vol = r.get("avg_volume_20d")
                st.metric("Avg Vol 20d",   f"{int(avg_vol):,}" if avg_vol else "—")
                st.metric("Volume Surge",  f"{r.get('volume_surge', '—')}x")

            st.divider()

            # ── Full factor score table ───────────────────────────────────────
            st.markdown("**All Sub-Factor Scores (0–100)**")

            factor_score_cols = [
                "price_above_sma20", "sma20_above_sma50", "sma50_above_sma200",
                "rsi_score", "macd_score", "breakout_score",
                "volatility_score", "drawdown_score", "downside_deviation_score",
                "avg_volume_score", "volume_surge_score",
            ]
            factor_score_cols = [c for c in factor_score_cols if c in r.index]

            if factor_score_cols:
                factor_df = pd.DataFrame({
                    "Factor": factor_score_cols,
                    "Score":  [r.get(c) for c in factor_score_cols],
                })
                factor_df["Score"] = pd.to_numeric(factor_df["Score"], errors="coerce")

                fig_factors = go.Figure(go.Bar(
                    x=factor_df["Score"],
                    y=factor_df["Factor"],
                    orientation="h",
                    marker_color=factor_df["Score"].apply(
                        lambda s: "#00C853" if s >= 65
                        else ("#FFD600" if s >= 40 else "#D50000")
                        if pd.notna(s) else "#cccccc"
                    ),
                    text=[f"{s:.0f}" if pd.notna(s) else "N/A"
                          for s in factor_df["Score"]],
                    textposition="outside",
                ))
                fig_factors.update_layout(
                    height=320,
                    margin=dict(l=0, r=60, t=10, b=0),
                    xaxis=dict(range=[0, 115]),
                    showlegend=False,
                )
                st.plotly_chart(fig_factors, use_container_width=True)

            st.caption(
                "Valuation scores (PE, PB, DY) are reserved for Phase 3 "
                "when fundamental data is available."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Import Data
# ══════════════════════════════════════════════════════════════════════════════

with tab_import:

    st.subheader("📥 Import CSV Data")
    st.write(
        "Upload a CSV file with historical OHLCV data. "
        "Validated rows are upserted into the SQLite database."
    )

    with st.expander("📋 Accepted CSV Format"):
        st.markdown("""
**Required columns** (flexible names accepted):

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

        headers         = list(preview_df.columns)
        has_ticker_col  = any(h.lower() in ["ticker", "symbol"] for h in headers)
        format_label    = "Combined (multi-ticker)" if has_ticker_col else "Per-ticker"
        st.info(f"Detected format: **{format_label}**")

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
                            f"✅ Import complete — "
                            f"database now has **{db['price_rows']:,}** rows "
                            f"across **{db['tickers_with_data']}** ticker(s)."
                        )
                        engine = reload_engine()
                        st.success("Dashboard refreshed. Switch to 📊 Dashboard.")
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
                    from src.ingestion.kaggle_csv_loader import load_all_from_kaggle_dir
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
                            st.success("Dashboard refreshed.")
                        else:
                            st.error("Validation failed or no clean rows.")

                except Exception as exc:
                    st.error(f"❌ Import failed: {exc}")

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption("PSX40 Stock Screener · Phase 2 · Professional Factor Scoring · Not financial advice.")