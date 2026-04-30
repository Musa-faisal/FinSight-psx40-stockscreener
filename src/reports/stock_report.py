"""
stock_report.py
---------------
Phase 9 — Single-Stock Analyst Report for PSX40 Screener.

Produces a structured analyst-style report dict for any ticker in the
screener DataFrame.  Safe to import and test in Jupyter Notebook.

Public API
----------
    build_stock_report(df, ticker) -> dict
    build_analyst_summary(report) -> str
    report_to_dataframe(report) -> pd.DataFrame
    get_report_warning_messages(report) -> list[str]
"""

from __future__ import annotations

import math
import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# SAFE HELPERS
# ═══════════════════════════════════════════════════════════════════

def _safe_get(row: pd.Series, col: str):
    """Return row[col] or None, treating NaN as None."""
    if col not in row.index:
        return None
    val = row[col]
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return val


def _safe_float(val, decimals: int = 2):
    """Return rounded float or None."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, decimals)
    except (TypeError, ValueError):
        return None


def _safe_text(val):
    """Return string or None."""
    if val is None:
        return None
    try:
        s = str(val)
        return s if s.strip() and s.lower() != "nan" else None
    except Exception:
        return None


def _fmt_number(val, decimals: int = 2, suffix: str = "") -> str:
    """Format a number for display, or return 'N/A'."""
    f = _safe_float(val, decimals)
    if f is None:
        return "N/A"
    return f"{f:,.{decimals}f}{suffix}"


def _fmt_pct(val, decimals: int = 1) -> str:
    """Format a decimal (e.g. 0.05) as a percentage string."""
    f = _safe_float(val, decimals + 2)
    if f is None:
        return "N/A"
    return f"{f * 100:,.{decimals}f}%"


def _fmt_vs_sector(val, decimals: int = 1) -> str:
    """Format a fractional vs-sector value (e.g. -0.15 → -15.0%)."""
    f = _safe_float(val, decimals + 2)
    if f is None:
        return "N/A"
    return f"{f * 100:+.1f}%"


# ═══════════════════════════════════════════════════════════════════
# BUILD REPORT
# ═══════════════════════════════════════════════════════════════════

def build_stock_report(df: pd.DataFrame, ticker: str) -> dict:
    """
    Build a comprehensive analyst report dict for *ticker*.

    Parameters
    ----------
    df : pd.DataFrame
        Full screener DataFrame (e.g. engine.get_screener_df()).
    ticker : str
        Ticker symbol (case-insensitive).

    Returns
    -------
    dict
        Structured report with sections: identity, scores, fundamentals,
        sector_comparison, technicals, verdict, summary, warnings.

    Raises
    ------
    ValueError
        If ticker is not found in df.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")

    ticker = ticker.strip().upper()
    match = df[df["ticker"].str.upper() == ticker]
    if match.empty:
        available = sorted(df["ticker"].dropna().unique().tolist())
        raise ValueError(
            f"Ticker '{ticker}' not found in screener. "
            f"Available tickers: {', '.join(available[:20])}"
            f"{'...' if len(available) > 20 else ''}"
        )

    row = match.iloc[0]

    # ── Identity ────────────────────────────────────────────────
    identity = {
        "ticker":         _safe_text(_safe_get(row, "ticker")),
        "name":           _safe_text(_safe_get(row, "name")),
        "sector":         _safe_text(_safe_get(row, "sector")),
        "latest_close":   _safe_float(_safe_get(row, "latest_close"), 2),
    }

    # ── Scores ────────────────────────────────────────────────────
    scores = {
        "technical_score":   _safe_float(_safe_get(row, "technical_score"),   2),
        "fundamental_score": _safe_float(_safe_get(row, "fundamental_score"), 2),
        "risk_score":        _safe_float(_safe_get(row, "risk_score"),        2),
        "composite_score":   _safe_float(_safe_get(row, "composite_score"),   2),
    }

    # ── Fundamentals ──────────────────────────────────────────────
    fundamentals = {
        "pe_ratio":            _safe_float(_safe_get(row, "pe_ratio"),            2),
        "pb_ratio":            _safe_float(_safe_get(row, "pb_ratio"),            2),
        "roe":                 _safe_float(_safe_get(row, "roe"),                 4),
        "debt_to_equity":      _safe_float(_safe_get(row, "debt_to_equity"),      2),
        "dividend_yield":      _safe_float(_safe_get(row, "dividend_yield"),      4),
        "net_profit_margin":   _safe_float(_safe_get(row, "net_profit_margin"),   4),
        "payout_ratio":        _safe_float(_safe_get(row, "payout_ratio"),        2),
        "data_quality_score":  _safe_float(_safe_get(row, "data_quality_score"),  1),
        "fundamental_notes":   _safe_text(_safe_get(row, "fundamental_notes")),
    }

    # ── Sector Comparison ───────────────────────────────────────────
    sector_comparison = {
        "sector_avg_pe":              _safe_float(_safe_get(row, "sector_avg_pe"),              2),
        "sector_median_pe":           _safe_float(_safe_get(row, "sector_median_pe"),           2),
        "pe_vs_sector":               _safe_float(_safe_get(row, "pe_vs_sector"),               4),
        "sector_avg_pb":              _safe_float(_safe_get(row, "sector_avg_pb"),              2),
        "pb_vs_sector":               _safe_float(_safe_get(row, "pb_vs_sector"),               4),
        "sector_avg_roe":             _safe_float(_safe_get(row, "sector_avg_roe"),             4),
        "roe_vs_sector":              _safe_float(_safe_get(row, "roe_vs_sector"),              4),
        "sector_avg_dividend_yield":  _safe_float(_safe_get(row, "sector_avg_dividend_yield"),   4),
        "dividend_yield_vs_sector":   _safe_float(_safe_get(row, "dividend_yield_vs_sector"),    4),
        "sector_avg_composite_score": _safe_float(_safe_get(row, "sector_avg_composite_score"), 2),
        "composite_score_vs_sector":  _safe_float(_safe_get(row, "composite_score_vs_sector"),  4),
        "sector_avg_risk_score":      _safe_float(_safe_get(row, "sector_avg_risk_score"),      2),
        "risk_score_vs_sector":       _safe_float(_safe_get(row, "risk_score_vs_sector"),       4),
        "sector_value_label":         _safe_text(_safe_get(row, "sector_value_label")),
    }

    # ── Technicals ──────────────────────────────────────────────────
    technicals = {
        "sma_20":        _safe_float(_safe_get(row, "sma_20"),        2),
        "sma_50":        _safe_float(_safe_get(row, "sma_50"),        2),
        "sma_200":       _safe_float(_safe_get(row, "sma_200"),       2),
        "rsi_14":        _safe_float(_safe_get(row, "rsi_14"),        2),
        "macd_hist":     _safe_float(_safe_get(row, "macd_hist"),     4),
        "return_1m":     _safe_float(_safe_get(row, "return_1m"),     4),
        "return_3m":     _safe_float(_safe_get(row, "return_3m"),     4),
        "return_6m":     _safe_float(_safe_get(row, "return_6m"),     4),
        "breakout_ratio": _safe_float(_safe_get(row, "breakout_ratio"), 4),
        "volatility_30d": _safe_float(_safe_get(row, "volatility_30d"), 4),
        "volatility":    _safe_float(_safe_get(row, "volatility"),    4),
        "volume_surge":  _safe_float(_safe_get(row, "volume_surge"),  3),
        "volume_ratio":  _safe_float(_safe_get(row, "volume_ratio"),  3),
        "avg_volume_20d": _safe_float(_safe_get(row, "avg_volume_20d"), 0),
    }

    # ── Verdict ─────────────────────────────────────────────────────
    verdict = {
        "final_verdict":     _safe_text(_safe_get(row, "final_verdict")) or _safe_text(_safe_get(row, "verdict")),
        "verdict_emoji":     _safe_text(_safe_get(row, "verdict_emoji")),
        "verdict_color":     _safe_text(_safe_get(row, "verdict_color")),
        "verdict_rationale": _safe_text(_safe_get(row, "verdict_rationale")),
    }

    report = {
        "identity":        identity,
        "scores":          scores,
        "fundamentals":    fundamentals,
        "sector_comparison": sector_comparison,
        "technicals":      technicals,
        "verdict":         verdict,
        "summary":         "",   # populated below
        "warnings":        [],   # populated below
    }

    report["summary"] = build_analyst_summary(report)
    report["warnings"] = get_report_warning_messages(report)

    return report


# ═══════════════════════════════════════════════════════════════════
# ANALYST SUMMARY
# ═══════════════════════════════════════════════════════════════════

def build_analyst_summary(report: dict) -> str:
    """
    Build a deterministic, data-driven analyst paragraph.
    No LLM calls.  Cautious language only.
    """
    parts: list[str] = []

    id_     = report["identity"]
    scores  = report["scores"]
    fund    = report["fundamentals"]
    sec     = report["sector_comparison"]
    tech    = report["technicals"]
    verdict = report["verdict"]

    name = id_.get("name") or id_.get("ticker") or "The company"
    sector = id_.get("sector") or "its sector"

    # ── Opening ───────────────────────────────────────────────────
    parts.append(f"{name} ({id_.get('ticker', 'N/A')}) operates in the {sector}.")

    # ── Valuation ─────────────────────────────────────────────────
    pe = fund.get("pe_ratio")
    pb = fund.get("pb_ratio")
    pe_vs = sec.get("pe_vs_sector")
    pb_vs = sec.get("pb_vs_sector")

    val_parts = []
    if pe is not None:
        val_parts.append(f"a P/E of {_fmt_number(pe)}")
    if pb is not None:
        val_parts.append(f"a P/B of {_fmt_number(pb)}")
    if val_parts:
        val_str = " and ".join(val_parts)
        parts.append(f"Valuation metrics suggest {val_str}.")

    if pe_vs is not None:
        if pe_vs < -0.1:
            parts.append("The stock appears cheaper than its sector peers on a P/E basis.")
        elif pe_vs > 0.1:
            parts.append("The stock appears more expensive than its sector peers on a P/E basis.")

    if pb_vs is not None:
        if pb_vs < -0.1:
            parts.append("Its price-to-book ratio also appears below the sector average.")
        elif pb_vs > 0.1:
            parts.append("Its price-to-book ratio appears above the sector average.")

    # ── Profitability ─────────────────────────────────────────────
    roe = fund.get("roe")
    npm = fund.get("net_profit_margin")
    roe_vs = sec.get("roe_vs_sector")

    if roe is not None:
        parts.append(f"ROE stands at {_fmt_pct(roe)}.")
    if npm is not None:
        parts.append(f"Net profit margin is {_fmt_pct(npm)}.")
    if roe_vs is not None:
        if roe_vs > 0.05:
            parts.append("Profitability appears stronger than the sector average.")
        elif roe_vs < -0.05:
            parts.append("Profitability appears weaker than the sector average.")

    # ── Debt / Risk ───────────────────────────────────────────────
    de = fund.get("debt_to_equity")
    risk = scores.get("risk_score")
    risk_vs = sec.get("risk_score_vs_sector")

    if de is not None:
        if de > 2:
            parts.append("The balance sheet carries elevated leverage relative to equity.")
        elif de < 0.5:
            parts.append("The balance sheet appears conservatively leveraged.")
        else:
            parts.append("Leverage appears moderate relative to equity.")

    if risk is not None:
        if risk > 70:
            parts.append("The overall risk score is elevated based on the available data.")
        elif risk < 35:
            parts.append("The overall risk profile appears relatively low.")

    if risk_vs is not None:
        if risk_vs > 0.1:
            parts.append("Risk is higher than the sector average.")
        elif risk_vs < -0.1:
            parts.append("Risk is lower than the sector average.")

    # ── Dividend ──────────────────────────────────────────────────
    dy = fund.get("dividend_yield")
    dy_vs = sec.get("dividend_yield_vs_sector")
    payout = fund.get("payout_ratio")

    if dy is not None and dy > 0:
        parts.append(f"The stock offers a dividend yield of approximately {_fmt_pct(dy)}.")
        if dy_vs is not None:
            if dy_vs > 0.05:
                parts.append("This yield appears above the sector average.")
            elif dy_vs < -0.05:
                parts.append("This yield appears below the sector average.")
        if payout is not None:
            if payout > 100:
                parts.append("The payout ratio exceeds 100%, which may suggest the dividend is not fully covered by earnings.")
            elif payout > 60:
                parts.append("The payout ratio is moderate to high.")
            elif payout > 0:
                parts.append("The payout ratio appears sustainable.")
    elif dy is not None and dy <= 0:
        parts.append("The company does not currently pay a dividend based on available data.")

    # ── Technical ─────────────────────────────────────────────────
    rsi = tech.get("rsi_14")
    ret1m = tech.get("return_1m")
    ret3m = tech.get("return_3m")
    tech_score = scores.get("technical_score")

    if tech_score is not None:
        if tech_score >= 70:
            parts.append("Technical indicators suggest a strong trend.")
        elif tech_score <= 40:
            parts.append("Technical indicators suggest a weak or deteriorating trend.")
        else:
            parts.append("Technical indicators are mixed.")

    if rsi is not None:
        if rsi < 30:
            parts.append("RSI is in oversold territory, which may suggest a potential bounce, though not guaranteed.")
        elif rsi > 70:
            parts.append("RSI is in overbought territory, which may suggest momentum exhaustion.")

    if ret1m is not None and ret3m is not None:
        if ret1m > 0 and ret3m > 0:
            parts.append("Price momentum has been positive over the past 1 and 3 months.")
        elif ret1m < 0 and ret3m < 0:
            parts.append("Price momentum has been negative over the past 1 and 3 months.")

    # ── Sector comparison ────────────────────────────────────────
    score_vs = sec.get("composite_score_vs_sector")
    label = sec.get("sector_value_label")

    if score_vs is not None:
        if score_vs > 0.15:
            parts.append("The composite score is meaningfully above the sector average.")
        elif score_vs < -0.15:
            parts.append("The composite score is below the sector average.")

    if label and label != "Insufficient sector data":
        parts.append(f"Sector benchmarking classifies this stock as: '{label}'.")

    # ── Verdict wrap-up ────────────────────────────────────────────
    final = verdict.get("final_verdict")
    if final:
        parts.append(f"Overall assessment: {final}.")

    rationale = verdict.get("verdict_rationale")
    if rationale:
        parts.append(rationale)

    # ── Disclaimer ────────────────────────────────────────────────
    parts.append(
        "This summary is based solely on the quantitative data available in the screener and does not constitute investment advice."
    )

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# FLATTEN TO DATAFRAME
# ═══════════════════════════════════════════════════════════════════

def report_to_dataframe(report: dict) -> pd.DataFrame:
    """
    Flatten a report dict into a tidy DataFrame with columns:
        section | metric | value

    Useful for Jupyter display.
    """
    rows: list[dict] = []

    for section_name, section_data in report.items():
        if section_name in ("summary", "warnings"):
            continue
        if not isinstance(section_data, dict):
            continue
        for metric, value in section_data.items():
            rows.append({
                "section": section_name,
                "metric":  metric,
                "value":   value if value is not None else "N/A",
            })

    # Add summary as a single row
    rows.append({
        "section": "summary",
        "metric":  "analyst_summary",
        "value":   report.get("summary", "N/A"),
    })

    # Add warnings
    warnings = report.get("warnings", [])
    if warnings:
        for i, w in enumerate(warnings):
            rows.append({
                "section": "warnings",
                "metric":  f"warning_{i+1}",
                "value":   w,
            })
    else:
        rows.append({
            "section": "warnings",
            "metric":  "warnings",
            "value":   "None",
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# WARNING MESSAGES
# ═══════════════════════════════════════════════════════════════════

def get_report_warning_messages(report: dict) -> list[str]:
    """
    Return a list of human-readable warnings based on data gaps
    or red flags in the report.
    """
    warnings: list[str] = []

    fund = report.get("fundamentals", {})
    sec = report.get("sector_comparison", {})
    tech = report.get("technicals", {})
    scores = report.get("scores", {})

    # Missing fundamentals
    core_fund = ["pe_ratio", "pb_ratio", "roe", "debt_to_equity", "dividend_yield"]
    missing_fund = [k for k in core_fund if fund.get(k) is None]
    if missing_fund:
        warnings.append(
            f"Missing fundamental data: {', '.join(missing_fund)}. "
            "Scores may be penalised."
        )

    # Data quality
    dq = fund.get("data_quality_score")
    if dq is not None and dq < 60:
        warnings.append(
            f"Low data quality score ({dq:.0f}/100). "
            "Fundamental metrics may be incomplete or unreliable."
        )

    # Missing sector comparison
    core_sec = ["sector_avg_pe", "pe_vs_sector", "composite_score_vs_sector"]
    missing_sec = [k for k in core_sec if sec.get(k) is None]
    if missing_sec:
        warnings.append(
            f"Missing sector comparison data: {', '.join(missing_sec)}."
        )

    # Sector label
    label = sec.get("sector_value_label")
    if label == "Insufficient sector data":
        warnings.append(
            "Insufficient sector peers for reliable benchmarking."
        )

    # Missing technicals
    core_tech = ["rsi_14", "sma_20", "sma_50", "macd_hist"]
    missing_tech = [k for k in core_tech if tech.get(k) is None]
    if missing_tech:
        warnings.append(
            f"Missing technical fields: {', '.join(missing_tech)}."
        )

    # Risk red flags
    risk = scores.get("risk_score")
    if risk is not None and risk >= 75:
        warnings.append(
            f"High risk score ({risk:.1f}/100). Exercise caution."
        )

    # Negative earnings / equity
    notes = fund.get("fundamental_notes") or ""
    if "Negative EPS" in notes:
        warnings.append("Negative or zero EPS — valuation ratios may be unreliable.")
    if "Negative equity" in notes:
        warnings.append("Negative equity — balance sheet risk is elevated.")

    return warnings
