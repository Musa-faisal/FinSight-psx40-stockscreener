"""
fundamental.py
--------------
Phase 4.2 — Fundamental Ratio Engine for PSX40 Screener.

Loads raw financial data, dividend data, and latest close prices
from the SQLite database and computes per-ticker financial ratios.

Public API
----------
    build_fundamental_metrics() -> pd.DataFrame
        One row per ticker with all ratios, quality scores, and notes.

    get_fundamental_for_ticker(ticker: str) -> dict
        Single-ticker dict — useful for dashboard detail panels.

Safety rules
------------
    - Division by zero        → NaN  (never crashes)
    - Negative EPS            → pe_ratio, payout_ratio = NaN
    - Negative equity         → roe, debt_to_equity   = NaN
    - Zero / negative revenue → net_profit_margin     = NaN
    - Zero / negative price   → dividend_yield         = NaN
    - Missing dividend data   → treated as 0 DPS
    - Missing financials      → row excluded unless price exists
    - Negative companies      → handled safely (SSGC, KEL, PSMC etc.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.database.db_manager import (
    load_latest_financials,
    load_annual_dividends,
    load_latest_prices,
)


# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

# Data quality score deductions
_DQ_MISSING_FINANCIALS = 20
_DQ_MISSING_DIVIDENDS  = 15
_DQ_MISSING_PRICE      = 15
_DQ_NEGATIVE_EPS       = 10
_DQ_NEGATIVE_EQUITY    = 10


# ═══════════════════════════════════════════════════════════════════
# SAFE DIVISION HELPER
# ═══════════════════════════════════════════════════════════════════

def _div(
    numerator: float | None,
    denominator: float | None,
    *,
    require_positive_denom: bool = False,
    require_positive_numer: bool = False,
    scale: float = 1.0,
) -> float | None:
    """
    Safe scalar division — always returns float or None.

    Parameters
    ----------
    numerator              : top of the fraction
    denominator            : bottom of the fraction
    require_positive_denom : if True, returns None when denom <= 0
    require_positive_numer : if True, returns None when numer <= 0
    scale                  : multiply result by this (e.g. 100 for %)

    Returns
    -------
    float rounded to 4 decimal places, or None if unsafe.
    """
    try:
        n = float(numerator)
        d = float(denominator)
    except (TypeError, ValueError):
        return None

    if np.isnan(n) or np.isnan(d):
        return None
    if d == 0:
        return None
    if require_positive_denom and d <= 0:
        return None
    if require_positive_numer and n <= 0:
        return None

    result = (n / d) * scale

    if np.isinf(result) or np.isnan(result):
        return None

    return round(result, 4)


# ═══════════════════════════════════════════════════════════════════
# RATIO CALCULATORS  (one function per ratio — small and readable)
# ═══════════════════════════════════════════════════════════════════

def _pe_ratio(close: float | None, eps: float | None) -> float | None:
    """Price / EPS.  NaN when EPS <= 0."""
    return _div(close, eps, require_positive_denom=True)


def _pb_ratio(
    close: float | None,
    book_value_per_share: float | None,
) -> float | None:
    """Price / Book Value Per Share.  NaN when BVPS <= 0."""
    return _div(close, book_value_per_share, require_positive_denom=True)


def _roe(
    net_profit: float | None,
    total_equity: float | None,
) -> float | None:
    """Net Profit / Total Equity × 100 (%).  NaN when equity <= 0."""
    return _div(net_profit, total_equity,
                require_positive_denom=True, scale=100.0)


def _debt_to_equity(
    total_debt: float | None,
    total_equity: float | None,
) -> float | None:
    """Total Debt / Total Equity.  NaN when equity <= 0."""
    return _div(total_debt, total_equity, require_positive_denom=True)


def _dividend_yield(
    annual_dps: float | None,
    close: float | None,
) -> float | None:
    """Annual DPS / Price × 100 (%).  NaN when price <= 0."""
    return _div(annual_dps, close, require_positive_denom=True, scale=100.0)


def _net_profit_margin(
    net_profit: float | None,
    revenue: float | None,
) -> float | None:
    """Net Profit / Revenue × 100 (%).  NaN when revenue <= 0."""
    return _div(net_profit, revenue, require_positive_denom=True, scale=100.0)


def _payout_ratio(
    annual_dps: float | None,
    eps: float | None,
) -> float | None:
    """Annual DPS / EPS × 100 (%).  NaN when EPS <= 0."""
    return _div(annual_dps, eps, require_positive_denom=True, scale=100.0)


# ═══════════════════════════════════════════════════════════════════
# DATA QUALITY SCORER
# ═══════════════════════════════════════════════════════════════════

def _data_quality_score(
    has_financials: bool,
    has_dividend:   bool,
    has_price:      bool,
    eps:            float | None,
    total_equity:   float | None,
) -> int:
    """
    Score from 0–100 reflecting how complete and reliable the data is.

    Deductions
    ----------
    -20  missing financial data
    -15  missing dividend data
    -15  missing latest price
    -10  EPS <= 0  (negative or zero earnings)
    -10  equity <= 0  (negative or zero equity)
    """
    score = 100

    if not has_financials:
        score -= _DQ_MISSING_FINANCIALS
    if not has_dividend:
        score -= _DQ_MISSING_DIVIDENDS
    if not has_price:
        score -= _DQ_MISSING_PRICE

    try:
        if eps is None or np.isnan(float(eps)) or float(eps) <= 0:
            score -= _DQ_NEGATIVE_EPS
    except (TypeError, ValueError):
        score -= _DQ_NEGATIVE_EPS

    try:
        if (total_equity is None
                or np.isnan(float(total_equity))
                or float(total_equity) <= 0):
            score -= _DQ_NEGATIVE_EQUITY
    except (TypeError, ValueError):
        score -= _DQ_NEGATIVE_EQUITY

    return max(0, min(100, score))


# ═══════════════════════════════════════════════════════════════════
# NOTES BUILDER
# ═══════════════════════════════════════════════════════════════════

def _build_notes(
    has_financials: bool,
    has_dividend:   bool,
    has_price:      bool,
    eps:            float | None,
    total_equity:   float | None,
) -> str:
    """
    Return a short plain-English note string describing data issues.
    Multiple issues are joined with " | ".
    """
    notes: list[str] = []

    if not has_financials:
        notes.append("Missing financial data")
    if not has_price:
        notes.append("Missing latest price")
    if not has_dividend:
        notes.append("Missing dividend data")

    try:
        if eps is not None and not np.isnan(float(eps)) and float(eps) <= 0:
            notes.append("Negative EPS")
    except (TypeError, ValueError):
        notes.append("Invalid EPS")

    try:
        if (total_equity is not None
                and not np.isnan(float(total_equity))
                and float(total_equity) <= 0):
            notes.append("Negative equity")
    except (TypeError, ValueError):
        notes.append("Invalid equity")

    return " | ".join(notes) if notes else "Complete data"


# ═══════════════════════════════════════════════════════════════════
# SINGLE-ROW BUILDER
# ═══════════════════════════════════════════════════════════════════

def _build_row(
    ticker:      str,
    fin_row:     pd.Series | None,
    div_row:     pd.Series | None,
    price_row:   pd.Series | None,
) -> dict:
    """
    Build one result dict for a single ticker.
    All inputs are optional — missing inputs produce NaN ratios and
    reduced data quality scores rather than crashes.

    Parameters
    ----------
    ticker    : PSX ticker symbol (already uppercased)
    fin_row   : row from load_latest_financials()  or None
    div_row   : row from load_annual_dividends()   or None
    price_row : row from load_latest_prices()      or None
    """

    # ── Flags ─────────────────────────────────────────────────────
    has_financials = fin_row is not None
    has_dividend   = div_row is not None
    has_price      = price_row is not None

    # ── Raw values ────────────────────────────────────────────────
    def _get(row: pd.Series | None, col: str) -> float | None:
        if row is None or col not in row.index:
            return None
        val = row[col]
        try:
            f = float(val)
            return None if np.isnan(f) else f
        except (TypeError, ValueError):
            return None

    fiscal_year          = int(fin_row["fiscal_year"]) if has_financials else None
    revenue              = _get(fin_row, "revenue")
    net_profit           = _get(fin_row, "net_profit")
    total_assets         = _get(fin_row, "total_assets")
    total_equity         = _get(fin_row, "total_equity")
    total_debt           = _get(fin_row, "total_debt")
    eps                  = _get(fin_row, "eps")
    book_value_per_share = _get(fin_row, "book_value_per_share")
    shares_outstanding   = _get(fin_row, "shares_outstanding")
    latest_close         = _get(price_row, "close")

    # Missing dividend → treat as 0 (not None) per spec
    annual_dps = _get(div_row, "annual_dps")
    if annual_dps is None:
        annual_dps = 0.0

    # ── Ratios ────────────────────────────────────────────────────
    pe_ratio           = _pe_ratio(latest_close, eps)
    pb_ratio           = _pb_ratio(latest_close, book_value_per_share)
    roe                = _roe(net_profit, total_equity)
    debt_to_equity     = _debt_to_equity(total_debt, total_equity)
    dividend_yield     = _dividend_yield(annual_dps, latest_close)
    net_profit_margin  = _net_profit_margin(net_profit, revenue)
    payout_ratio       = _payout_ratio(annual_dps, eps)

    # ── Quality ───────────────────────────────────────────────────
    dq_score = _data_quality_score(
        has_financials = has_financials,
        has_dividend   = has_dividend,
        has_price      = has_price,
        eps            = eps,
        total_equity   = total_equity,
    )
    notes = _build_notes(
        has_financials = has_financials,
        has_dividend   = has_dividend,
        has_price      = has_price,
        eps            = eps,
        total_equity   = total_equity,
    )

    return {
        # Identity
        "ticker":                  ticker,
        "fiscal_year":             fiscal_year,
        # Price
        "latest_close":            latest_close,
        # Raw financials
        "revenue":                 revenue,
        "net_profit":              net_profit,
        "total_assets":            total_assets,
        "total_equity":            total_equity,
        "total_debt":              total_debt,
        "eps":                     eps,
        "book_value_per_share":    book_value_per_share,
        "shares_outstanding":      shares_outstanding,
        # Dividend
        "annual_dividend_per_share": annual_dps,
        # Ratios
        "pe_ratio":                pe_ratio,
        "pb_ratio":                pb_ratio,
        "roe":                     roe,
        "debt_to_equity":          debt_to_equity,
        "dividend_yield":          dividend_yield,
        "net_profit_margin":       net_profit_margin,
        "payout_ratio":            payout_ratio,
        # Quality
        "data_quality_score":      dq_score,
        "fundamental_notes":       notes,
    }


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════

def build_fundamental_metrics() -> pd.DataFrame:
    """
    Load raw data from SQLite and return a tidy DataFrame of
    fundamental ratios — one row per ticker.

    Steps
    -----
    1. Load latest annual financials    (load_latest_financials)
    2. Load latest annual dividends     (load_annual_dividends)
    3. Load latest close prices         (load_latest_prices)
    4. Build the universe of tickers    (union of all three sources)
    5. For each ticker build one row    (_build_row)
    6. Return sorted DataFrame

    Returns
    -------
    pd.DataFrame with columns:
        ticker, fiscal_year, latest_close,
        revenue, net_profit, total_assets, total_equity, total_debt,
        eps, book_value_per_share, shares_outstanding,
        annual_dividend_per_share,
        pe_ratio, pb_ratio, roe, debt_to_equity,
        dividend_yield, net_profit_margin, payout_ratio,
        data_quality_score, fundamental_notes

    Notes
    -----
    - Returns empty DataFrame (not an exception) when DB has no data.
    - Tickers with partial data (e.g. price but no financials) still
      appear — data_quality_score and notes reflect what is missing.
    """

    # ── Load from DB ──────────────────────────────────────────────
    try:
        fin_df   = load_latest_financials()
    except Exception as exc:
        print(f"[fundamental] Warning: could not load financials — {exc}")
        fin_df   = pd.DataFrame()

    try:
        div_df   = load_annual_dividends()
    except Exception as exc:
        print(f"[fundamental] Warning: could not load dividends — {exc}")
        div_df   = pd.DataFrame()

    try:
        price_df = load_latest_prices()
    except Exception as exc:
        print(f"[fundamental] Warning: could not load prices — {exc}")
        price_df = pd.DataFrame()

    # ── Reduce dividend table to latest year per ticker ───────────
    if not div_df.empty and "fiscal_year" in div_df.columns:
        div_latest = (
            div_df
            .sort_values(["ticker", "fiscal_year"], ascending=[True, False])
            .groupby("ticker", as_index=False)
            .first()
        )
    else:
        div_latest = pd.DataFrame(columns=["ticker", "annual_dps"])

    # ── Build lookup dicts (ticker → Series) for O(1) access ─────
    fin_lookup   = (
        {r["ticker"]: r for _, r in fin_df.iterrows()}
        if not fin_df.empty else {}
    )
    div_lookup   = (
        {r["ticker"]: r for _, r in div_latest.iterrows()}
        if not div_latest.empty else {}
    )
    price_lookup = (
        {r["ticker"]: r for _, r in price_df.iterrows()}
        if not price_df.empty else {}
    )

    # ── Universe: every ticker appearing in any source ────────────
    all_tickers = sorted(
        set(fin_lookup.keys())
        | set(div_lookup.keys())
        | set(price_lookup.keys())
    )

    if not all_tickers:
        print("[fundamental] No tickers found in any data source.")
        return pd.DataFrame()

    # ── Build rows ────────────────────────────────────────────────
    rows: list[dict] = []
    for ticker in all_tickers:
        try:
            row = _build_row(
                ticker    = ticker,
                fin_row   = fin_lookup.get(ticker),
                div_row   = div_lookup.get(ticker),
                price_row = price_lookup.get(ticker),
            )
            rows.append(row)
        except Exception as exc:
            # Log and skip — one bad ticker must not crash the engine
            print(f"[fundamental] Skipping {ticker} due to error: {exc}")
            continue

    if not rows:
        return pd.DataFrame()

    # ── Assemble DataFrame ────────────────────────────────────────
    result = pd.DataFrame(rows)

    # Enforce column order explicitly
    ordered_cols = [
        "ticker", "fiscal_year", "latest_close",
        "revenue", "net_profit", "total_assets",
        "total_equity", "total_debt",
        "eps", "book_value_per_share", "shares_outstanding",
        "annual_dividend_per_share",
        "pe_ratio", "pb_ratio", "roe", "debt_to_equity",
        "dividend_yield", "net_profit_margin", "payout_ratio",
        "data_quality_score", "fundamental_notes",
    ]
    ordered_cols = [c for c in ordered_cols if c in result.columns]
    result       = result[ordered_cols].sort_values("ticker").reset_index(drop=True)

    print(
        f"[fundamental] Built metrics for {len(result)} ticker(s) | "
        f"avg quality score: "
        f"{result['data_quality_score'].mean():.1f}/100"
    )
    return result


# ═══════════════════════════════════════════════════════════════════
# SINGLE-TICKER HELPER
# ═══════════════════════════════════════════════════════════════════

def get_fundamental_for_ticker(ticker: str) -> dict:
    """
    Return fundamental metrics for a single ticker as a plain dict.

    Useful for:
    - Dashboard detail panels
    - Jupyter cell inspection
    - Unit testing individual tickers

    Returns empty dict if the ticker has no data at all.
    """
    ticker = ticker.strip().upper()
    df     = build_fundamental_metrics()

    if df.empty:
        return {}

    row = df[df["ticker"] == ticker]
    if row.empty:
        return {}

    # Replace NaN with None for cleaner downstream use
    return {
        k: (None if (isinstance(v, float) and np.isnan(v)) else v)
        for k, v in row.iloc[0].to_dict().items()
    }
