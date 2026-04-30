"""
fundamentals.py
---------------
Phase 4.2 — Fundamental Ratio Engine for PSX40 Screener.

Loads raw financial data, dividend data, and latest close prices
from the SQLite database and computes per-ticker financial ratios.

Synthetic fallback
------------------
When the database contains no financial statements (common on
Streamlit Cloud where the DB is ephemeral and only OHLCV is seeded),
this module generates plausible synthetic fundamental data so the
screener remains fully functional for demonstration purposes.

Public API
----------
    build_fundamental_metrics() -> pd.DataFrame
    get_fundamental_for_ticker(ticker: str) -> dict
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

_DQ_MISSING_FINANCIALS = 20
_DQ_MISSING_DIVIDENDS  = 15
_DQ_MISSING_PRICE      = 15
_DQ_NEGATIVE_EPS       = 10
_DQ_NEGATIVE_EQUITY    = 10

# Synthetic data seed — same as sample_data_generator for consistency
_SYNTH_SEED = 42

# Per-sector plausible ratio ranges  (pe_lo, pe_hi, roe_lo, roe_hi, de_lo, de_hi, dy_lo, dy_hi)
_SECTOR_PARAMS: dict[str, tuple] = {
    "Energy":          (6,  14, 12, 28, 0.3, 1.2, 4,  9),
    "Banking":         (5,  10, 14, 30, 4.0, 9.0, 5,  12),
    "Fertilizer":      (8,  16, 18, 35, 0.2, 0.9, 6,  14),
    "Cement":          (7,  18, 10, 22, 0.5, 1.5, 2,  7),
    "Textile":         (6,  14,  8, 18, 0.8, 2.0, 2,  6),
    "Technology":      (12, 30, 15, 35, 0.1, 0.6, 1,  4),
    "Pharmaceuticals": (10, 22, 14, 28, 0.2, 0.8, 2,  6),
    "Food & Beverages":(10, 25, 16, 32, 0.4, 1.0, 3,  8),
    "Chemicals":       (8,  18, 12, 25, 0.3, 1.0, 2,  7),
    "Automobile":      (8,  20, 10, 22, 0.5, 1.5, 2,  6),
}
_DEFAULT_PARAMS = (8, 20, 12, 25, 0.4, 1.2, 3, 8)


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
# RATIO CALCULATORS
# ═══════════════════════════════════════════════════════════════════

def _pe_ratio(close, eps):
    return _div(close, eps, require_positive_denom=True)

def _pb_ratio(close, bvps):
    return _div(close, bvps, require_positive_denom=True)

def _roe(net_profit, total_equity):
    return _div(net_profit, total_equity, require_positive_denom=True, scale=100.0)

def _debt_to_equity(total_debt, total_equity):
    return _div(total_debt, total_equity, require_positive_denom=True)

def _dividend_yield(annual_dps, close):
    return _div(annual_dps, close, require_positive_denom=True, scale=100.0)

def _net_profit_margin(net_profit, revenue):
    return _div(net_profit, revenue, require_positive_denom=True, scale=100.0)

def _payout_ratio(annual_dps, eps):
    return _div(annual_dps, eps, require_positive_denom=True, scale=100.0)


# ═══════════════════════════════════════════════════════════════════
# DATA QUALITY SCORER
# ═══════════════════════════════════════════════════════════════════

def _data_quality_score(has_financials, has_dividend, has_price, eps, total_equity) -> int:
    score = 100
    if not has_financials: score -= _DQ_MISSING_FINANCIALS
    if not has_dividend:   score -= _DQ_MISSING_DIVIDENDS
    if not has_price:      score -= _DQ_MISSING_PRICE
    try:
        if eps is None or np.isnan(float(eps)) or float(eps) <= 0:
            score -= _DQ_NEGATIVE_EPS
    except (TypeError, ValueError):
        score -= _DQ_NEGATIVE_EPS
    try:
        if total_equity is None or np.isnan(float(total_equity)) or float(total_equity) <= 0:
            score -= _DQ_NEGATIVE_EQUITY
    except (TypeError, ValueError):
        score -= _DQ_NEGATIVE_EQUITY
    return max(0, min(100, score))


def _build_notes(has_financials, has_dividend, has_price, eps, total_equity) -> str:
    notes: list[str] = []
    if not has_financials: notes.append("Missing financial data")
    if not has_price:      notes.append("Missing latest price")
    if not has_dividend:   notes.append("Missing dividend data")
    try:
        if eps is not None and not np.isnan(float(eps)) and float(eps) <= 0:
            notes.append("Negative EPS")
    except (TypeError, ValueError):
        notes.append("Invalid EPS")
    try:
        if total_equity is not None and not np.isnan(float(total_equity)) and float(total_equity) <= 0:
            notes.append("Negative equity")
    except (TypeError, ValueError):
        notes.append("Invalid equity")
    return " | ".join(notes) if notes else "Complete data"


# ═══════════════════════════════════════════════════════════════════
# SINGLE-ROW BUILDER
# ═══════════════════════════════════════════════════════════════════

def _build_row(ticker, fin_row, div_row, price_row) -> dict:
    has_financials = fin_row is not None
    has_dividend   = div_row is not None
    has_price      = price_row is not None

    def _get(row, col):
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

    annual_dps = _get(div_row, "annual_dps")
    if annual_dps is None:
        annual_dps = 0.0

    pe_ratio          = _pe_ratio(latest_close, eps)
    pb_ratio          = _pb_ratio(latest_close, book_value_per_share)
    roe               = _roe(net_profit, total_equity)
    debt_to_equity    = _debt_to_equity(total_debt, total_equity)
    dividend_yield    = _dividend_yield(annual_dps, latest_close)
    net_profit_margin = _net_profit_margin(net_profit, revenue)
    payout_ratio      = _payout_ratio(annual_dps, eps)

    dq_score = _data_quality_score(has_financials, has_dividend, has_price, eps, total_equity)
    notes    = _build_notes(has_financials, has_dividend, has_price, eps, total_equity)

    return {
        "ticker": ticker, "fiscal_year": fiscal_year, "latest_close": latest_close,
        "revenue": revenue, "net_profit": net_profit, "total_assets": total_assets,
        "total_equity": total_equity, "total_debt": total_debt, "eps": eps,
        "book_value_per_share": book_value_per_share,
        "shares_outstanding": shares_outstanding,
        "annual_dividend_per_share": annual_dps,
        "pe_ratio": pe_ratio, "pb_ratio": pb_ratio, "roe": roe,
        "debt_to_equity": debt_to_equity, "dividend_yield": dividend_yield,
        "net_profit_margin": net_profit_margin, "payout_ratio": payout_ratio,
        "data_quality_score": dq_score, "fundamental_notes": notes,
    }


# ═══════════════════════════════════════════════════════════════════
# SYNTHETIC FALLBACK
# ═══════════════════════════════════════════════════════════════════

def _generate_synthetic_fundamentals(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate plausible synthetic fundamental ratios for every ticker
    in price_df.  Called when the DB has no financial statement data.

    Uses a fixed RNG seed so results are reproducible across restarts.
    """
    rng = np.random.default_rng(_SYNTH_SEED)

    # Try to get universe for sector info
    sector_map: dict[str, str] = {}
    try:
        from config.stock_universe import PSX40_UNIVERSE  # type: ignore
        for entry in PSX40_UNIVERSE:
            if isinstance(entry, dict):
                sector_map[entry.get("ticker", "")] = entry.get("sector", "")
    except Exception:
        pass

    rows: list[dict] = []
    for _, price_row in price_df.iterrows():
        ticker = str(price_row["ticker"])
        close  = float(price_row["close"]) if "close" in price_row.index else None

        sector = sector_map.get(ticker, "")
        pe_lo, pe_hi, roe_lo, roe_hi, de_lo, de_hi, dy_lo, dy_hi = (
            _SECTOR_PARAMS.get(sector, _DEFAULT_PARAMS)
        )

        pe  = round(rng.uniform(pe_lo, pe_hi), 2)
        roe = round(rng.uniform(roe_lo, roe_hi), 2)      # percent
        de  = round(rng.uniform(de_lo, de_hi), 2)
        dy  = round(rng.uniform(dy_lo, dy_hi), 2)        # percent
        npm = round(rng.uniform(5, 25), 2)               # net profit margin %
        pb  = round(rng.uniform(0.8, 4.0), 2)

        eps  = round(close / pe, 2) if (close and pe) else None
        bvps = round(close / pb, 2) if (close and pb) else None
        annual_dps = round((dy / 100) * close, 2) if (close and dy) else 0.0
        payout = round(_payout_ratio(annual_dps, eps) or 0, 2)

        rows.append({
            "ticker":                   ticker,
            "fiscal_year":              2024,
            "latest_close":             close,
            "revenue":                  None,
            "net_profit":               None,
            "total_assets":             None,
            "total_equity":             None,
            "total_debt":               None,
            "eps":                      eps,
            "book_value_per_share":     bvps,
            "shares_outstanding":       None,
            "annual_dividend_per_share": annual_dps,
            "pe_ratio":                 pe,
            "pb_ratio":                 pb,
            "roe":                      roe,
            "debt_to_equity":           de,
            "dividend_yield":           dy,
            "net_profit_margin":        npm,
            "payout_ratio":             payout,
            "data_quality_score":       55,   # partial — synthetic
            "fundamental_notes":        "Synthetic data (no DB financials)",
        })

    result = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    print(
        f"[fundamental] Synthetic fallback — generated ratios for "
        f"{len(result)} ticker(s)."
    )
    return result


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════

def build_fundamental_metrics() -> pd.DataFrame:
    """
    Load raw data from SQLite and return a tidy DataFrame of
    fundamental ratios — one row per ticker.

    Falls back to synthetic data when DB has no financial statements,
    so the dashboard always shows meaningful values.
    """

    # ── Load from DB ──────────────────────────────────────────────
    try:
        fin_df = load_latest_financials()
    except Exception as exc:
        print(f"[fundamental] Warning: could not load financials — {exc}")
        fin_df = pd.DataFrame()

    try:
        div_df = load_annual_dividends()
    except Exception as exc:
        print(f"[fundamental] Warning: could not load dividends — {exc}")
        div_df = pd.DataFrame()

    try:
        price_df = load_latest_prices()
    except Exception as exc:
        print(f"[fundamental] Warning: could not load prices — {exc}")
        price_df = pd.DataFrame()

    # ── Synthetic fallback when no financial data exists ──────────
    if fin_df.empty and div_df.empty:
        if price_df.empty:
            print("[fundamental] No data in any source — returning empty.")
            return pd.DataFrame()
        print(
            "[fundamental] No financial/dividend data in DB — "
            "using synthetic fallback."
        )
        return _generate_synthetic_fundamentals(price_df)

    # ── Normal path — real DB data ────────────────────────────────

    # Reduce dividend table to latest year per ticker
    if not div_df.empty and "fiscal_year" in div_df.columns:
        div_latest = (
            div_df
            .sort_values(["ticker", "fiscal_year"], ascending=[True, False])
            .groupby("ticker", as_index=False)
            .first()
        )
    else:
        div_latest = pd.DataFrame(columns=["ticker", "annual_dps"])

    fin_lookup   = {r["ticker"]: r for _, r in fin_df.iterrows()} if not fin_df.empty else {}
    div_lookup   = {r["ticker"]: r for _, r in div_latest.iterrows()} if not div_latest.empty else {}
    price_lookup = {r["ticker"]: r for _, r in price_df.iterrows()} if not price_df.empty else {}

    all_tickers = sorted(
        set(fin_lookup.keys()) | set(div_lookup.keys()) | set(price_lookup.keys())
    )

    if not all_tickers:
        print("[fundamental] No tickers found in any data source.")
        return pd.DataFrame()

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
            print(f"[fundamental] Skipping {ticker} due to error: {exc}")
            continue

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

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
    result = result[ordered_cols].sort_values("ticker").reset_index(drop=True)

    print(
        f"[fundamental] Built metrics for {len(result)} ticker(s) | "
        f"avg quality score: {result['data_quality_score'].mean():.1f}/100"
    )
    return result


# ═══════════════════════════════════════════════════════════════════
# SINGLE-TICKER HELPER
# ═══════════════════════════════════════════════════════════════════

def get_fundamental_for_ticker(ticker: str) -> dict:
    ticker = ticker.strip().upper()
    df     = build_fundamental_metrics()
    if df.empty:
        return {}
    row = df[df["ticker"] == ticker]
    if row.empty:
        return {}
    return {
        k: (None if (isinstance(v, float) and np.isnan(v)) else v)
        for k, v in row.iloc[0].to_dict().items()
    }
