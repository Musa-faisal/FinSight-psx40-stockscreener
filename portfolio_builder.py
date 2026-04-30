"""
portfolio_builder.py
--------------------
Phase 10 — Portfolio Builder for PSX40 Screener.

Constructs weighted stock portfolios from the screener DataFrame
produced by ScreenerEngine.get_screener_df().

Public API
----------
    select_top_stocks(df, top_n, min_composite, max_risk, sectors)
    build_equal_weight_portfolio(selected_df)
    build_inverse_volatility_portfolio(selected_df)
    build_portfolio(df, top_n, weighting, min_composite, max_risk, sectors)
    summarize_portfolio(portfolio_df) -> dict

Rules
-----
    - Input DataFrames are never mutated.
    - Missing columns never raise exceptions.
    - All numeric reads use pd.to_numeric(errors="coerce").
    - Weight column is always named "weight".
    - weighting_method column is always present.
    - Weights always sum to 1.0 (within floating-point tolerance).
    - Fallback to equal weight when volatility is unusable.
    - Returns empty DataFrames (not exceptions) when no rows qualify.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

_WEIGHTING_METHODS = ("equal", "inverse_volatility")

_VOL_COLS   = ("volatility_30d", "volatility")   # preference order
_SCORE_COL  = "composite_score"
_RISK_COL   = "risk_score"
_SECTOR_COL = "sector"
_TICKER_COL = "ticker"


# ═══════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════

def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a numeric Series for *col*, NaN where missing/invalid."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _has(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def _safe_mean(df: pd.DataFrame, col: str) -> float | None:
    """Return mean of *col* or None when column is absent / all-NaN."""
    s = _num(df, col).dropna()
    return round(float(s.mean()), 2) if not s.empty else None


def _normalise_weights(raw: pd.Series) -> pd.Series:
    """
    Normalise *raw* so weights sum to exactly 1.0.
    Falls back to equal weight if sum is zero or all-NaN.
    """
    total = raw.sum()
    if total <= 0 or np.isnan(total):
        n = len(raw)
        return pd.Series(1.0 / n, index=raw.index) if n > 0 else raw
    return raw / total


# ═══════════════════════════════════════════════════════════════════
# 1. SELECT TOP STOCKS
# ═══════════════════════════════════════════════════════════════════

def select_top_stocks(
    df: pd.DataFrame,
    top_n: int = 10,
    min_composite: float | None = None,
    max_risk: float | None = None,
    sectors: list[str] | None = None,
) -> pd.DataFrame:
    """
    Filter and rank the screener DataFrame to produce a candidate
    stock selection for the portfolio.

    Parameters
    ----------
    df            : full screener DataFrame from ScreenerEngine
    top_n         : maximum number of stocks to select
    min_composite : minimum composite_score to include (optional)
    max_risk      : maximum risk_score to include (optional)
    sectors       : restrict to these sector names (optional)

    Returns
    -------
    pd.DataFrame — top_n rows sorted by composite_score descending.
                   Empty DataFrame when no rows pass the filters.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # ── Sector filter ─────────────────────────────────────────────
    if sectors and _has(out, _SECTOR_COL):
        out = out[out[_SECTOR_COL].isin(sectors)]

    # ── Composite score filter ────────────────────────────────────
    if min_composite is not None and _has(out, _SCORE_COL):
        scores = _num(out, _SCORE_COL)
        out    = out[scores.fillna(0) >= min_composite]

    # ── Risk score filter ─────────────────────────────────────────
    if max_risk is not None and _has(out, _RISK_COL):
        risks = _num(out, _RISK_COL)
        out   = out[risks.fillna(100) <= max_risk]

    if out.empty:
        return pd.DataFrame()

    # ── Sort by composite score descending ────────────────────────
    if _has(out, _SCORE_COL):
        out = out.sort_values(
            by=_SCORE_COL,
            ascending=False,
            na_position="last",
        )

    return out.head(top_n).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# 2. EQUAL WEIGHT PORTFOLIO
# ═══════════════════════════════════════════════════════════════════

def build_equal_weight_portfolio(selected_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign equal weight to every stock in *selected_df*.

    Parameters
    ----------
    selected_df : output of select_top_stocks() or any filtered df

    Returns
    -------
    pd.DataFrame with "weight" and "weighting_method" columns added.
    Weights sum to 1.0.  Empty DataFrame returned safely.
    """
    if selected_df is None or selected_df.empty:
        return pd.DataFrame()

    out    = selected_df.copy()
    n      = len(out)
    out["weight"]           = round(1.0 / n, 10)
    out["weighting_method"] = "equal"

    return out.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# 3. INVERSE VOLATILITY PORTFOLIO
# ═══════════════════════════════════════════════════════════════════

def build_inverse_volatility_portfolio(selected_df: pd.DataFrame) -> pd.DataFrame:
    """
    Weight each stock by the inverse of its volatility so that
    lower-volatility stocks receive a larger allocation.

    Volatility column priority: volatility_30d → volatility.
    Falls back to equal weight when:
      - no volatility column exists
      - all volatility values are NaN / zero

    Parameters
    ----------
    selected_df : output of select_top_stocks()

    Returns
    -------
    pd.DataFrame with "weight" and "weighting_method" columns added.
    Weights sum to 1.0.  Empty DataFrame returned safely.
    """
    if selected_df is None or selected_df.empty:
        return pd.DataFrame()

    out = selected_df.copy()

    # ── Locate best available volatility column ───────────────────
    vol_col = None
    for col in _VOL_COLS:
        if _has(out, col):
            vol_col = col
            break

    # ── Attempt inverse-vol weighting ────────────────────────────
    used_method = "equal"

    if vol_col is not None:
        vol = _num(out, vol_col)

        # Replace zero / negative / NaN with NaN so they get equal share
        vol = vol.where(vol > 0, other=np.nan)
        valid_count = vol.notna().sum()

        if valid_count > 0:
            # Stocks with missing vol get the median inverse-vol weight
            inv_vol       = 1.0 / vol
            median_inv    = inv_vol.median()
            inv_vol       = inv_vol.fillna(median_inv)

            raw_weights   = _normalise_weights(inv_vol)
            out["weight"] = raw_weights.round(10).values
            used_method   = "inverse_volatility"

    # ── Fallback to equal weight ──────────────────────────────────
    if used_method == "equal":
        n             = len(out)
        out["weight"] = round(1.0 / n, 10)

    out["weighting_method"] = used_method

    # ── Enforce exact sum = 1.0 by adjusting the largest weight ──
    weight_series  = pd.to_numeric(out["weight"], errors="coerce").fillna(0)
    residual       = 1.0 - weight_series.sum()
    largest_idx    = weight_series.idxmax()
    out.at[largest_idx, "weight"] = float(out.at[largest_idx, "weight"]) + residual

    return out.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# 4. BUILD PORTFOLIO  (master function)
# ═══════════════════════════════════════════════════════════════════

def build_portfolio(
    df: pd.DataFrame,
    top_n: int = 10,
    weighting: str = "equal",
    min_composite: float | None = None,
    max_risk: float | None = None,
    sectors: list[str] | None = None,
) -> pd.DataFrame:
    """
    Select stocks and assign weights in one call.

    Parameters
    ----------
    df            : full screener DataFrame from ScreenerEngine
    top_n         : number of stocks to hold
    weighting     : "equal" or "inverse_volatility"
    min_composite : minimum composite_score filter (optional)
    max_risk      : maximum risk_score filter (optional)
    sectors       : restrict to these sectors (optional)

    Returns
    -------
    pd.DataFrame with "weight" and "weighting_method" columns.
    Weights sum to 1.0.  Empty DataFrame when no stocks qualify.

    Raises
    ------
    ValueError if *weighting* is not a supported method.
    """
    if weighting not in _WEIGHTING_METHODS:
        raise ValueError(
            f"Unsupported weighting method: '{weighting}'. "
            f"Choose from: {_WEIGHTING_METHODS}"
        )

    # ── Step 1: selection ─────────────────────────────────────────
    selected = select_top_stocks(
        df,
        top_n=top_n,
        min_composite=min_composite,
        max_risk=max_risk,
        sectors=sectors,
    )

    if selected.empty:
        return pd.DataFrame()

    # ── Step 2: weighting ─────────────────────────────────────────
    if weighting == "equal":
        return build_equal_weight_portfolio(selected)
    else:
        return build_inverse_volatility_portfolio(selected)


# ═══════════════════════════════════════════════════════════════════
# 5. PORTFOLIO SUMMARY
# ═══════════════════════════════════════════════════════════════════

def summarize_portfolio(portfolio_df: pd.DataFrame) -> dict:
    """
    Produce a summary dict for a built portfolio DataFrame.

    Parameters
    ----------
    portfolio_df : output of build_portfolio() or any of the
                   build_*_portfolio() functions

    Returns
    -------
    dict with keys:
        holdings          (int)
        weight_sum        (float)   — should be ~1.0
        weighting_method  (str | None)
        avg_composite_score   (float | None)
        avg_technical_score   (float | None)
        avg_fundamental_score (float | None)
        avg_risk_score        (float | None)
        avg_dividend_yield    (float | None)
        top_holding           (str | None)   — ticker with highest weight
        sector_allocation     (dict)          — sector → summed weight
    """
    empty_summary = {
        "holdings":             0,
        "weight_sum":           0.0,
        "weighting_method":     None,
        "avg_composite_score":  None,
        "avg_technical_score":  None,
        "avg_fundamental_score": None,
        "avg_risk_score":       None,
        "avg_dividend_yield":   None,
        "top_holding":          None,
        "sector_allocation":    {},
    }

    if portfolio_df is None or portfolio_df.empty:
        return empty_summary

    df = portfolio_df.copy()

    # ── Weight column ─────────────────────────────────────────────
    weights   = _num(df, "weight").fillna(0)
    weight_sum = round(float(weights.sum()), 6)

    # ── Weighting method ─────────────────────────────────────────
    if _has(df, "weighting_method") and not df["weighting_method"].isna().all():
        method = df["weighting_method"].dropna().iloc[0]
    else:
        method = None

    # ── Top holding (highest weight) ──────────────────────────────
    top_holding = None
    if _has(df, _TICKER_COL) and not weights.empty:
        top_idx     = weights.idxmax()
        top_holding = df.at[top_idx, _TICKER_COL] \
            if _has(df, _TICKER_COL) else None

    # ── Weighted average scores ───────────────────────────────────
    def _wavg(col: str) -> float | None:
        """Compute weight-averaged score for *col*."""
        if not _has(df, col):
            return None
        vals   = _num(df, col)
        valid  = vals.notna()
        if not valid.any():
            return None
        w_sum  = weights[valid].sum()
        if w_sum <= 0:
            return None
        return round(float((vals[valid] * weights[valid]).sum() / w_sum), 2)

    # ── Simple (non-weighted) average for non-score columns ──────
    def _savg(col: str) -> float | None:
        return _safe_mean(df, col)

    # ── Sector allocation — sum of weights per sector ─────────────
    sector_allocation: dict[str, float] = {}
    if _has(df, _SECTOR_COL):
        for sector, group in df.groupby(_SECTOR_COL, sort=False):
            sec_weight = _num(group, "weight").fillna(0).sum()
            if sector and not pd.isna(sector):
                sector_allocation[str(sector)] = round(float(sec_weight), 4)

    return {
        "holdings":              len(df),
        "weight_sum":            weight_sum,
        "weighting_method":      method,
        "avg_composite_score":   _wavg("composite_score"),
        "avg_technical_score":   _wavg("technical_score"),
        "avg_fundamental_score": _wavg("fundamental_score"),
        "avg_risk_score":        _wavg("risk_score"),
        "avg_dividend_yield":    _savg("dividend_yield"),
        "top_holding":           top_holding,
        "sector_allocation":     sector_allocation,
    }