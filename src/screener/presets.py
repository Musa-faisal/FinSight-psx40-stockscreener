"""
presets.py
----------
Phase 8 — Preset Screeners for PSX40 Screener.

Reusable preset filters that slice the final engine DataFrame into
useful investment groups.  Safe to import and test in Jupyter Notebook.

Public API
----------
    PRESET_NAMES
    get_preset_names() -> list[str]
    apply_preset(df: pd.DataFrame, preset_name: str) -> pd.DataFrame
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# PRESET REGISTRY
# ═══════════════════════════════════════════════════════════════════

PRESET_NAMES = [
    "Strong Buy Candidates",
    "Undervalued Quality",
    "Dividend Picks",
    "Momentum Leaders",
    "Low Risk Blue Chips",
    "Oversold Bounce",
    "Avoid / High Risk",
    "Sector Relative Value",
]


def get_preset_names() -> list[str]:
    """Return a copy of the preset name list."""
    return list(PRESET_NAMES)


# ═══════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════

def _has_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def _as_num(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a numeric Series (NaN where invalid) for *col*."""
    if _has_col(df, col):
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _verdict_col(df: pd.DataFrame) -> str | None:
    """Return the best-available verdict column name, or None."""
    if _has_col(df, "final_verdict"):
        return "final_verdict"
    if _has_col(df, "verdict"):
        return "verdict"
    return None


# ═══════════════════════════════════════════════════════════════════
# PRESET FILTERS
# ═══════════════════════════════════════════════════════════════════

def _preset_strong_buy(df: pd.DataFrame) -> pd.DataFrame:
    vcol = _verdict_col(df)
    if vcol is None:
        return df.iloc[0:0]

    mask = df[vcol] == "Strong Buy"
    if _has_col(df, "composite_score"):
        mask = mask & (_as_num(df, "composite_score") >= 70)
    if _has_col(df, "risk_score"):
        mask = mask & (_as_num(df, "risk_score") <= 60)

    out = df[mask]
    sort_col = "composite_score" if _has_col(df, "composite_score") else vcol
    return out.sort_values(by=sort_col, ascending=False)


def _preset_undervalued_quality(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    if _has_col(df, "pe_vs_sector"):
        mask = mask & (_as_num(df, "pe_vs_sector") < 0)
    if _has_col(df, "pb_vs_sector"):
        mask = mask & (_as_num(df, "pb_vs_sector") < 0)
    if _has_col(df, "roe_vs_sector"):
        mask = mask & (_as_num(df, "roe_vs_sector") > 0)
    if _has_col(df, "fundamental_score"):
        mask = mask & (_as_num(df, "fundamental_score") >= 60)
    if _has_col(df, "composite_score"):
        mask = mask & (_as_num(df, "composite_score") >= 55)

    out = df[mask]
    sort_col = "composite_score" if _has_col(df, "composite_score") else (
        "fundamental_score" if _has_col(df, "fundamental_score") else df.columns[0]
    )
    return out.sort_values(by=sort_col, ascending=False)


def _preset_dividend_picks(df: pd.DataFrame) -> pd.DataFrame:
    if not _has_col(df, "dividend_yield"):
        return df.iloc[0:0]

    mask = _as_num(df, "dividend_yield") > 0
    if _has_col(df, "dividend_yield_vs_sector"):
        mask = mask & (_as_num(df, "dividend_yield_vs_sector") >= 0)
    if _has_col(df, "risk_score"):
        mask = mask & (_as_num(df, "risk_score") <= 65)
    if _has_col(df, "payout_ratio"):
        pr = _as_num(df, "payout_ratio")
        mask = mask & ((pr <= 100) | pr.isna())

    out = df[mask]
    sort_cols = []
    sort_asc = []
    if _has_col(df, "dividend_yield"):
        sort_cols.append("dividend_yield")
        sort_asc.append(False)
    if _has_col(df, "composite_score"):
        sort_cols.append("composite_score")
        sort_asc.append(False)
    if not sort_cols:
        sort_cols = [df.columns[0]]
        sort_asc = [False]
    return out.sort_values(by=sort_cols, ascending=sort_asc)


def _preset_momentum_leaders(df: pd.DataFrame) -> pd.DataFrame:
    if not _has_col(df, "technical_score"):
        return df.iloc[0:0]

    mask = _as_num(df, "technical_score") >= 65
    if _has_col(df, "return_1m"):
        mask = mask & (_as_num(df, "return_1m") > 0)
    if _has_col(df, "return_3m"):
        mask = mask & (_as_num(df, "return_3m") > 0)
    if _has_col(df, "latest_close") and _has_col(df, "sma_50"):
        mask = mask & (_as_num(df, "latest_close") > _as_num(df, "sma_50"))

    out = df[mask]
    sort_col = "technical_score" if _has_col(df, "technical_score") else df.columns[0]
    return out.sort_values(by=sort_col, ascending=False)


def _preset_low_risk_blue_chips(df: pd.DataFrame) -> pd.DataFrame:
    if not _has_col(df, "risk_score"):
        return df.iloc[0:0]

    mask = _as_num(df, "risk_score") <= 40
    if _has_col(df, "composite_score"):
        mask = mask & (_as_num(df, "composite_score") >= 55)

    # Liquidity filter (optional)
    liq_mask = pd.Series(False, index=df.index)
    if _has_col(df, "avg_volume_20d"):
        liq_mask = liq_mask | (_as_num(df, "avg_volume_20d") > 0)
    if _has_col(df, "volume_ratio"):
        liq_mask = liq_mask | (_as_num(df, "volume_ratio") > 0.3)
    if _has_col(df, "avg_volume_20d") or _has_col(df, "volume_ratio"):
        mask = mask & liq_mask

    out = df[mask]
    sort_cols = []
    sort_asc = []
    if _has_col(df, "risk_score"):
        sort_cols.append("risk_score")
        sort_asc.append(True)
    if _has_col(df, "composite_score"):
        sort_cols.append("composite_score")
        sort_asc.append(False)
    if not sort_cols:
        sort_cols = [df.columns[0]]
        sort_asc = [True]
    return out.sort_values(by=sort_cols, ascending=sort_asc)


def _preset_oversold_bounce(df: pd.DataFrame) -> pd.DataFrame:
    if not _has_col(df, "rsi_14"):
        return df.iloc[0:0]

    mask = _as_num(df, "rsi_14") < 35
    if _has_col(df, "technical_score"):
        mask = mask & (_as_num(df, "technical_score") >= 40)
    if _has_col(df, "risk_score"):
        mask = mask & (_as_num(df, "risk_score") <= 70)

    out = df[mask]
    sort_col = "rsi_14" if _has_col(df, "rsi_14") else df.columns[0]
    return out.sort_values(by=sort_col, ascending=True)


def _preset_avoid_high_risk(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)

    vcol = _verdict_col(df)
    if vcol is not None:
        mask = mask | (df[vcol] == "Avoid")
    if _has_col(df, "risk_score"):
        mask = mask | (_as_num(df, "risk_score") >= 75)
    if _has_col(df, "composite_score"):
        mask = mask | (_as_num(df, "composite_score") < 40)

    out = df[mask]
    sort_col = "risk_score" if _has_col(df, "risk_score") else (vcol if vcol else df.columns[0])
    return out.sort_values(by=sort_col, ascending=False)


def _preset_sector_relative_value(df: pd.DataFrame) -> pd.DataFrame:
    if not _has_col(df, "sector_value_label"):
        return df.iloc[0:0]

    mask = df["sector_value_label"].isin([
        "Sector Value Leader",
        "Attractive vs Sector",
        "Potential Sector Value",
        "Sector Outperformer",
    ])
    if _has_col(df, "composite_score_vs_sector"):
        mask = mask & (_as_num(df, "composite_score_vs_sector") > 0)
    if _has_col(df, "pe_vs_sector"):
        mask = mask & (_as_num(df, "pe_vs_sector") < 0)

    out = df[mask]
    sort_col = "composite_score_vs_sector" if _has_col(df, "composite_score_vs_sector") else df.columns[0]
    return out.sort_values(by=sort_col, ascending=False)


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════

_PRESET_FUNCS = {
    "Strong Buy Candidates":   _preset_strong_buy,
    "Undervalued Quality":     _preset_undervalued_quality,
    "Dividend Picks":          _preset_dividend_picks,
    "Momentum Leaders":        _preset_momentum_leaders,
    "Low Risk Blue Chips":     _preset_low_risk_blue_chips,
    "Oversold Bounce":         _preset_oversold_bounce,
    "Avoid / High Risk":       _preset_avoid_high_risk,
    "Sector Relative Value":   _preset_sector_relative_value,
}


def apply_preset(df: pd.DataFrame, preset_name: str) -> pd.DataFrame:
    """
    Apply a named preset filter to *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Full screener DataFrame (e.g. engine.get_screener_df()).
    preset_name : str
        One of the names in PRESET_NAMES.

    Returns
    -------
    pd.DataFrame
        Filtered, sorted result.  Empty if no rows match.
        Input DataFrame is never mutated.
    """
    if df.empty:
        return df.copy().reset_index(drop=True)

    if preset_name not in PRESET_NAMES:
        raise ValueError(
            f"Unknown preset '{preset_name}'. "
            f"Available presets: {', '.join(PRESET_NAMES)}"
        )

    out = _PRESET_FUNCS[preset_name](df).copy()
    return out.reset_index(drop=True)
