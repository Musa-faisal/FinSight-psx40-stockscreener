"""
sector_benchmark.py
-------------------
Phase 7 — Sector Benchmarking for PSX40 Screener.

Public API
----------
    build_sector_metrics(df) -> pd.DataFrame
        Accepts a scored stock DataFrame.
        Returns one row per sector with all sector-level statistics.

    apply_sector_benchmarks(df) -> pd.DataFrame
        Accepts a scored stock DataFrame.
        Merges sector metrics back onto every stock row and adds:
          - sector_avg_*, sector_median_pe, sector_stock_count
          - *_vs_sector      (relative difference, fractional)
          - *_sector_z       (z-score vs sector)
          - sector_value_label

    build_sector_benchmarks() -> tuple[pd.DataFrame, pd.DataFrame]
        Notebook helper — runs full engine pipeline and returns
        (sector_summary, stock_sector).

Safety rules
------------
    - Sectors with < 2 valid stocks for a metric → NaN avg / std / median.
    - Zero std → z-score is 0.0, not a crash.
    - Zero / NaN sector average → vs-sector returns NaN, not a crash.
    - All missing values are handled safely — screener never crashes.
    - Existing benchmark columns are dropped before re-running to avoid
      duplicate _x/_y suffixes on repeated calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Column mapping: (stock_column_name, short_key_for_metric_names) ─

_METRIC_COLS: list[tuple[str, str]] = [
    ("pe_ratio",        "pe"),
    ("pb_ratio",        "pb"),
    ("roe",             "roe"),
    ("dividend_yield",  "dividend_yield"),
    ("composite_score", "composite_score"),
    ("risk_score",      "risk_score"),
]

# Minimum number of valid (non-NaN) stocks in a sector before
# statistics are considered reliable.
_MIN_VALID: int = 2


# ════════════════════════════════════════════════════════════════════
# SCALAR HELPERS
# ════════════════════════════════════════════════════════════════════

def _safe_rel(stock_val, sector_avg) -> float | None:
    """
    Relative difference: (stock - sector_avg) / |sector_avg|.

    Returns None if either input is NaN, None, or sector_avg ≈ 0.
    The result is fractional (e.g. 0.15 means +15 %).
    """
    try:
        sv = float(stock_val)
        sa = float(sector_avg)
    except (TypeError, ValueError):
        return None

    if np.isnan(sv) or np.isnan(sa):
        return None
    if abs(sa) < 1e-9:
        return None

    result = (sv - sa) / abs(sa)
    if np.isinf(result) or np.isnan(result):
        return None
    return round(result, 6)


def _safe_z(stock_val, sector_avg, sector_std) -> float:
    """
    Z-score: (stock - sector_avg) / sector_std.

    Returns:
      - 0.0  if sector_std is zero (all sector peers identical).
      - NaN  if any input is NaN / missing.
    """
    try:
        sv  = float(stock_val)
        sa  = float(sector_avg)
        ss  = float(sector_std)
    except (TypeError, ValueError):
        return np.nan

    if np.isnan(sv) or np.isnan(sa) or np.isnan(ss):
        return np.nan
    if abs(ss) < 1e-9:
        return 0.0

    result = (sv - sa) / ss
    if np.isinf(result) or np.isnan(result):
        return np.nan
    return round(result, 6)


def _sector_value_label(
    composite_z: float | None,
    pe_z: float | None,
    sector_valid_composite: int,
) -> str:
    """
    Assign a human-readable benchmark label to a stock based on how
    its composite_score z-score (and optionally pe z-score) compare
    to its sector peers.

    Labels (in priority order)
    --------------------------
    "Insufficient sector data"  — fewer than _MIN_VALID valid peers
    "Sector Value Leader"       — composite_z >= 1.0
    "Attractive vs Sector"      — composite_z >= 0.25
    "In Line with Sector"       — |composite_z| < 0.25
    "Expensive vs Sector"       — composite_z <= -0.25 AND pe_z >= 0.5
    "Weak vs Sector"            — composite_z <= -0.25 (all other cases)
    """
    if sector_valid_composite < _MIN_VALID:
        return "Insufficient sector data"

    # Safely convert z-scores
    try:
        cz = float(composite_z)
        if np.isnan(cz):
            return "Insufficient sector data"
    except (TypeError, ValueError):
        return "Insufficient sector data"

    try:
        pz = float(pe_z)
        pz_ok = not np.isnan(pz)
    except (TypeError, ValueError):
        pz_ok = False
        pz    = np.nan

    if cz >= 1.0:
        return "Sector Value Leader"
    if cz >= 0.25:
        return "Attractive vs Sector"
    if cz > -0.25:
        return "In Line with Sector"
    # Below sector average
    if pz_ok and pz >= 0.5:
        return "Expensive vs Sector"
    return "Weak vs Sector"


# ════════════════════════════════════════════════════════════════════
# PUBLIC: build_sector_metrics
# ════════════════════════════════════════════════════════════════════

def build_sector_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a scored stock DataFrame and returns one row per sector
    containing all required sector-level statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Scored stock DataFrame — must contain a 'sector' column and
        the relevant metric columns (pe_ratio, pb_ratio, etc.).

    Returns
    -------
    pd.DataFrame
        One row per sector.  Columns include:
          sector, sector_stock_count,
          sector_valid_{key}_count,
          sector_avg_{key},  sector_std_{key}   (for all six metrics)
          sector_median_pe                       (PE only)

        Stats are NaN for any metric where the sector has < _MIN_VALID
        valid (non-NaN) values.
    """
    if df.empty or "sector" not in df.columns:
        return pd.DataFrame()

    df = df.copy()

    # Coerce all metric columns to numeric before grouping
    for stock_col, _ in _METRIC_COLS:
        if stock_col in df.columns:
            df[stock_col] = pd.to_numeric(df[stock_col], errors="coerce")

    rows: list[dict] = []

    for sector, grp in df.groupby("sector", sort=True):
        row: dict = {
            "sector":             sector,
            "sector_stock_count": int(len(grp)),
        }

        for stock_col, key in _METRIC_COLS:
            series: pd.Series = (
                grp[stock_col].dropna()
                if stock_col in grp.columns
                else pd.Series([], dtype=float)
            )
            count = int(len(series))

            row[f"sector_valid_{key}_count"] = count

            if count >= _MIN_VALID:
                row[f"sector_avg_{key}"] = round(float(series.mean()),       6)
                row[f"sector_std_{key}"] = round(float(series.std(ddof=1)),  6)
            else:
                row[f"sector_avg_{key}"] = np.nan
                row[f"sector_std_{key}"] = np.nan

            # Median is only produced for PE (per spec)
            if key == "pe":
                row["sector_median_pe"] = (
                    round(float(series.median()), 6)
                    if count >= _MIN_VALID
                    else np.nan
                )

        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════
# PUBLIC: apply_sector_benchmarks
# ════════════════════════════════════════════════════════════════════

def apply_sector_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges sector-level metrics onto every stock row and adds
    per-stock relative and z-score benchmark columns.

    Parameters
    ----------
    df : pd.DataFrame
        Full scored stock DataFrame (after three-pillar scoring).

    Returns
    -------
    pd.DataFrame
        Same rows as input, with additional columns:
          sector_stock_count
          sector_avg_{key}          for each of 6 metrics
          sector_median_pe
          {key}_vs_sector           relative difference (fractional)
          {key}_sector_z            z-score vs sector
          sector_value_label

    Notes
    -----
    - Pre-existing benchmark columns are dropped before re-adding to
      prevent duplicate _x/_y columns on repeated calls.
    - The sector 'sector' column from the input is preserved; sector_df
      is merged on it, not replacing it.
    """
    if df.empty or "sector" not in df.columns:
        return df

    # ── Drop any pre-existing benchmark columns to avoid duplicates ──
    drop_cols = [
        c for c in df.columns
        if (
            c.startswith("sector_avg_")
            or c.startswith("sector_median_")
            or c.startswith("sector_std_")
            or c.startswith("sector_valid_")
            or c in ("sector_stock_count", "sector_value_label")
            or c.endswith("_vs_sector")
            or c.endswith("_sector_z")
        )
    ]
    df = df.drop(columns=drop_cols, errors="ignore").copy()

    # ── Build sector summary ─────────────────────────────────────────
    sector_df = build_sector_metrics(df)
    if sector_df.empty:
        return df

    # ── Merge sector stats onto each stock row ───────────────────────
    df = df.merge(sector_df, on="sector", how="left")

    # Coerce all numeric stock metric columns after merge
    for stock_col, _ in _METRIC_COLS:
        if stock_col in df.columns:
            df[stock_col] = pd.to_numeric(df[stock_col], errors="coerce")

    # ── Relative differences (*_vs_sector) ──────────────────────────
    for stock_col, key in _METRIC_COLS:
        avg_col = f"sector_avg_{key}"
        vs_col  = f"{key}_vs_sector"

        if stock_col in df.columns and avg_col in df.columns:
            df[avg_col] = pd.to_numeric(df[avg_col], errors="coerce")
            df[vs_col]  = [
                _safe_rel(sv, sa)
                for sv, sa in zip(df[stock_col], df[avg_col])
            ]
        else:
            df[vs_col] = np.nan

    # ── Z-scores (*_sector_z) ────────────────────────────────────────
    for stock_col, key in _METRIC_COLS:
        avg_col = f"sector_avg_{key}"
        std_col = f"sector_std_{key}"
        z_col   = f"{key}_sector_z"

        if (
            stock_col in df.columns
            and avg_col in df.columns
            and std_col in df.columns
        ):
            df[avg_col] = pd.to_numeric(df[avg_col], errors="coerce")
            df[std_col] = pd.to_numeric(df[std_col], errors="coerce")
            df[z_col]   = [
                _safe_z(sv, sa, ss)
                for sv, sa, ss in zip(df[stock_col], df[avg_col], df[std_col])
            ]
        else:
            df[z_col] = np.nan

    # ── Sector value labels ──────────────────────────────────────────
    def _label_row(row) -> str:
        valid_composite = int(
            row.get("sector_valid_composite_score_count") or 0
        )
        return _sector_value_label(
            composite_z             = row.get("composite_score_sector_z"),
            pe_z                    = row.get("pe_sector_z"),
            sector_valid_composite  = valid_composite,
        )

    df["sector_value_label"] = df.apply(_label_row, axis=1)

    n_labelled = int((df["sector_value_label"] != "Insufficient sector data").sum())
    print(
        f"[sector_benchmark] Applied benchmarks — "
        f"{len(df)} stocks across "
        f"{df['sector'].nunique()} sectors | "
        f"{n_labelled} stocks labelled."
    )

    return df.reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════
# PUBLIC: build_sector_benchmarks  (notebook / test helper)
# ════════════════════════════════════════════════════════════════════

def build_sector_benchmarks() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Notebook-friendly helper that runs the full engine pipeline and
    returns benchmark results without needing to manage the engine
    directly.

    Returns
    -------
    sector_summary : pd.DataFrame
        One row per sector — output of build_sector_metrics().
    stock_sector : pd.DataFrame
        Full engine screener DataFrame with all sector benchmark
        columns already applied (not raw unmerged data).

    Usage
    -----
    >>> from src.analysis.sector_benchmark import build_sector_benchmarks
    >>> sector_summary, stock_sector = build_sector_benchmarks()
    >>> print(sector_summary)
    >>> print(stock_sector[["ticker", "sector", "composite_score_vs_sector"]].head())
    """
    # Lazy import to avoid circular dependencies
    from src.screener.engine import ScreenerEngine  # noqa: PLC0415

    engine = ScreenerEngine()
    engine.run()

    # engine.get_screener_df() already contains sector benchmarks —
    # they are applied inside engine.run() via _run_sector_benchmarks()
    stock_sector   = engine.get_screener_df()
    sector_summary = build_sector_metrics(stock_sector)

    return sector_summary, stock_sector
