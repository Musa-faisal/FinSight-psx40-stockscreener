"""
csv_loader.py
-------------
Utility to load OHLCV data from a user-supplied CSV file into the database.
Expected CSV columns (case-insensitive):
    ticker, date, open, high, low, close, volume

This module is a placeholder for Phase 1 — real PSX scraping comes later.
"""

import pandas as pd
from pathlib import Path

# Canonical column names after normalisation
REQUIRED_COLUMNS = {"ticker", "date", "open", "high", "low", "close", "volume"}

# Accepted date formats (tried in order)
DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"]


def _parse_dates(series: pd.Series) -> pd.Series:
    """Try each DATE_FORMATS pattern until one succeeds."""
    for fmt in DATE_FORMATS:
        try:
            return pd.to_datetime(series, format=fmt)
        except (ValueError, TypeError):
            continue
    # Fallback: let pandas infer
    return pd.to_datetime(series, infer_datetime_format=True)


def load_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Read a CSV file and return a clean, validated OHLCV DataFrame.

    Returns
    -------
    pd.DataFrame with columns: ticker, date, open, high, low, close, volume
    Raises ValueError if required columns are missing.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    # Normalise column names: strip whitespace, lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Keep only required columns
    df = df[list(REQUIRED_COLUMNS)].copy()

    # Clean types
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["date"]   = _parse_dates(df["date"])
    df["date"]   = df["date"].dt.date  # store as Python date, not datetime

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")

    # Drop rows where critical fields are null
    before = len(df)
    df.dropna(subset=["ticker", "date", "close"], inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"  [csv_loader] Dropped {dropped} rows with null ticker/date/close")

    df.sort_values(["ticker", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"  [csv_loader] Loaded {len(df)} rows for {df['ticker'].nunique()} ticker(s) from {path.name}")
    return df
