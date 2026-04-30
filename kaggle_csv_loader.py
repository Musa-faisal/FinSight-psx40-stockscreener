"""
kaggle_csv_loader.py
--------------------
Phase 2 — Load historical OHLCV data from CSV files stored in
data/raw/kaggle/

Confirmed working with:
    compiled_psx_historical_2017_2025.csv
    Columns: DATE, ticker, LDCP, OPEN, HIGH, LOW, CLOSE, CHANGE, CHANGE (%), VOLUME

Supports two formats
--------------------
1. Combined CSV  — one file, has a ticker/symbol column
2. Per-ticker CSV — one file per stock, ticker inferred from filename

Column normalisation
--------------------
All column names are lowercased and stripped before mapping.
Unrecognised or unwanted columns (LDCP, CHANGE, CHANGE (%)) are silently dropped.
"""

import re
from pathlib import Path

import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

KAGGLE_DIR = Path("data/raw/kaggle")

# Maps lowercased-stripped column name → canonical name
# Add any new aliases here without touching the rest of the code.
COLUMN_ALIASES: dict[str, str] = {
    # date
    "date":           "date",
    "trade_date":     "date",
    "tradedate":      "date",
    # ticker
    "ticker":         "ticker",
    "symbol":         "ticker",
    # open
    "open":           "open",
    # high
    "high":           "high",
    # low
    "low":            "low",
    # close
    "close":          "close",
    "adj close":      "close",
    "adj_close":      "close",
    "adjusted close": "close",
    # volume
    "volume":         "volume",
    "vol":            "volume",
}

# Columns we want in the final output — in this order
CANONICAL_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]

DATE_FORMATS = [
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%d-%b-%Y",
    "%d %b %Y",
    "%b %d, %Y",
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalise_column_name(col: str) -> str:
    """Lowercase and strip a single column name."""
    return col.strip().lower()


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename DataFrame columns to canonical names using COLUMN_ALIASES.
    Columns with no alias match are kept as-is (will be dropped later).
    Normalisation is case-insensitive and strips leading/trailing spaces.
    """
    rename_map = {}
    for col in df.columns:
        normalised = _normalise_column_name(col)
        if normalised in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[normalised]
    df = df.rename(columns=rename_map)
    return df


def _parse_date_column(series: pd.Series) -> pd.Series:
    """
    Try each DATE_FORMATS pattern in order.
    Fall back to pandas inference if none match.
    Returns a Series of Python date objects.
    """
    for fmt in DATE_FORMATS:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="raise")
            return parsed.dt.date
        except (ValueError, TypeError):
            continue
    # Final fallback
    parsed = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
    return parsed.dt.date


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Remove commas, strip spaces, convert to float."""
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )


def _clean_volume(series: pd.Series) -> pd.Series:
    """Remove commas, convert to integer. Nulls become 0."""
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype("int64")
    )


def _check_missing_canonical(df: pd.DataFrame, source: str) -> list[str]:
    """Return canonical columns that are absent from df after mapping."""
    return [c for c in CANONICAL_COLUMNS if c not in df.columns]


def _apply_types_and_clean(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Cast all canonical columns to their correct types and remove bad rows.
    Assumes df already has canonical column names.
    """
    # Ticker
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # Date
    df["date"] = _parse_date_column(df["date"])

    # OHLC prices
    for col in ["open", "high", "low", "close"]:
        df[col] = _clean_numeric(df[col])

    # Volume
    df["volume"] = _clean_volume(df["volume"])

    # Drop rows where ticker, date, or close is missing
    before = len(df)
    df = df.dropna(subset=["ticker", "date", "close"])
    dropped = before - len(df)
    if dropped:
        print(f"  [{source}] Dropped {dropped} rows with missing ticker / date / close")

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def _infer_ticker_from_filename(path: Path) -> str:
    """
    Extract ticker from filename stem.
    Examples:
      HBL.csv           → HBL
      hbl_daily.csv     → HBL
      PSO_2020_2024.csv → PSO
    """
    stem   = path.stem
    ticker = stem.split("_")[0].upper()
    ticker = re.sub(r"[^A-Z0-9]", "", ticker)
    return ticker


# ── Public loaders ────────────────────────────────────────────────────────────

def load_combined_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Load a single CSV that contains data for multiple tickers.
    The file must resolve to a 'ticker' column after normalisation.

    Works with:
        compiled_psx_historical_2017_2025.csv
        Columns: DATE, ticker, LDCP, OPEN, HIGH, LOW, CLOSE, CHANGE, CHANGE (%), VOLUME

    Returns a clean DataFrame with columns: ticker, date, open, high, low, close, volume
    Raises ValueError if required canonical columns cannot be resolved.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"  [combined_csv] Reading '{path.name}' ...")
    df = pd.read_csv(path, low_memory=False)

    print(f"  [combined_csv] Raw columns: {list(df.columns)}")

    # Map to canonical names
    df = _map_columns(df)

    # Check all required canonical columns are now present
    missing = _check_missing_canonical(df, path.name)
    if missing:
        raise ValueError(
            f"[combined_csv] '{path.name}' is missing columns after normalisation: {missing}\n"
            f"Columns after mapping: {list(df.columns)}\n"
            f"Tip: add the missing column aliases to COLUMN_ALIASES in kaggle_csv_loader.py"
        )

    # Keep only canonical columns — silently drop LDCP, CHANGE, CHANGE (%), etc.
    df = df[CANONICAL_COLUMNS].copy()

    # Type coercion + row cleaning
    df = _apply_types_and_clean(df, source=path.name)

    print(
        f"  [combined_csv] Loaded {len(df):,} rows for "
        f"{df['ticker'].nunique()} ticker(s) from '{path.name}'"
    )
    return df


def load_per_ticker_csv(filepath: str | Path, ticker: str | None = None) -> pd.DataFrame:
    """
    Load a single-ticker CSV file.
    If the file has no ticker/symbol column, the ticker is inferred from the filename.

    Returns a clean DataFrame with columns: ticker, date, open, high, low, close, volume
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    inferred_ticker = ticker if ticker else _infer_ticker_from_filename(path)

    print(f"  [per_ticker_csv] Reading '{path.name}' (ticker={inferred_ticker}) ...")
    df = pd.read_csv(path, low_memory=False)

    print(f"  [per_ticker_csv] Raw columns: {list(df.columns)}")

    df = _map_columns(df)

    # Inject ticker column if the file did not have one
    if "ticker" not in df.columns:
        df["ticker"] = inferred_ticker

    missing = _check_missing_canonical(df, path.name)
    if missing:
        raise ValueError(
            f"[per_ticker_csv] '{path.name}' is missing columns after normalisation: {missing}\n"
            f"Columns after mapping: {list(df.columns)}"
        )

    df = df[CANONICAL_COLUMNS].copy()
    df = _apply_types_and_clean(df, source=path.name)

    print(
        f"  [per_ticker_csv] Loaded {len(df):,} rows for "
        f"ticker '{inferred_ticker}' from '{path.name}'"
    )
    return df


def load_all_from_kaggle_dir(directory: str | Path = KAGGLE_DIR) -> pd.DataFrame:
    """
    Auto-detect and load all CSV files found in the given directory.

    Detection logic per file
    ------------------------
    - After column normalisation, if a 'ticker' column is resolvable → combined format
    - Otherwise → per-ticker format (ticker inferred from filename)

    Returns one combined DataFrame, or an empty DataFrame if no CSVs are found.
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"  [kaggle_dir] Directory not found: {directory}")
        return pd.DataFrame()

    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        print(f"  [kaggle_dir] No CSV files found in {directory}")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for csv_path in csv_files:
        try:
            # Peek at headers to decide format — read 0 rows for speed
            peek    = pd.read_csv(csv_path, nrows=0)
            headers = [_normalise_column_name(c) for c in peek.columns]

            # If any header maps to 'ticker' → combined file
            is_combined = any(
                COLUMN_ALIASES.get(h) == "ticker" for h in headers
            )

            if is_combined:
                df = load_combined_csv(csv_path)
            else:
                df = load_per_ticker_csv(csv_path)

            frames.append(df)

        except Exception as exc:
            errors.append(f"  ❌ {csv_path.name}: {exc}")

    if errors:
        print("\n[kaggle_dir] Errors encountered:")
        for e in errors:
            print(e)

    if not frames:
        print("[kaggle_dir] No data loaded successfully.")
        return pd.DataFrame()

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    print(
        f"\n[kaggle_dir] Total: {len(combined):,} rows | "
        f"{combined['ticker'].nunique()} ticker(s) | "
        f"{len(csv_files)} file(s) scanned"
    )
    return combined