"""
fundamental_loader.py
---------------------
Phase 4.1 — Load and normalise fundamental financial and dividend
data from CSV files into clean pandas DataFrames ready for
upsert into SQLite via db_manager.

Financial CSV
-------------
  Path    : data/raw/sample/sample_financials_2025_raw.csv
  Source columns (as delivered):
      ticker, fiscal_year, period_type,
      revenue (Billions), net_profit (Billions),
      total_assets (Billions), total_equity (Billions),
      total_debt (Billions), eps (PKR),
      book_value_per_share (PKR), shares_outstanding (Millions),
      source

  Normalised output columns:
      ticker, fiscal_year, period_type,
      revenue, net_profit, total_assets, total_equity, total_debt,
      eps, book_value_per_share, shares_outstanding, source

Dividend CSV
------------
  Path    : data/raw/sample/sample_dividends_2025.csv
  Columns : ticker, ex_dividend_date, fiscal_year,
            dividend_per_share, dividend_type, source

Conversion rules
----------------
  *Billions columns  → multiply by 1_000_000_000
  *Millions columns  → multiply by 1_000_000
  eps / book_value   → unchanged (already in PKR)
  ticker             → uppercase + stripped
  period_type        → uppercase + stripped (annual → ANNUAL)
  Negative values    → supported; not filtered out
  Zero revenue       → supported; not filtered out
  source             → kept as text
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import numpy as np


# ── Default paths ─────────────────────────────────────────────────────────────

FINANCIALS_CSV_PATH = Path("data/raw/sample/sample_financials_2025_raw.csv")
DIVIDENDS_CSV_PATH  = Path("data/raw/sample/sample_dividends_2025.csv")

# ── Multipliers ───────────────────────────────────────────────────────────────

BILLION = 1_000_000_000
MILLION = 1_000_000

# ── Column alias maps ─────────────────────────────────────────────────────────
# Maps any variant found in the CSV → canonical output column name.
# All matching is done after lowercasing + stripping the CSV header.

_FINANCIAL_ALIASES: dict[str, str] = {
    "ticker":                          "ticker",
    "fiscal_year":                     "fiscal_year",
    "period_type":                     "period_type",
    # Billion-unit columns
    "revenue (billions)":              "_revenue_b",
    "revenue(billions)":               "_revenue_b",
    "revenue":                         "_revenue_b",
    "net_profit (billions)":           "_net_profit_b",
    "net_profit(billions)":            "_net_profit_b",
    "net_profit":                      "_net_profit_b",
    "total_assets (billions)":         "_total_assets_b",
    "total_assets(billions)":          "_total_assets_b",
    "total_assets":                    "_total_assets_b",
    "total_equity (billions)":         "_total_equity_b",
    "total_equity(billions)":          "_total_equity_b",
    "total_equity":                    "_total_equity_b",
    "total_debt (billions)":           "_total_debt_b",
    "total_debt(billions)":            "_total_debt_b",
    "total_debt":                      "_total_debt_b",
    # Million-unit columns
    "shares_outstanding (millions)":   "_shares_m",
    "shares_outstanding(millions)":    "_shares_m",
    "shares_outstanding":              "_shares_m",
    # PKR columns — no conversion needed
    "eps (pkr)":                       "eps",
    "eps(pkr)":                        "eps",
    "eps":                             "eps",
    "book_value_per_share (pkr)":      "book_value_per_share",
    "book_value_per_share(pkr)":       "book_value_per_share",
    "book_value_per_share":            "book_value_per_share",
    # Metadata
    "source":                          "source",
}

_DIVIDEND_ALIASES: dict[str, str] = {
    "ticker":              "ticker",
    "ex_dividend_date":    "ex_dividend_date",
    "fiscal_year":         "fiscal_year",
    "dividend_per_share":  "dividend_per_share",
    "dividend_type":       "dividend_type",
    "source":              "source",
}

# ── Required canonical columns ────────────────────────────────────────────────

_FINANCIAL_REQUIRED  = ["ticker", "fiscal_year", "period_type"]
_DIVIDEND_REQUIRED   = ["ticker", "fiscal_year"]

# ── Date formats ──────────────────────────────────────────────────────────────

_DATE_FORMATS = [
    "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y",
    "%m/%d/%Y", "%Y/%m/%d", "%d-%b-%Y",
]


# ═══════════════════════════════════════════════════════════════════
# Validation result container
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LoadResult:
    """Holds the cleaned DataFrame and a validation summary."""
    df:        pd.DataFrame
    passed:    bool = True
    issues:    list[str] = field(default_factory=list)
    warnings:  list[str] = field(default_factory=list)
    rows_in:   int = 0
    rows_out:  int = 0

    def fail(self, msg: str) -> None:
        self.passed = False
        self.issues.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        lines  = [f"Load {status}  ({self.rows_in} in → {self.rows_out} out)"]
        if self.issues:
            lines.append("  Issues:")
            lines.extend(f"    • {i}" for i in self.issues)
        if self.warnings:
            lines.append("  Warnings:")
            lines.extend(f"    • {w}" for w in self.warnings)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════

def _normalise_col_name(col: str) -> str:
    """Lowercase + strip a column header."""
    return col.strip().lower()


def _map_columns(df: pd.DataFrame, alias_map: dict[str, str]) -> pd.DataFrame:
    """Rename columns using alias_map after normalising headers."""
    rename = {}
    for col in df.columns:
        norm = _normalise_col_name(col)
        if norm in alias_map:
            rename[col] = alias_map[norm]
    return df.rename(columns=rename)


def _parse_numeric(series: pd.Series, col_label: str = "") -> pd.Series:
    """
    Safely convert a series to float.
    Handles:
      - commas in numbers  "1,234.5" → 1234.5
      - parentheses for negatives  "(500)" → -500.0
      - plain negatives "-500"
      - zero and empty strings
    """
    s = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
    )
    # Convert parentheses notation (500) → -500
    has_parens = s.str.match(r"^\(.*\)$")
    s = s.str.replace(r"^\((.+)\)$", r"-\1", regex=True)
    return pd.to_numeric(s, errors="coerce")


def _parse_dates_safe(series: pd.Series) -> pd.Series:
    """Try multiple date formats; return Series of Python date objects."""
    for fmt in _DATE_FORMATS:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="raise")
            return parsed.dt.date
        except (ValueError, TypeError):
            continue
    parsed = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
    return parsed.dt.date


def _check_required(df: pd.DataFrame,
                    required: list[str],
                    source: str) -> list[str]:
    """Return list of required columns absent from df."""
    return [c for c in required if c not in df.columns]


# ═══════════════════════════════════════════════════════════════════
# FINANCIAL LOADER
# ═══════════════════════════════════════════════════════════════════

def load_financials_csv(
    filepath: str | Path = FINANCIALS_CSV_PATH,
) -> LoadResult:
    """
    Load, normalise, and validate a financial fundamentals CSV.

    Steps
    -----
    1. Read raw CSV
    2. Map column names via _FINANCIAL_ALIASES
    3. Convert Billion-unit columns → PKR
    4. Convert Million-unit columns → absolute count
    5. Normalise ticker (uppercase) and period_type (uppercase)
    6. Cast fiscal_year to int
    7. Drop rows missing ticker / fiscal_year / period_type
    8. Remove duplicate (ticker, fiscal_year, period_type) rows
    9. Return LoadResult

    Returns
    -------
    LoadResult with .df ready for upsert_financials()
    """
    path   = Path(filepath)
    result = LoadResult(df=pd.DataFrame())

    if not path.exists():
        result.fail(f"File not found: {path}")
        return result

    print(f"  [financials_loader] Reading '{path.name}' ...")
    raw         = pd.read_csv(path, low_memory=False)
    result.rows_in = len(raw)
    print(f"  [financials_loader] Raw columns: {list(raw.columns)}")

    # ── 1. Map columns ────────────────────────────────────────────
    df = _map_columns(raw, _FINANCIAL_ALIASES)

    # ── 2. Check required columns are resolvable ──────────────────
    missing = _check_required(df, _FINANCIAL_REQUIRED, path.name)
    if missing:
        result.fail(
            f"Missing required columns after mapping: {missing}\n"
            f"Columns after mapping: {list(df.columns)}\n"
            f"Tip: add the missing aliases to _FINANCIAL_ALIASES."
        )
        return result

    # ── 3. Convert Billion-unit columns → PKR ────────────────────
    billion_map = {
        "_revenue_b":     "revenue",
        "_net_profit_b":  "net_profit",
        "_total_assets_b": "total_assets",
        "_total_equity_b": "total_equity",
        "_total_debt_b":   "total_debt",
    }
    for raw_col, out_col in billion_map.items():
        if raw_col in df.columns:
            df[out_col] = _parse_numeric(df[raw_col], raw_col) * BILLION
            df.drop(columns=[raw_col], inplace=True)
        else:
            df[out_col] = np.nan
            result.warn(f"Column '{raw_col}' not found — '{out_col}' set to NaN")

    # ── 4. Convert Million-unit columns → absolute count ──────────
    if "_shares_m" in df.columns:
        df["shares_outstanding"] = _parse_numeric(df["_shares_m"]) * MILLION
        df.drop(columns=["_shares_m"], inplace=True)
    else:
        df["shares_outstanding"] = np.nan
        result.warn("Column 'shares_outstanding' not found — set to NaN")

    # ── 5. PKR columns — parse numeric, no unit conversion ────────
    for col in ["eps", "book_value_per_share"]:
        if col in df.columns:
            df[col] = _parse_numeric(df[col], col)
        else:
            df[col] = np.nan
            result.warn(f"Column '{col}' not found — set to NaN")

    # ── 6. Normalise ticker and period_type ───────────────────────
    df["ticker"]      = df["ticker"].astype(str).str.strip().str.upper()
    df["period_type"] = df["period_type"].astype(str).str.strip().str.upper()

    # ── 7. Cast fiscal_year to int ────────────────────────────────
    df["fiscal_year"] = pd.to_numeric(
        df["fiscal_year"], errors="coerce"
    ).dropna().astype(int)

    # ── 8. Handle source column ───────────────────────────────────
    if "source" not in df.columns:
        df["source"] = None

    # ── 9. Drop rows missing key fields ───────────────────────────
    before = len(df)
    df.dropna(subset=["ticker", "fiscal_year", "period_type"], inplace=True)
    dropped = before - len(df)
    if dropped:
        result.warn(f"Dropped {dropped} rows with missing ticker/fiscal_year/period_type")

    # Ensure fiscal_year is integer after dropna
    df["fiscal_year"] = df["fiscal_year"].astype(int)

    # ── 10. Remove duplicates ─────────────────────────────────────
    dup_key = ["ticker", "fiscal_year", "period_type"]
    before  = len(df)
    df      = df.drop_duplicates(subset=dup_key, keep="last")
    dups    = before - len(df)
    if dups:
        result.warn(f"Removed {dups} duplicate (ticker, fiscal_year, period_type) rows")

    # ── 11. Keep only canonical output columns in correct order ───
    output_cols = [
        "ticker", "fiscal_year", "period_type",
        "revenue", "net_profit",
        "total_assets", "total_equity", "total_debt",
        "eps", "book_value_per_share", "shares_outstanding",
        "source",
    ]
    # Only keep columns that exist
    output_cols = [c for c in output_cols if c in df.columns]
    df = df[output_cols].reset_index(drop=True)

    result.df       = df
    result.rows_out = len(df)

    # ── 12. Basic sanity checks (warnings only) ───────────────────
    zero_rev = (df["revenue"] == 0).sum() if "revenue" in df.columns else 0
    if zero_rev:
        result.warn(f"{zero_rev} rows have zero revenue (kept — may be valid)")

    neg_equity = (df["total_equity"].fillna(0) < 0).sum() \
        if "total_equity" in df.columns else 0
    if neg_equity:
        result.warn(f"{neg_equity} rows have negative total_equity (kept — technically valid)")

    tickers = df["ticker"].nunique()
    years   = sorted(df["fiscal_year"].unique().tolist())
    result.warn(
        f"Clean data: {len(df)} rows | {tickers} ticker(s) | "
        f"fiscal years: {years}"
    )

    print(
        f"  [financials_loader] Loaded {len(df):,} rows for "
        f"{tickers} ticker(s) from '{path.name}'"
    )
    return result


# ═══════════════════════════════════════════════════════════════════
# DIVIDEND LOADER
# ═══════════════════════════════════════════════════════════════════

def load_dividends_csv(
    filepath: str | Path = DIVIDENDS_CSV_PATH,
) -> LoadResult:
    """
    Load, normalise, and validate a dividend history CSV.

    Steps
    -----
    1. Read raw CSV
    2. Map column names via _DIVIDEND_ALIASES
    3. Normalise ticker (uppercase)
    4. Parse ex_dividend_date safely
    5. Normalise dividend_type (strip + title-case)
    6. Cast fiscal_year to int
    7. Parse dividend_per_share to float (support negatives / commas)
    8. Drop rows missing ticker / fiscal_year
    9. Remove duplicate (ticker, ex_dividend_date, dividend_type) rows

    Returns
    -------
    LoadResult with .df ready for upsert_dividends()
    """
    path   = Path(filepath)
    result = LoadResult(df=pd.DataFrame())

    if not path.exists():
        result.fail(f"File not found: {path}")
        return result

    print(f"  [dividends_loader] Reading '{path.name}' ...")
    raw         = pd.read_csv(path, low_memory=False)
    result.rows_in = len(raw)
    print(f"  [dividends_loader] Raw columns: {list(raw.columns)}")

    # ── 1. Map columns ────────────────────────────────────────────
    df = _map_columns(raw, _DIVIDEND_ALIASES)

    # ── 2. Check required columns ─────────────────────────────────
    missing = _check_required(df, _DIVIDEND_REQUIRED, path.name)
    if missing:
        result.fail(
            f"Missing required columns after mapping: {missing}\n"
            f"Columns after mapping: {list(df.columns)}"
        )
        return result

    # ── 3. Normalise ticker ───────────────────────────────────────
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    # ── 4. Parse ex_dividend_date ────────────────────────────────
    if "ex_dividend_date" in df.columns:
        df["ex_dividend_date"] = _parse_dates_safe(df["ex_dividend_date"])
        bad_dates = df["ex_dividend_date"].isna().sum()
        if bad_dates:
            result.warn(f"{bad_dates} rows have unparseable ex_dividend_date (kept as NULL)")
    else:
        df["ex_dividend_date"] = None
        result.warn("Column 'ex_dividend_date' not found — set to NULL")

    # ── 5. Normalise dividend_type ────────────────────────────────
    if "dividend_type" in df.columns:
        df["dividend_type"] = (
            df["dividend_type"]
            .astype(str)
            .str.strip()
            .str.title()
        )
        df["dividend_type"] = df["dividend_type"].replace(
            {"Nan": None, "None": None, "": None}
        )
    else:
        df["dividend_type"] = None
        result.warn("Column 'dividend_type' not found — set to NULL")

    # ── 6. Cast fiscal_year ───────────────────────────────────────
    df["fiscal_year"] = pd.to_numeric(
        df["fiscal_year"], errors="coerce"
    )

    # ── 7. Parse dividend_per_share ───────────────────────────────
    if "dividend_per_share" in df.columns:
        df["dividend_per_share"] = _parse_numeric(df["dividend_per_share"])
        neg_dps = (df["dividend_per_share"].fillna(0) < 0).sum()
        if neg_dps:
            result.warn(f"{neg_dps} rows have negative dividend_per_share (kept)")
    else:
        df["dividend_per_share"] = np.nan
        result.warn("Column 'dividend_per_share' not found — set to NaN")

    # ── 8. Handle source ──────────────────────────────────────────
    if "source" not in df.columns:
        df["source"] = None

    # ── 9. Drop rows missing key identifiers ─────────────────────
    before = len(df)
    df.dropna(subset=["ticker", "fiscal_year"], inplace=True)
    dropped = before - len(df)
    if dropped:
        result.warn(f"Dropped {dropped} rows with missing ticker or fiscal_year")

    df["fiscal_year"] = df["fiscal_year"].astype(int)

    # ── 10. Remove duplicates ─────────────────────────────────────
    dup_key = ["ticker", "ex_dividend_date", "dividend_type"]
    dup_key = [c for c in dup_key if c in df.columns]
    before  = len(df)
    df      = df.drop_duplicates(subset=dup_key, keep="last")
    dups    = before - len(df)
    if dups:
        result.warn(f"Removed {dups} duplicate rows on {dup_key}")

    # ── 11. Canonical output columns ──────────────────────────────
    output_cols = [
        "ticker", "ex_dividend_date", "fiscal_year",
        "dividend_per_share", "dividend_type", "source",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    df = df[output_cols].reset_index(drop=True)

    result.df       = df
    result.rows_out = len(df)

    tickers = df["ticker"].nunique()
    years   = sorted(df["fiscal_year"].unique().tolist())
    result.warn(
        f"Clean data: {len(df)} rows | {tickers} ticker(s) | "
        f"fiscal years: {years}"
    )

    print(
        f"  [dividends_loader] Loaded {len(df):,} rows for "
        f"{tickers} ticker(s) from '{path.name}'"
    )
    return result


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE — load + validate + raise
# ═══════════════════════════════════════════════════════════════════

def load_financials_or_raise(
    filepath: str | Path = FINANCIALS_CSV_PATH,
) -> pd.DataFrame:
    """
    Load financials CSV, print summary, raise on failure.
    Returns clean DataFrame on success.
    """
    result = load_financials_csv(filepath)
    print(result.summary())
    if not result.passed:
        raise ValueError(
            f"Financial CSV load failed for '{filepath}'. See issues above."
        )
    return result.df


def load_dividends_or_raise(
    filepath: str | Path = DIVIDENDS_CSV_PATH,
) -> pd.DataFrame:
    """
    Load dividends CSV, print summary, raise on failure.
    Returns clean DataFrame on success.
    """
    result = load_dividends_csv(filepath)
    print(result.summary())
    if not result.passed:
        raise ValueError(
            f"Dividend CSV load failed for '{filepath}'. See issues above."
        )
    return result.df
