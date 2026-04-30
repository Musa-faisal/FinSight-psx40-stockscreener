"""
data_validator.py
-----------------
Validates an OHLCV DataFrame before it is written to the database.
Phase 2 adds stricter OHLC relationship checks and duplicate removal.

Returns a ValidationResult with pass/fail status and a list of issues.
"""

from dataclasses import dataclass, field
import pandas as pd
import numpy as np


REQUIRED_COLUMNS  = ["ticker", "date", "open", "high", "low", "close", "volume"]
MIN_ROWS_PER_TICKER = 30


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    passed:   bool       = True
    issues:   list[str]  = field(default_factory=list)
    warnings: list[str]  = field(default_factory=list)
    rows_before: int     = 0
    rows_after:  int     = 0

    def fail(self, msg: str) -> None:
        self.passed = False
        self.issues.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        lines  = [f"Validation {status}  ({self.rows_before} → {self.rows_after} rows)"]
        if self.issues:
            lines.append("  Issues:")
            lines.extend(f"    • {i}" for i in self.issues)
        if self.warnings:
            lines.append("  Warnings:")
            lines.extend(f"    • {w}" for w in self.warnings)
        return "\n".join(lines)


# ── Cleaning helpers (Phase 2) ────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove duplicate (ticker, date) rows, keeping the last occurrence.
    Returns (cleaned_df, number_removed).
    """
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    return df.reset_index(drop=True), before - len(df)


def remove_invalid_ohlc(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drop rows that violate OHLC relationships or have non-positive close.
    Phase 2 requirement: fail rows individually rather than failing the whole file.
    """
    before = len(df)
    mask = (
        (df["close"]  >  0)            &
        (df["high"]   >= df["low"])    &
        (df["high"]   >= df["open"])   &
        (df["high"]   >= df["close"])  &
        (df["low"]    <= df["open"])   &
        (df["low"]    <= df["close"])
    )
    df = df[mask].reset_index(drop=True)
    return df, before - len(df)


def parse_dates_safely(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Coerce date column to Python date objects.
    Drops rows where date could not be parsed.
    """
    before = len(df)
    df = df.copy()

    if not pd.api.types.is_object_dtype(df["date"]) and \
       not pd.api.types.is_string_dtype(df["date"]):
        # Already datetime-like
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        except Exception:
            pass
    else:
        df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True, errors="coerce").dt.date

    df.dropna(subset=["date"], inplace=True)
    return df.reset_index(drop=True), before - len(df)


# ── Main validator ────────────────────────────────────────────────────────────

def validate_ohlcv(
    df: pd.DataFrame,
    source: str = "unknown",
    auto_clean: bool = True,
) -> tuple[pd.DataFrame, "ValidationResult"]:
    """
    Validate and optionally clean an OHLCV DataFrame.

    Phase 2 behaviour
    -----------------
    - auto_clean=True  → bad rows are removed and returned as a clean df
    - auto_clean=False → issues are reported but df is unchanged

    Parameters
    ----------
    df         : input OHLCV DataFrame
    source     : label for error messages
    auto_clean : whether to remove bad rows automatically

    Returns
    -------
    (cleaned_df, ValidationResult)
    """
    result = ValidationResult(rows_before=len(df))
    df = df.copy()

    # ── 1. Schema ─────────────────────────────────────────────────────────────
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        result.fail(f"[{source}] Missing columns: {missing_cols}")
        result.rows_after = len(df)
        return df, result

    # ── 2. Empty ──────────────────────────────────────────────────────────────
    if df.empty:
        result.fail(f"[{source}] DataFrame is empty")
        result.rows_after = 0
        return df, result

    # ── 3. Parse dates safely ─────────────────────────────────────────────────
    df, bad_dates = parse_dates_safely(df)
    if bad_dates > 0:
        if auto_clean:
            result.warn(f"[{source}] Removed {bad_dates} rows with unparseable dates")
        else:
            result.fail(f"[{source}] {bad_dates} rows have unparseable dates")

    # ── 4. Remove duplicate (ticker, date) pairs ──────────────────────────────
    df, dups = remove_duplicates(df)
    if dups > 0:
        if auto_clean:
            result.warn(f"[{source}] Removed {dups} duplicate (ticker, date) rows")
        else:
            result.fail(f"[{source}] {dups} duplicate (ticker, date) rows found")

    # ── 5. OHLC relationship + positive close ─────────────────────────────────
    df, bad_ohlc = remove_invalid_ohlc(df)
    if bad_ohlc > 0:
        if auto_clean:
            result.warn(
                f"[{source}] Removed {bad_ohlc} rows that failed OHLC checks "
                f"(close>0, high>=low/open/close, low<=open/close)"
            )
        else:
            result.fail(f"[{source}] {bad_ohlc} rows failed OHLC relationship checks")

    # ── 6. Null checks (after cleaning) ──────────────────────────────────────
    for col in REQUIRED_COLUMNS:
        null_count = df[col].isna().sum()
        if null_count > 0:
            result.fail(f"[{source}] Column '{col}' still has {null_count} null value(s) after cleaning")

    # ── 7. Negative volume ────────────────────────────────────────────────────
    if (df["volume"] < 0).any():
        result.fail(f"[{source}] 'volume' contains negative values")

    if (df["volume"] == 0).sum() > len(df) * 0.05:
        result.warn(f"[{source}] More than 5% of rows have zero volume")

    # ── 8. Per-ticker row count ───────────────────────────────────────────────
    row_counts = df.groupby("ticker").size()
    thin = row_counts[row_counts < MIN_ROWS_PER_TICKER].index.tolist()
    if thin:
        result.warn(
            f"[{source}] Tickers with fewer than {MIN_ROWS_PER_TICKER} rows "
            f"(indicators may be incomplete): {thin}"
        )

    # ── 9. Extreme single-day moves ───────────────────────────────────────────
    df_s = df.sort_values(["ticker", "date"])
    pct  = df_s.groupby("ticker")["close"].pct_change().abs()
    spikes = (pct > 0.30).sum()
    if spikes > 0:
        result.warn(f"[{source}] {spikes} single-day price moves > 30% detected")

    # ── 10. Summary info ──────────────────────────────────────────────────────
    if not df.empty:
        min_date = df["date"].min()
        max_date = df["date"].max()
        tickers  = df["ticker"].nunique()
        result.warn(
            f"[{source}] Clean data: {len(df)} rows | "
            f"{tickers} ticker(s) | {min_date} → {max_date}"
        )

    result.rows_after = len(df)
    return df.reset_index(drop=True), result


def validate_and_raise(df: pd.DataFrame, source: str = "unknown") -> pd.DataFrame:
    """
    Validate df, print summary, raise ValueError if validation fails.
    Returns the cleaned DataFrame on success.
    """
    clean_df, result = validate_ohlcv(df, source=source, auto_clean=True)
    print(result.summary())
    if not result.passed:
        raise ValueError(
            f"OHLCV validation failed for source '{source}'. See issues above."
        )
    return clean_df
