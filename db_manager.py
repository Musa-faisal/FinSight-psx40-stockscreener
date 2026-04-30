"""
db_manager.py
-------------
All database read/write helpers for the PSX40 screener.

Phase 4.1 additions
-------------------
  upsert_financials(df, batch_size=500)
  upsert_dividends(df, batch_size=500)
  load_financials(ticker=None)
  load_dividends(ticker=None)
  load_latest_financials()
  load_annual_dividends()

All existing Phase 1-3 functions are preserved unchanged.
"""

import math
import pandas as pd
from sqlalchemy import text, delete
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.database.connection import engine, SessionLocal
from src.database.models import (
    StockUniverse,
    StockPrice,
    StockFinancials,
    StockDividends,
)


# ── Batch size constant ───────────────────────────────────────────────────────

UPSERT_BATCH_SIZE = 500   # safe for SQLite 999-variable limit across all tables


# ═══════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════

def _batch_upsert(
    session,
    model,
    records: list[dict],
    conflict_index_elements: list[str],
    update_keys: list[str],
    batch_size: int = UPSERT_BATCH_SIZE,
    label: str = "rows",
) -> None:
    """
    Generic batched upsert for any SQLAlchemy model.
    Commits after every batch to avoid partial-failure rollbacks.

    Parameters
    ----------
    session                  : active SQLAlchemy Session
    model                    : SQLAlchemy ORM class
    records                  : list of row dicts
    conflict_index_elements  : columns forming the unique constraint
    update_keys              : columns to overwrite on conflict
    batch_size               : rows per INSERT statement
    label                    : display label for progress output
    """
    total         = len(records)
    total_batches = math.ceil(total / batch_size)

    for batch_num, start in enumerate(range(0, total, batch_size), start=1):
        chunk = records[start: start + batch_size]

        stmt = sqlite_insert(model).values(chunk)
        set_ = {k: getattr(stmt.excluded, k) for k in update_keys}
        stmt = stmt.on_conflict_do_update(
            index_elements=conflict_index_elements,
            set_=set_,
        )
        session.execute(stmt)
        session.commit()

        if batch_num % 10 == 0 or batch_num == total_batches:
            done = min(start + batch_size, total)
            print(
                f"  [upsert_{label}] batch {batch_num}/{total_batches} "
                f"({done:,}/{total:,} rows)"
            )


# ═══════════════════════════════════════════════════════════════════
# UNIVERSE  (unchanged from Phase 1)
# ═══════════════════════════════════════════════════════════════════

def upsert_universe(records: list[dict]) -> None:
    """Insert or replace rows in stock_universe."""
    with SessionLocal() as session:
        stmt = sqlite_insert(StockUniverse).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker"],
            set_={
                "name":       stmt.excluded.name,
                "sector":     stmt.excluded.sector,
                "base_price": stmt.excluded.base_price,
            },
        )
        session.execute(stmt)
        session.commit()


def load_universe() -> pd.DataFrame:
    """Return the full PSX40 universe as a DataFrame."""
    with engine.connect() as conn:
        return pd.read_sql(
            text(
                "SELECT ticker, name, sector, base_price "
                "FROM stock_universe ORDER BY sector, ticker"
            ),
            conn,
        )


# ═══════════════════════════════════════════════════════════════════
# OHLCV PRICES  (unchanged from Phase 2)
# ═══════════════════════════════════════════════════════════════════

def upsert_prices(df: pd.DataFrame, batch_size: int = UPSERT_BATCH_SIZE) -> None:
    """
    Bulk-upsert OHLCV rows into stock_prices in batches.
    Avoids SQLite 'too many SQL variables' error.
    """
    if df.empty:
        print("  [upsert_prices] Nothing to insert — DataFrame is empty.")
        return

    records       = df.to_dict(orient="records")
    total         = len(records)
    total_batches = math.ceil(total / batch_size)

    print(
        f"  [upsert_prices] Inserting {total:,} rows "
        f"in {total_batches} batch(es) of {batch_size} ..."
    )

    with SessionLocal() as session:
        _batch_upsert(
            session       = session,
            model         = StockPrice,
            records       = records,
            conflict_index_elements = ["ticker", "date"],
            update_keys   = ["open", "high", "low", "close", "volume"],
            batch_size    = batch_size,
            label         = "prices",
        )

    print(f"  [upsert_prices] Done — {total:,} rows upserted.")


def load_prices(ticker: str) -> pd.DataFrame:
    """Return full OHLCV history for a single ticker, sorted by date asc."""
    with engine.connect() as conn:
        return pd.read_sql(
            text("""
                SELECT date, open, high, low, close, volume
                FROM   stock_prices
                WHERE  ticker = :ticker
                ORDER  BY date ASC
            """),
            conn,
            params={"ticker": ticker},
            parse_dates=["date"],
        )


def load_all_prices() -> pd.DataFrame:
    """Return OHLCV for every ticker, sorted by ticker then date."""
    with engine.connect() as conn:
        return pd.read_sql(
            text(
                "SELECT ticker, date, open, high, low, close, volume "
                "FROM stock_prices ORDER BY ticker, date"
            ),
            conn,
            parse_dates=["date"],
        )


def load_latest_prices() -> pd.DataFrame:
    """Return the most-recent OHLCV row for each ticker."""
    with engine.connect() as conn:
        return pd.read_sql(
            text("""
                SELECT sp.ticker, sp.date, sp.open, sp.high,
                       sp.low, sp.close, sp.volume
                FROM   stock_prices sp
                INNER JOIN (
                    SELECT ticker, MAX(date) AS max_date
                    FROM   stock_prices
                    GROUP  BY ticker
                ) latest
                  ON sp.ticker = latest.ticker
                 AND sp.date   = latest.max_date
                ORDER BY sp.ticker
            """),
            conn,
            parse_dates=["date"],
        )


def get_available_tickers() -> list[str]:
    """Return sorted list of tickers that have OHLCV data in the DB."""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT DISTINCT ticker FROM stock_prices ORDER BY ticker")
        )
        return [row[0] for row in result]


def delete_prices(ticker: str) -> None:
    """Remove all OHLCV rows for a given ticker."""
    with SessionLocal() as session:
        session.execute(
            delete(StockPrice).where(StockPrice.ticker == ticker)
        )
        session.commit()


# ═══════════════════════════════════════════════════════════════════
# FUNDAMENTALS  (Phase 4.1)
# ═══════════════════════════════════════════════════════════════════

_FINANCIAL_UPDATE_KEYS = [
    "revenue", "net_profit",
    "total_assets", "total_equity", "total_debt",
    "eps", "book_value_per_share", "shares_outstanding",
    "source",
]


def upsert_financials(
    df: pd.DataFrame,
    batch_size: int = UPSERT_BATCH_SIZE,
) -> None:
    """
    Bulk-upsert fundamental financial rows into stock_financials.

    Unique constraint : ticker + fiscal_year + period_type
    On conflict       : update all financial columns + source

    Parameters
    ----------
    df         : clean DataFrame from fundamental_loader.load_financials_csv()
    batch_size : rows per INSERT (default 500 — safe for SQLite)
    """
    if df.empty:
        print("  [upsert_financials] Nothing to insert — DataFrame is empty.")
        return

    records = df.to_dict(orient="records")
    total   = len(records)
    print(
        f"  [upsert_financials] Inserting {total:,} rows "
        f"in {math.ceil(total / batch_size)} batch(es) ..."
    )

    with SessionLocal() as session:
        _batch_upsert(
            session                 = session,
            model                   = StockFinancials,
            records                 = records,
            conflict_index_elements = ["ticker", "fiscal_year", "period_type"],
            update_keys             = _FINANCIAL_UPDATE_KEYS,
            batch_size              = batch_size,
            label                   = "financials",
        )

    print(f"  [upsert_financials] Done — {total:,} rows upserted.")


def load_financials(ticker: str | None = None) -> pd.DataFrame:
    """
    Load fundamental data from stock_financials.

    Parameters
    ----------
    ticker : if provided, filter to a single ticker; else return all

    Returns
    -------
    pd.DataFrame ordered by ticker, fiscal_year desc, period_type
    """
    where  = "WHERE ticker = :ticker" if ticker else ""
    params = {"ticker": ticker.strip().upper()} if ticker else {}

    with engine.connect() as conn:
        return pd.read_sql(
            text(f"""
                SELECT
                    ticker, fiscal_year, period_type,
                    revenue, net_profit,
                    total_assets, total_equity, total_debt,
                    eps, book_value_per_share, shares_outstanding,
                    source
                FROM  stock_financials
                {where}
                ORDER BY ticker, fiscal_year DESC, period_type
            """),
            conn,
            params=params,
        )


def load_latest_financials() -> pd.DataFrame:
    """
    Return the most-recent ANNUAL row for every ticker that has
    fundamental data.  Useful for ratio calculations in Phase 4.2.
    """
    with engine.connect() as conn:
        return pd.read_sql(
            text("""
                SELECT sf.*
                FROM   stock_financials sf
                INNER JOIN (
                    SELECT   ticker, MAX(fiscal_year) AS max_year
                    FROM     stock_financials
                    WHERE    period_type = 'ANNUAL'
                    GROUP BY ticker
                ) latest
                  ON sf.ticker      = latest.ticker
                 AND sf.fiscal_year = latest.max_year
                 AND sf.period_type = 'ANNUAL'
                ORDER BY sf.ticker
            """),
            conn,
        )


# ═══════════════════════════════════════════════════════════════════
# DIVIDENDS  (Phase 4.1)
# ═══════════════════════════════════════════════════════════════════

_DIVIDEND_UPDATE_KEYS = [
    "fiscal_year", "dividend_per_share", "source",
]


def upsert_dividends(
    df: pd.DataFrame,
    batch_size: int = UPSERT_BATCH_SIZE,
) -> None:
    """
    Bulk-upsert dividend rows into stock_dividends.

    Unique constraint : ticker + ex_dividend_date + dividend_type
    On conflict       : update fiscal_year, dividend_per_share, source

    Parameters
    ----------
    df         : clean DataFrame from fundamental_loader.load_dividends_csv()
    batch_size : rows per INSERT (default 500 — safe for SQLite)
    """
    if df.empty:
        print("  [upsert_dividends] Nothing to insert — DataFrame is empty.")
        return

    records = df.to_dict(orient="records")
    total   = len(records)
    print(
        f"  [upsert_dividends] Inserting {total:,} rows "
        f"in {math.ceil(total / batch_size)} batch(es) ..."
    )

    with SessionLocal() as session:
        _batch_upsert(
            session                 = session,
            model                   = StockDividends,
            records                 = records,
            conflict_index_elements = ["ticker", "ex_dividend_date", "dividend_type"],
            update_keys             = _DIVIDEND_UPDATE_KEYS,
            batch_size              = batch_size,
            label                   = "dividends",
        )

    print(f"  [upsert_dividends] Done — {total:,} rows upserted.")


def load_dividends(ticker: str | None = None) -> pd.DataFrame:
    """
    Load dividend history from stock_dividends.

    Parameters
    ----------
    ticker : if provided, filter to a single ticker; else return all

    Returns
    -------
    pd.DataFrame ordered by ticker, ex_dividend_date desc
    """
    where  = "WHERE ticker = :ticker" if ticker else ""
    params = {"ticker": ticker.strip().upper()} if ticker else {}

    with engine.connect() as conn:
        return pd.read_sql(
            text(f"""
                SELECT
                    ticker, ex_dividend_date, fiscal_year,
                    dividend_per_share, dividend_type, source
                FROM  stock_dividends
                {where}
                ORDER BY ticker, ex_dividend_date DESC
            """),
            conn,
            params=params,
            parse_dates=["ex_dividend_date"],
        )


def load_annual_dividends() -> pd.DataFrame:
    """
    Return total annual dividend per share per ticker per fiscal year.
    Aggregates interim + final dividends into a single annual DPS figure.
    Used for dividend yield calculations in Phase 4.2.
    """
    with engine.connect() as conn:
        return pd.read_sql(
            text("""
                SELECT
                    ticker,
                    fiscal_year,
                    SUM(dividend_per_share)  AS annual_dps,
                    COUNT(*)                 AS payment_count
                FROM  stock_dividends
                GROUP BY ticker, fiscal_year
                ORDER BY ticker, fiscal_year DESC
            """),
            conn,
        )


# ═══════════════════════════════════════════════════════════════════
# DATABASE SUMMARY  (updated to include Phase 4.1 tables)
# ═══════════════════════════════════════════════════════════════════

def db_summary() -> dict:
    """
    Return quick stats about what is currently stored in the DB.
    Covers all four tables: universe, prices, financials, dividends.
    """
    with engine.connect() as conn:

        def _count(table: str) -> int:
            return conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0

        def _distinct(table: str, col: str) -> int:
            return conn.execute(
                text(f"SELECT COUNT(DISTINCT {col}) FROM {table}")
            ).scalar() or 0

        return {
            # Existing
            "universe_stocks":       _count("stock_universe"),
            "price_rows":            _count("stock_prices"),
            "tickers_with_prices":   _distinct("stock_prices",    "ticker"),
            # Phase 4.1
            "financial_rows":        _count("stock_financials"),
            "tickers_with_financials": _distinct("stock_financials", "ticker"),
            "dividend_rows":         _count("stock_dividends"),
            "tickers_with_dividends":  _distinct("stock_dividends",  "ticker"),
        }