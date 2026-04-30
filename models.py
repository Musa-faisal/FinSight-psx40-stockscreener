"""
models.py
---------
SQLAlchemy ORM table definitions for the PSX40 screener.

Tables
------
  stock_universe   — 40-stock PSX universe
  stock_prices     — OHLCV price history (one row per ticker per trading day)
  stock_financials — Annual/quarterly fundamental data
  stock_dividends  — Dividend payment history
"""

from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    BigInteger,
    Date,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import validates

from src.database.connection import Base


# ═══════════════════════════════════════════════════════════════════
# StockUniverse
# ═══════════════════════════════════════════════════════════════════

class StockUniverse(Base):
    """One row per PSX stock in the 40-stock universe."""

    __tablename__ = "stock_universe"

    ticker      = Column(String(20),  primary_key=True, nullable=False)
    name        = Column(String(120), nullable=False)
    sector      = Column(String(60),  nullable=False)
    base_price  = Column(Float,       nullable=False, default=100.0)

    @validates("ticker")
    def normalise_ticker(self, key, value: str) -> str:
        return value.strip().upper()

    @validates("base_price")
    def check_base_price(self, key, value: float) -> float:
        if value <= 0:
            raise ValueError(f"base_price must be > 0, got {value}")
        return value

    def __repr__(self) -> str:
        return (
            f"<StockUniverse ticker={self.ticker!r} "
            f"name={self.name!r} sector={self.sector!r}>"
        )


# ═══════════════════════════════════════════════════════════════════
# StockPrice
# ═══════════════════════════════════════════════════════════════════

class StockPrice(Base):
    """One row per ticker per trading day — raw OHLCV data."""

    __tablename__ = "stock_prices"

    id      = Column(Integer,    primary_key=True, autoincrement=True)
    ticker  = Column(String(20), nullable=False, index=True)
    date    = Column(Date,       nullable=False)
    open    = Column(Float,      nullable=False)
    high    = Column(Float,      nullable=False)
    low     = Column(Float,      nullable=False)
    close   = Column(Float,      nullable=False)
    volume  = Column(BigInteger, nullable=False, default=0)

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_stock_prices_ticker_date"),
        Index("ix_stock_prices_ticker_date", "ticker", "date"),
    )

    @validates("ticker")
    def normalise_ticker(self, key, value: str) -> str:
        return value.strip().upper()

    @validates("open", "high", "low", "close")
    def check_positive_price(self, key, value: float) -> float:
        if value is not None and value < 0:
            raise ValueError(f"{key} must be >= 0, got {value}")
        return value

    @validates("volume")
    def check_volume(self, key, value: int) -> int:
        if value is not None and value < 0:
            raise ValueError(f"volume must be >= 0, got {value}")
        return value

    def __repr__(self) -> str:
        return (
            f"<StockPrice ticker={self.ticker!r} date={self.date} "
            f"close={self.close} volume={self.volume}>"
        )


# ═══════════════════════════════════════════════════════════════════
# StockFinancials
# ═══════════════════════════════════════════════════════════════════

class StockFinancials(Base):
    """
    Annual or quarterly fundamental data for one ticker.
    All monetary fields are stored in base units (PKR, not Billions).
    Shares outstanding stored as absolute count (not Millions).
    """

    __tablename__ = "stock_financials"

    id                   = Column(Integer,    primary_key=True, autoincrement=True)
    ticker               = Column(String(20), nullable=False, index=True)
    fiscal_year          = Column(Integer,    nullable=False)
    period_type          = Column(String(20), nullable=False)   # ANNUAL / Q1 / Q2 / Q3 / Q4

    # Income statement
    revenue              = Column(Float, nullable=True)         # PKR
    net_profit           = Column(Float, nullable=True)         # PKR

    # Balance sheet
    total_assets         = Column(Float, nullable=True)         # PKR
    total_equity         = Column(Float, nullable=True)         # PKR
    total_debt           = Column(Float, nullable=True)         # PKR

    # Per-share data
    eps                  = Column(Float, nullable=True)         # PKR
    book_value_per_share = Column(Float, nullable=True)         # PKR
    shares_outstanding   = Column(Float, nullable=True)         # absolute count

    # Metadata
    source               = Column(String(120), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "ticker", "fiscal_year", "period_type",
            name="uq_financials_ticker_year_period",
        ),
        Index("ix_financials_ticker_year", "ticker", "fiscal_year"),
    )

    @validates("ticker")
    def normalise_ticker(self, key, value: str) -> str:
        return value.strip().upper()

    @validates("period_type")
    def normalise_period_type(self, key, value: str) -> str:
        return value.strip().upper()

    @validates("fiscal_year")
    def check_fiscal_year(self, key, value) -> int:
        v = int(value)
        if not (1990 <= v <= 2100):
            raise ValueError(f"fiscal_year out of range: {v}")
        return v

    def __repr__(self) -> str:
        return (
            f"<StockFinancials ticker={self.ticker!r} "
            f"year={self.fiscal_year} period={self.period_type} "
            f"revenue={self.revenue} eps={self.eps}>"
        )


# ═══════════════════════════════════════════════════════════════════
# StockDividends
# ═══════════════════════════════════════════════════════════════════

class StockDividends(Base):
    """
    Dividend payment record for one ticker.
    One row per unique (ticker, ex_dividend_date, dividend_type).
    """

    __tablename__ = "stock_dividends"

    id                 = Column(Integer,    primary_key=True, autoincrement=True)
    ticker             = Column(String(20), nullable=False, index=True)
    ex_dividend_date   = Column(Date,       nullable=True)
    fiscal_year        = Column(Integer,    nullable=False)
    dividend_per_share = Column(Float,      nullable=True)   # PKR
    dividend_type      = Column(String(40), nullable=True)   # Cash / Stock / Interim / Final
    source             = Column(String(120), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "ticker", "ex_dividend_date", "dividend_type",
            name="uq_dividends_ticker_date_type",
        ),
        Index("ix_dividends_ticker_year", "ticker", "fiscal_year"),
    )

    @validates("ticker")
    def normalise_ticker(self, key, value: str) -> str:
        return value.strip().upper()

    @validates("fiscal_year")
    def check_fiscal_year(self, key, value) -> int:
        v = int(value)
        if not (1990 <= v <= 2100):
            raise ValueError(f"fiscal_year out of range: {v}")
        return v

    def __repr__(self) -> str:
        return (
            f"<StockDividends ticker={self.ticker!r} "
            f"ex_date={self.ex_dividend_date} "
            f"dps={self.dividend_per_share} type={self.dividend_type}>"
        )