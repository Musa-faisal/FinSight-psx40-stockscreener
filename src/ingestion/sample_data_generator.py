"""
sample_data_generator.py
------------------------
Generates realistic synthetic OHLCV price data for all PSX40 tickers.

Exported
--------
  generate_all_sample_data() -> pd.DataFrame
      Returns a DataFrame with columns:
          ticker | date | open | high | low | close | volume
      Covers the last 365 calendar days for every ticker in PSX40_UNIVERSE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Try to import the universe; fall back to a minimal stub ──────
try:
    from config.stock_universe import PSX40_UNIVERSE  # type: ignore
except ImportError:
    # Minimal fallback so the module never hard-crashes at import time
    PSX40_UNIVERSE = [
        {"ticker": "ENGRO"},
        {"ticker": "LUCK"},
        {"ticker": "HBL"},
        {"ticker": "UBL"},
        {"ticker": "MCB"},
        {"ticker": "OGDC"},
        {"ticker": "PPL"},
        {"ticker": "PSO"},
        {"ticker": "FFC"},
        {"ticker": "EFERT"},
    ]

# ── Constants ────────────────────────────────────────────────────

_DAYS        = 365          # trading history length (calendar days)
_ANNUAL_VOL  = 0.35         # annualised volatility  (35 % — typical PSX)
_DAILY_VOL   = _ANNUAL_VOL / (252 ** 0.5)
_DRIFT       = 0.0002       # slight upward drift per day
_RNG_SEED    = 42           # reproducible across deploys

# Reasonable per-sector starting-price bands (PKR)
_SECTOR_PRICE_BANDS: dict[str, tuple[float, float]] = {
    "Energy":          (80,  350),
    "Banking":         (20,  120),
    "Fertilizer":      (60,  300),
    "Cement":          (80,  500),
    "Textile":         (30,  200),
    "Technology":      (20,  150),
    "Pharmaceuticals": (40,  250),
    "Food & Beverages":(50,  400),
    "Chemicals":       (40,  200),
    "Automobile":      (100, 600),
}
_DEFAULT_PRICE_BAND = (30, 400)


# ── Helpers ──────────────────────────────────────────────────────

def _start_price(ticker_info: dict, rng: np.random.Generator) -> float:
    """Pick a reproducible starting price for a ticker."""
    sector = ticker_info.get("sector", "")
    lo, hi = _SECTOR_PRICE_BANDS.get(sector, _DEFAULT_PRICE_BAND)
    return float(rng.uniform(lo, hi))


def _generate_ticker_ohlcv(
    ticker: str,
    start_price: float,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simulate daily OHLCV for a single ticker using Geometric Brownian Motion.
    """
    n = len(dates)

    # Close prices via GBM
    shocks  = rng.normal(_DRIFT, _DAILY_VOL, size=n)
    log_ret = np.cumsum(shocks)
    closes  = start_price * np.exp(log_ret)

    # Daily intraday range  ≈ 1.5 × daily vol
    range_frac = np.abs(rng.normal(0, _DAILY_VOL * 1.5, size=n))
    range_frac = np.clip(range_frac, 0.002, 0.10)   # 0.2 % – 10 %

    highs  = closes * (1 + range_frac)
    lows   = closes * (1 - range_frac)
    opens  = lows + rng.uniform(0, 1, size=n) * (highs - lows)

    # Volume — base 1 M shares ± log-normal noise
    base_vol = rng.uniform(500_000, 5_000_000)
    volumes  = (base_vol * rng.lognormal(0, 0.5, size=n)).astype(int)
    volumes  = np.clip(volumes, 10_000, 50_000_000)

    return pd.DataFrame(
        {
            "ticker": ticker,
            "date":   dates,
            "open":   np.round(opens,  2),
            "high":   np.round(highs,  2),
            "low":    np.round(lows,   2),
            "close":  np.round(closes, 2),
            "volume": volumes,
        }
    )


# ── Public API ───────────────────────────────────────────────────

def generate_all_sample_data() -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for every ticker in PSX40_UNIVERSE.

    Returns
    -------
    pd.DataFrame
        Columns: ticker | date | open | high | low | close | volume
        Sorted by ticker then date, index reset.
    """
    rng   = np.random.default_rng(_RNG_SEED)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=_DAYS)

    frames: list[pd.DataFrame] = []

    for entry in PSX40_UNIVERSE:
        ticker      = entry["ticker"] if isinstance(entry, dict) else str(entry)
        start_price = _start_price(
            entry if isinstance(entry, dict) else {}, rng
        )
        df = _generate_ticker_ohlcv(ticker, start_price, dates, rng)
        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=["ticker", "date", "open", "high", "low", "close", "volume"]
        )

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    print(
        f"[sample_data_generator] Generated {len(combined):,} rows "
        f"for {combined['ticker'].nunique()} tickers."
    )
    return combined
