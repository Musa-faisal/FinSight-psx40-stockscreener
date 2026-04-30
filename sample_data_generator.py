import numpy as np
import pandas as pd
from datetime import date, timedelta

from config.settings import OHLCV_DAYS, SAMPLE_TICKERS_COUNT
from config.stock_universe import PSX40_UNIVERSE

# First 10 tickers from the universe get OHLCV data generated
SAMPLE_TICKERS = [s["ticker"] for s in PSX40_UNIVERSE[:SAMPLE_TICKERS_COUNT]]


def _generate_ohlcv_for_ticker(
    ticker: str,
    base_price: float,
    days: int = OHLCV_DAYS,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate realistic OHLCV data for a single ticker using a
    Geometric Brownian Motion (GBM) price walk.

    Parameters
    ----------
    ticker     : stock symbol
    base_price : starting close price (from universe definition)
    days       : number of trading days to simulate
    seed       : optional random seed for reproducibility
    """
    rng = np.random.default_rng(seed=seed if seed is not None else hash(ticker) % (2**31))

    # GBM parameters — mild drift, realistic daily volatility
    mu    = 0.0003          # ~7% annualised drift
    sigma = 0.018           # ~1.8% daily volatility

    # Simulate daily log returns
    log_returns = rng.normal(loc=mu, scale=sigma, size=days)
    close_prices = base_price * np.exp(np.cumsum(log_returns))

    # Derive OHLC from close using realistic intraday noise
    daily_range_pct = np.abs(rng.normal(loc=0.015, scale=0.007, size=days)).clip(0.003, 0.06)

    high  = close_prices * (1 + daily_range_pct * rng.uniform(0.3, 1.0, size=days))
    low   = close_prices * (1 - daily_range_pct * rng.uniform(0.3, 1.0, size=days))
    open_ = low + rng.uniform(0, 1, size=days) * (high - low)

    # Ensure OHLC consistency: low <= open,close <= high
    low   = np.minimum(low,   np.minimum(open_, close_prices))
    high  = np.maximum(high,  np.maximum(open_, close_prices))

    # Volume: base volume with random spikes
    base_volume  = rng.integers(500_000, 5_000_000)
    volume_noise = rng.lognormal(mean=0, sigma=0.5, size=days)
    volume       = (base_volume * volume_noise).astype(int).clip(10_000)

    # Build date index (skip weekends — PSX is Mon-Fri)
    end_date   = date.today()
    all_dates  = [end_date - timedelta(days=i) for i in range(days * 2)]
    trade_dates = [d for d in all_dates if d.weekday() < 5][:days]
    trade_dates = sorted(trade_dates)  # ascending

    df = pd.DataFrame({
        "ticker": ticker,
        "date":   trade_dates,
        "open":   np.round(open_,         2),
        "high":   np.round(high,          2),
        "low":    np.round(low,           2),
        "close":  np.round(close_prices,  2),
        "volume": volume,
    })

    return df


def generate_all_sample_data(days: int = OHLCV_DAYS) -> pd.DataFrame:
    """
    Generate OHLCV data for all SAMPLE_TICKERS and return as one DataFrame.
    Uses ticker hash as seed so output is stable between runs.
    """
    universe_map = {s["ticker"]: s["base_price"] for s in PSX40_UNIVERSE}
    frames = []

    for ticker in SAMPLE_TICKERS:
        base_price = universe_map.get(ticker, 100.0)
        df = _generate_ohlcv_for_ticker(ticker, base_price, days=days)
        frames.append(df)
        print(f"  [generator] {ticker}: {len(df)} rows generated (base ≈ {base_price})")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  [generator] Total rows: {len(combined)} across {len(SAMPLE_TICKERS)} tickers")
    return combined