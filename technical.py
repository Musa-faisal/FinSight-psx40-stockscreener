"""
technical.py
------------
Compute all technical indicators for a single-ticker OHLCV DataFrame.
Phase 2 upgrade adds:
    SMA 200, MACD, 52-week high, rolling returns,
    30d volatility, max drawdown, downside deviation.

Input  : DataFrame with columns  date, open, high, low, close, volume
Output : Same DataFrame with all indicator columns appended.
"""

import numpy as np
import pandas as pd

from config.settings import (
    SMA_SHORT, SMA_LONG, SMA_200,
    RSI_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    VOLUME_RATIO_WINDOW,
    VOLATILITY_WINDOW,
    DRAWDOWN_WINDOW,
    DOWNSIDE_WINDOW,
    RETURN_1M, RETURN_3M, RETURN_6M,
    HIGH_52W_WINDOW,
)


# ── Moving Averages ───────────────────────────────────────────────────────────

def add_sma(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA 20, SMA 50, SMA 200."""
    df = df.copy()
    for window in [SMA_SHORT, SMA_LONG, SMA_200]:
        df[f"sma_{window}"] = (
            df["close"]
            .rolling(window=window, min_periods=window)
            .mean()
            .round(4)
        )
    return df


# ── RSI ───────────────────────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """Wilder smoothed RSI."""
    df    = df.copy()
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"rsi_{RSI_PERIOD}"] = (100 - (100 / (1 + rs))).round(2)
    return df


# ── MACD ──────────────────────────────────────────────────────────────────────

def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add MACD line, signal line, and histogram.
        macd_line     = EMA(fast) - EMA(slow)
        macd_signal   = EMA(macd_line, signal_period)
        macd_hist     = macd_line - macd_signal
    """
    df    = df.copy()
    ema_f = df["close"].ewm(span=MACD_FAST,  adjust=False).mean()
    ema_s = df["close"].ewm(span=MACD_SLOW,  adjust=False).mean()

    df["macd_line"]   = (ema_f - ema_s).round(4)
    df["macd_signal"] = df["macd_line"].ewm(span=MACD_SIGNAL, adjust=False).mean().round(4)
    df["macd_hist"]   = (df["macd_line"] - df["macd_signal"]).round(4)
    return df


# ── Rolling Returns ───────────────────────────────────────────────────────────

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add percentage price returns over 1M, 3M, 6M windows.
    Expressed as decimals, e.g. 0.05 = +5%.
    """
    df = df.copy()
    for label, window in [
        ("return_1m", RETURN_1M),
        ("return_3m", RETURN_3M),
        ("return_6m", RETURN_6M),
    ]:
        df[label] = df["close"].pct_change(periods=window).round(4)
    return df


# ── 52-Week High / Breakout ───────────────────────────────────────────────────

def add_52w_high(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling 52-week high and breakout ratio.
        high_52w        = max(close) over trailing 252 days
        breakout_ratio  = close / high_52w   (1.0 = at 52w high)
    """
    df = df.copy()
    df["high_52w"] = (
        df["close"]
        .rolling(window=HIGH_52W_WINDOW, min_periods=20)
        .max()
        .round(4)
    )
    df["breakout_ratio"] = (
        (df["close"] / df["high_52w"].replace(0, np.nan))
        .round(4)
    )
    return df


# ── Volatility ────────────────────────────────────────────────────────────────

def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Annualised 30-day volatility from log returns.
    Expressed as decimal, e.g. 0.28 = 28% annualised.
    """
    df = df.copy()
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["volatility_30d"] = (
        log_ret
        .rolling(window=VOLATILITY_WINDOW, min_periods=5)
        .std()
        .mul(np.sqrt(252))
        .round(4)
    )
    return df


# ── Max Drawdown ──────────────────────────────────────────────────────────────

def add_max_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling max drawdown over DRAWDOWN_WINDOW trading days.
    Expressed as a negative decimal, e.g. -0.25 = 25% drawdown.
    """
    df = df.copy()

    def _rolling_mdd(series: pd.Series, window: int) -> pd.Series:
        result = pd.Series(index=series.index, dtype=float)
        for i in range(len(series)):
            if i < window:
                window_slice = series.iloc[:i + 1]
            else:
                window_slice = series.iloc[i - window + 1: i + 1]
            if len(window_slice) < 2:
                result.iloc[i] = np.nan
                continue
            roll_max = window_slice.cummax()
            drawdown = (window_slice - roll_max) / roll_max.replace(0, np.nan)
            result.iloc[i] = drawdown.min()
        return result

    # Vectorised rolling MDD — efficient version
    close = df["close"].values
    mdd   = np.full(len(close), np.nan)

    for i in range(1, len(close)):
        start  = max(0, i - DRAWDOWN_WINDOW + 1)
        window = close[start: i + 1]
        roll_max = np.maximum.accumulate(window)
        with np.errstate(invalid="ignore", divide="ignore"):
            dd = (window - roll_max) / roll_max
        mdd[i] = dd.min() if len(dd) > 0 else np.nan

    df["max_drawdown"] = np.round(mdd, 4)
    return df


# ── Downside Deviation ────────────────────────────────────────────────────────

def add_downside_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Annualised downside deviation (semi-deviation of negative daily returns).
    Uses a 0% minimum acceptable return threshold.
    """
    df      = df.copy()
    daily_r = df["close"].pct_change()

    def _downside_dev(returns: pd.Series) -> float:
        neg = returns[returns < 0]
        if len(neg) < 2:
            return np.nan
        return float(np.sqrt((neg ** 2).mean()) * np.sqrt(252))

    df["downside_deviation"] = (
        daily_r
        .rolling(window=DOWNSIDE_WINDOW, min_periods=5)
        .apply(_downside_dev, raw=False)
        .round(4)
    )
    return df


# ── Volume Indicators ─────────────────────────────────────────────────────────

def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    avg_volume_20d — rolling 20-day mean volume
    volume_surge   — today's volume / avg_volume_20d
    """
    df = df.copy()
    df["avg_volume_20d"] = (
        df["volume"]
        .rolling(window=VOLUME_RATIO_WINDOW, min_periods=5)
        .mean()
        .round(0)
        .astype("Int64")
    )
    df["volume_surge"] = (
        (df["volume"] / df["avg_volume_20d"].replace(0, np.nan))
        .round(3)
    )
    return df


# ── SMA Trend Flags ───────────────────────────────────────────────────────────

def add_sma_trend_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add three binary trend flags (1 = condition true, 0 = false):
        price_above_sma20  : close > SMA20
        sma20_above_sma50  : SMA20 > SMA50
        sma50_above_sma200 : SMA50 > SMA200
    """
    df = df.copy()

    df["price_above_sma20"] = (
        (df["close"] > df[f"sma_{SMA_SHORT}"]).astype(int)
        if f"sma_{SMA_SHORT}" in df.columns else 0
    )
    df["sma20_above_sma50"] = (
        (df[f"sma_{SMA_SHORT}"] > df[f"sma_{SMA_LONG}"]).astype(int)
        if f"sma_{SMA_LONG}" in df.columns else 0
    )
    df["sma50_above_sma200"] = (
        (df[f"sma_{SMA_LONG}"] > df[f"sma_{SMA_200}"]).astype(int)
        if f"sma_{SMA_200}" in df.columns else 0
    )
    return df


# ── Master pipeline ───────────────────────────────────────────────────────────

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full indicator pipeline on a sorted single-ticker OHLCV DataFrame.

    Added columns
    -------------
    Trend    : sma_20, sma_50, sma_200,
               price_above_sma20, sma20_above_sma50, sma50_above_sma200,
               return_1m, return_3m, return_6m
    Momentum : rsi_14, macd_line, macd_signal, macd_hist,
               high_52w, breakout_ratio
    Risk     : volatility_30d, max_drawdown, downside_deviation
    Volume   : avg_volume_20d, volume_surge
    """
    if df.empty:
        return df

    df = df.sort_values("date").reset_index(drop=True)
    df = add_sma(df)
    df = add_sma_trend_flags(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_returns(df)
    df = add_52w_high(df)
    df = add_volatility(df)
    df = add_max_drawdown(df)
    df = add_downside_deviation(df)
    df = add_volume_indicators(df)
    return df


def get_latest_indicators(df: pd.DataFrame) -> dict:
    """
    Return a flat dict of the most-recent row's indicator values.
    Expects df to have passed through compute_all_indicators().
    """
    if df.empty:
        return {}

    last = df.iloc[-1]

    def _safe(col: str, round_to: int = 4):
        val = last.get(col, np.nan)
        if val is None:
            return None
        try:
            f = float(val)
            return None if np.isnan(f) else round(f, round_to)
        except (TypeError, ValueError):
            return None

    def _safe_int(col: str):
        val = last.get(col, None)
        try:
            return int(val) if val is not None and not pd.isna(val) else None
        except (TypeError, ValueError):
            return None

    return {
        # price
        "latest_close":         _safe("close", 2),
        # trend
        f"sma_{SMA_SHORT}":     _safe(f"sma_{SMA_SHORT}", 2),
        f"sma_{SMA_LONG}":      _safe(f"sma_{SMA_LONG}",  2),
        f"sma_{SMA_200}":       _safe(f"sma_{SMA_200}",   2),
        "price_above_sma20":    _safe_int("price_above_sma20"),
        "sma20_above_sma50":    _safe_int("sma20_above_sma50"),
        "sma50_above_sma200":   _safe_int("sma50_above_sma200"),
        "return_1m":            _safe("return_1m", 4),
        "return_3m":            _safe("return_3m", 4),
        "return_6m":            _safe("return_6m", 4),
        # momentum
        f"rsi_{RSI_PERIOD}":    _safe(f"rsi_{RSI_PERIOD}", 2),
        "macd_line":            _safe("macd_line",   4),
        "macd_signal":          _safe("macd_signal", 4),
        "macd_hist":            _safe("macd_hist",   4),
        "high_52w":             _safe("high_52w",    2),
        "breakout_ratio":       _safe("breakout_ratio", 4),
        # risk
        "volatility_30d":       _safe("volatility_30d",      4),
        "max_drawdown":         _safe("max_drawdown",         4),
        "downside_deviation":   _safe("downside_deviation",   4),
        # volume
        "avg_volume_20d":       _safe_int("avg_volume_20d"),
        "volume_surge":         _safe("volume_surge", 3),
    }