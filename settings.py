import os
from dotenv import load_dotenv

load_dotenv()

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "data/psx40.db")
DB_URL  = f"sqlite:///{DB_PATH}"

# ── App ───────────────────────────────────────────────────────────────────────
APP_TITLE = os.getenv("APP_TITLE", "PSX40 Stock Screener")
DEBUG     = os.getenv("DEBUG", "False").lower() == "true"

# ── Data generation ───────────────────────────────────────────────────────────
SAMPLE_TICKERS_COUNT = 10
OHLCV_DAYS           = 365

# ── Indicator windows ─────────────────────────────────────────────────────────
SMA_SHORT          = 20
SMA_LONG           = 50
SMA_200            = 200
RSI_PERIOD         = 14
MACD_FAST          = 12
MACD_SLOW          = 26
MACD_SIGNAL        = 9
VOLUME_RATIO_WINDOW = 20
VOLATILITY_WINDOW  = 30      # 30-day rolling volatility
DRAWDOWN_WINDOW    = 252     # 1-year lookback for max drawdown
DOWNSIDE_WINDOW    = 30      # downside deviation window

# ── Return periods (trading days) ─────────────────────────────────────────────
RETURN_1M  = 21
RETURN_3M  = 63
RETURN_6M  = 126

# ── 52-week high window ───────────────────────────────────────────────────────
HIGH_52W_WINDOW = 252

# ── Factor weights — must sum to 1.0 ─────────────────────────────────────────
# Group weights
WEIGHT_TREND      = 0.35
WEIGHT_MOMENTUM   = 0.30
WEIGHT_RISK       = 0.20
WEIGHT_VOLUME     = 0.15

# Within TREND (6 factors)
TREND_FACTOR_WEIGHTS = {
    "price_above_sma20":  0.20,
    "sma20_above_sma50":  0.20,
    "sma50_above_sma200": 0.20,
    "return_1m":          0.15,
    "return_3m":          0.15,
    "return_6m":          0.10,
}

# Within MOMENTUM (3 factors)
MOMENTUM_FACTOR_WEIGHTS = {
    "rsi_score":      0.35,
    "macd_score":     0.35,
    "breakout_score": 0.30,
}

# Within RISK (3 factors) — lower risk = higher score
RISK_FACTOR_WEIGHTS = {
    "volatility_score":        0.40,
    "drawdown_score":          0.35,
    "downside_deviation_score": 0.25,
}

# Within VOLUME (2 factors)
VOLUME_FACTOR_WEIGHTS = {
    "avg_volume_score":  0.40,
    "volume_surge_score": 0.60,
}

# ── Valuation placeholders ────────────────────────────────────────────────────
# Reserved for Phase 3 fundamentals
VALUATION_FACTOR_WEIGHTS = {
    "pe_score": 0.40,
    "pb_score": 0.35,
    "dy_score": 0.25,
}