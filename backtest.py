"""
backtest.py
-----------
Phase 10 — Portfolio Backtester for PSX40 Screener.

Public API
----------
    build_price_matrix(engine, tickers)          -> pd.DataFrame
    calculate_portfolio_returns(matrix, weights) -> pd.Series
    calculate_max_drawdown(cumulative_returns)   -> float
    backtest_portfolio(engine, portfolio_df)     -> dict
    backtest_to_dataframe(result)               -> pd.DataFrame

Result dict structure
---------------------
    {
        "status":           "ok" | "warning" | "error",
        "metrics": {
            "cumulative_return":     float | None,
            "annualized_return":     float | None,
            "annualized_volatility": float | None,
            "sharpe_ratio":          float | None,
            "max_drawdown":          float | None,
            "start_date":            str   | None,
            "end_date":              str   | None,
            "trading_days":          int,
            "tickers":               list[str],
        },
        "daily_returns":    pd.Series,
        "cumulative_returns": pd.Series,
        "price_matrix":     pd.DataFrame,
        "warnings":         list[str],
    }
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────

_TRADING_DAYS = 252
_MIN_ROWS     = 30
_FFILL_LIMIT  = 5


# ═══════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════

def _to_float(value) -> float | None:
    try:
        f = float(value)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _round(value, decimals: int = 4) -> float | None:
    f = _to_float(value)
    return round(f, decimals) if f is not None else None


def _empty_metrics(tickers: list[str] | None = None) -> dict:
    """Return a metrics dict with all values set to None / defaults."""
    return {
        "cumulative_return":     None,
        "annualized_return":     None,
        "annualized_volatility": None,
        "sharpe_ratio":          None,
        "max_drawdown":          None,
        "start_date":            None,
        "end_date":              None,
        "trading_days":          0,
        "tickers":               tickers or [],
    }


def _empty_result(
    status:  str = "error",
    warns:   list[str] | None = None,
    tickers: list[str] | None = None,
) -> dict:
    """Return a fully-structured result dict with empty/None values."""
    return {
        "status":             status,
        "metrics":            _empty_metrics(tickers),
        "daily_returns":      pd.Series(dtype=float),
        "cumulative_returns": pd.Series(dtype=float),
        "price_matrix":       pd.DataFrame(),
        "warnings":           warns or [],
    }


# ═══════════════════════════════════════════════════════════════════
# 1. BUILD PRICE MATRIX
# ═══════════════════════════════════════════════════════════════════

def build_price_matrix(engine, tickers: list[str]) -> pd.DataFrame:
    """
    Load close price history for each ticker and assemble a wide
    DataFrame indexed by date.

    - Rows where ALL values are NaN are dropped.
    - Short gaps are forward-filled (limit = 5 days).
    - Missing tickers produce a warning and are skipped.
    """
    if not tickers:
        warnings.warn("[backtest] build_price_matrix: empty ticker list.")
        return pd.DataFrame()

    frames: dict[str, pd.Series] = {}

    for ticker in tickers:
        try:
            df = engine.get_price_df(ticker)

            if df is None or df.empty:
                warnings.warn(f"[backtest] No price data for {ticker} — skipping.")
                continue

            if "date" not in df.columns or "close" not in df.columns:
                warnings.warn(
                    f"[backtest] {ticker}: missing 'date' or 'close' — skipping."
                )
                continue

            series = (
                df[["date", "close"]]
                .copy()
                .assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce"))
                .dropna(subset=["date"])
                .set_index("date")["close"]
                .pipe(pd.to_numeric, errors="coerce")
                .rename(ticker)
            )
            frames[ticker] = series

        except Exception as exc:
            warnings.warn(f"[backtest] Error loading {ticker}: {exc} — skipping.")
            continue

    if not frames:
        warnings.warn("[backtest] build_price_matrix: no price data loaded.")
        return pd.DataFrame()

    matrix = pd.concat(frames.values(), axis=1)
    matrix.index = pd.to_datetime(matrix.index)
    matrix.sort_index(inplace=True)
    matrix.dropna(how="all", inplace=True)
    matrix.ffill(limit=_FFILL_LIMIT, inplace=True)

    print(
        f"[backtest] Price matrix: {matrix.shape[1]} ticker(s) | "
        f"{len(matrix)} rows | "
        f"{matrix.index[0].date()} → {matrix.index[-1].date()}"
    )
    return matrix


# ═══════════════════════════════════════════════════════════════════
# 2. CALCULATE PORTFOLIO RETURNS
# ═══════════════════════════════════════════════════════════════════

def calculate_portfolio_returns(
    price_matrix: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """
    Compute daily weighted portfolio returns.

    FutureWarning fix: uses .ffill().pct_change(fill_method=None)
    instead of the deprecated default fill_method='pad'.
    """
    if price_matrix is None or price_matrix.empty:
        warnings.warn("[backtest] calculate_portfolio_returns: empty price matrix.")
        return pd.Series(dtype=float)

    if not weights:
        warnings.warn("[backtest] calculate_portfolio_returns: empty weights dict.")
        return pd.Series(dtype=float)

    valid_tickers = [t for t in weights if t in price_matrix.columns]
    missing       = [t for t in weights if t not in price_matrix.columns]

    if missing:
        warnings.warn(f"[backtest] Tickers not in price matrix (skipped): {missing}")

    if not valid_tickers:
        warnings.warn("[backtest] No valid tickers after filtering.")
        return pd.Series(dtype=float)

    # Renormalise weights
    raw_w = np.array([weights[t] for t in valid_tickers], dtype=float)
    total = raw_w.sum()
    if total <= 0 or np.isnan(total):
        warnings.warn("[backtest] Weight sum is zero — falling back to equal weight.")
        raw_w = np.ones(len(valid_tickers))
        total = float(len(valid_tickers))
    norm_w = raw_w / total

    # ── FutureWarning fix ─────────────────────────────────────────
    # Old (deprecated): prices.pct_change()
    # New (explicit):   prices.ffill().pct_change(fill_method=None)
    prices  = price_matrix[valid_tickers].copy()
    returns = prices.ffill().pct_change(fill_method=None)
    returns.fillna(0, inplace=True)

    port_returns = (returns * norm_w).sum(axis=1)
    port_returns.name = "portfolio_return"

    return port_returns


# ═══════════════════════════════════════════════════════════════════
# 3. CALCULATE MAX DRAWDOWN
# ═══════════════════════════════════════════════════════════════════

def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Compute maximum peak-to-trough drawdown.
    Returns 0.0 when input is empty or invalid.
    Result is a negative decimal, e.g. -0.25 = -25%.
    """
    if cumulative_returns is None or cumulative_returns.empty:
        return 0.0

    cum = pd.to_numeric(cumulative_returns, errors="coerce").dropna()
    if len(cum) < 2:
        return 0.0

    rolling_peak = cum.cummax()

    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = (cum - rolling_peak) / rolling_peak.replace(0, np.nan)

    drawdowns = drawdowns.replace([np.inf, -np.inf], np.nan).fillna(0)
    return round(float(drawdowns.min()), 6)


# ═══════════════════════════════════════════════════════════════════
# 4. BACKTEST PORTFOLIO
# ═══════════════════════════════════════════════════════════════════

def backtest_portfolio(engine, portfolio_df: pd.DataFrame) -> dict:
    """
    Run a full historical backtest for a built portfolio.

    Parameters
    ----------
    engine       : ScreenerEngine instance (must have run() called)
    portfolio_df : output of build_portfolio() with 'ticker' + 'weight'

    Returns
    -------
    dict — always has the structure defined in the module docstring.
    Never raises — all errors are captured in "status" and "warnings".
    """
    warn_list: list[str] = []

    def _warn(msg: str) -> None:
        warn_list.append(msg)
        warnings.warn(f"[backtest] {msg}")

    # ── Validate inputs ───────────────────────────────────────────
    if portfolio_df is None or portfolio_df.empty:
        _warn("portfolio_df is empty — cannot backtest.")
        return _empty_result("error", warn_list)

    if "ticker" not in portfolio_df.columns:
        _warn("portfolio_df missing 'ticker' column.")
        return _empty_result("error", warn_list)

    if "weight" not in portfolio_df.columns:
        _warn("portfolio_df missing 'weight' column.")
        return _empty_result("error", warn_list)

    # ── Extract tickers and weights ───────────────────────────────
    tickers = portfolio_df["ticker"].dropna().str.upper().tolist()
    raw_wts = pd.to_numeric(portfolio_df["weight"], errors="coerce").fillna(0)
    weights = dict(zip(tickers, raw_wts.tolist()))

    if not tickers:
        _warn("No valid tickers found in portfolio_df.")
        return _empty_result("error", warn_list, tickers)

    # ── Build price matrix ────────────────────────────────────────
    price_matrix = build_price_matrix(engine, tickers)

    if price_matrix.empty:
        _warn("Could not build price matrix — no price data available.")
        return _empty_result("error", warn_list, tickers)

    # ── Compute daily returns ─────────────────────────────────────
    daily_returns = calculate_portfolio_returns(price_matrix, weights)

    if daily_returns.empty:
        _warn("Could not compute portfolio returns.")
        return _empty_result("error", warn_list, tickers)

    # Drop the first row (NaN from pct_change)
    daily_returns = daily_returns.iloc[1:]

    status = "ok"
    if len(daily_returns) < _MIN_ROWS:
        status = "warning"
        _warn(
            f"Only {len(daily_returns)} trading days — "
            f"metrics may be unreliable (minimum recommended: {_MIN_ROWS})."
        )

    # ── Cumulative returns ────────────────────────────────────────
    cumulative_returns = (1 + daily_returns).cumprod()

    # ── Date range ────────────────────────────────────────────────
    start_date   = str(daily_returns.index[0].date())
    end_date     = str(daily_returns.index[-1].date())
    trading_days = len(daily_returns)

    # ── Cumulative return ─────────────────────────────────────────
    final_value       = float(cumulative_returns.iloc[-1])
    cumulative_return = _round(final_value - 1.0, 4)

    # ── Annualised return ─────────────────────────────────────────
    annualized_return = None
    if trading_days >= 2:
        try:
            years             = trading_days / _TRADING_DAYS
            annualized_return = _round((final_value ** (1.0 / years)) - 1.0, 4)
        except (ValueError, ZeroDivisionError):
            _warn("Could not compute annualized return.")

    # ── Annualised volatility ─────────────────────────────────────
    daily_std = float(daily_returns.std())

    if not np.isnan(daily_std) and daily_std > 0:
        annualized_volatility = _round(daily_std * np.sqrt(_TRADING_DAYS), 4)
    else:
        annualized_volatility = None
        _warn("Annualised volatility is zero or NaN.")

    # ── Sharpe ratio ─────────────────────────────────────────────
    sharpe_ratio = None
    if (
        annualized_volatility is not None
        and annualized_volatility > 0
        and annualized_return is not None
    ):
        sharpe_ratio = _round(annualized_return / annualized_volatility, 4)

    # ── Max drawdown ──────────────────────────────────────────────
    max_drawdown = _round(calculate_max_drawdown(cumulative_returns), 4)

    # ── Assemble metrics sub-dict ─────────────────────────────────
    metrics = {
        "cumulative_return":     cumulative_return,
        "annualized_return":     annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio":          sharpe_ratio,
        "max_drawdown":          max_drawdown,
        "start_date":            start_date,
        "end_date":              end_date,
        "trading_days":          trading_days,
        "tickers":               tickers,
    }

    print(
        f"[backtest] Done — {trading_days} days | "
        f"cumulative: {(cumulative_return or 0)*100:+.2f}% | "
        f"max drawdown: {(max_drawdown or 0)*100:.2f}%"
    )

    return {
        "status":             status,
        "metrics":            metrics,
        "daily_returns":      daily_returns,
        "cumulative_returns": cumulative_returns,
        "price_matrix":       price_matrix,
        "warnings":           warn_list,
    }


# ═══════════════════════════════════════════════════════════════════
# 5. BACKTEST TO DATAFRAME
# ═══════════════════════════════════════════════════════════════════

def backtest_to_dataframe(result: dict) -> pd.DataFrame:
    """
    Convert a backtest result dict into a tidy two-column DataFrame.

    Reads all scalar values from result["metrics"].

    Returns
    -------
    pd.DataFrame with columns: metric, value
    """
    if not result:
        return pd.DataFrame(columns=["metric", "value"])

    metrics = result.get("metrics", {})
    if not metrics:
        return pd.DataFrame(columns=["metric", "value"])

    pct_keys = {
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
    }

    rows = []

    # Status row from top-level result
    rows.append({
        "metric": "Status",
        "value":  str(result.get("status", "unknown")),
    })

    for key, val in metrics.items():
        if val is None:
            display_val = "N/A"
        elif key in pct_keys and isinstance(val, float):
            display_val = f"{val * 100:+.2f}%"
        elif isinstance(val, list):
            display_val = ", ".join(str(v) for v in val)
        elif isinstance(val, float):
            display_val = f"{val:.4f}"
        else:
            display_val = str(val)

        rows.append({
            "metric": key.replace("_", " ").title(),
            "value":  display_val,
        })

    # Warnings row
    warns = result.get("warnings", [])
    rows.append({
        "metric": "Warnings",
        "value":  "; ".join(warns) if warns else "None",
    })

    return pd.DataFrame(rows).reset_index(drop=True)