"""
scoring_model.py
----------------
Phase 5B — Professional three-pillar scoring for PSX40 screener.

Scores produced
---------------
  technical_score    (0–100)  — trend, momentum, volume, volatility
  fundamental_score  (0–100)  — quality, valuation, income
  risk_score         (0–100)  — higher = riskier
  composite_score    (0–100)  — weighted blend of all three pillars
  demo_score                  — alias for composite_score (backward compat)

Composite formula
-----------------
  composite = technical * 0.40
            + fundamental * 0.40
            + (100 - risk) * 0.20

All sub-scorers return 0–100 and treat NaN inputs as a neutral
default (50) unless stated otherwise.  No scorer raises an exception.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import RSI_PERIOD


# ═══════════════════════════════════════════════════════════════════
# PILLAR WEIGHTS
# ═══════════════════════════════════════════════════════════════════

_W_TECHNICAL    = 0.40
_W_FUNDAMENTAL  = 0.40
_W_RISK_INVERSE = 0.20   # we use (100 - risk_score) so low risk helps

_NEUTRAL = 50.0


# ═══════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════

def _safe(value, default: float = _NEUTRAL) -> float:
    """Return float(value) or default when value is None / NaN."""
    try:
        f = float(value)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(np.clip(value, lo, hi))


def _interp(value: float, x_low: float, x_high: float,
            y_low: float = 0.0, y_high: float = 100.0) -> float:
    """Linear interpolation clamped to [y_low, y_high]."""
    return float(np.clip(
        np.interp(value, [x_low, x_high], [y_low, y_high]),
        y_low, y_high,
    ))


# ═══════════════════════════════════════════════════════════════════
# TECHNICAL SUB-SCORERS
# ═══════════════════════════════════════════════════════════════════

def _ts_sma_trend(row: dict) -> float:
    """
    Trend score from SMA flags and breakout position.
    Each binary flag contributes equally; breakout adds a boost.
    """
    score = 0.0
    flags = [
        row.get("price_above_sma20", None),
        row.get("sma20_above_sma50", None),
        row.get("sma50_above_sma200", None),
    ]
    valid = [f for f in flags if f is not None]
    if valid:
        score += (sum(int(f) for f in valid) / len(valid)) * 50.0

    br = _safe(row.get("breakout_ratio"), default=None)
    if br is not None:
        score += _interp(br, 0.70, 1.0, 0.0, 50.0)
    else:
        score += 25.0   # neutral if unavailable

    return _clamp(score)


def _ts_rsi(row: dict) -> float:
    """
    RSI score — peaks at 55 (healthy uptrend), penalises extremes.
    """
    rsi = _safe(row.get(f"rsi_{RSI_PERIOD}"), default=None)
    if rsi is None:
        return _NEUTRAL
    if rsi <= 20:
        return _interp(rsi, 0, 20, 0, 15)
    elif rsi <= 45:
        return _interp(rsi, 20, 45, 15, 70)
    elif rsi <= 65:
        return _interp(rsi, 45, 65, 70, 100)
    elif rsi <= 80:
        return _interp(rsi, 65, 80, 100, 35)
    else:
        return _interp(rsi, 80, 100, 35, 5)


def _ts_volume(row: dict) -> float:
    """Volume surge ratio — above-average volume is rewarded."""
    surge = _safe(row.get("volume_surge", row.get("volume_ratio")), default=None)
    if surge is None:
        return _NEUTRAL
    return _clamp(_interp(surge, 0.0, 3.0, 0.0, 100.0))


def _ts_volatility(row: dict) -> float:
    """
    Moderate volatility is ideal.
    Tent function peaking at ~22% annualised.
    """
    vol = _safe(row.get("volatility_30d", row.get("volatility")), default=None)
    if vol is None:
        return _NEUTRAL
    pct = vol * 100
    if pct <= 5:
        return _interp(pct, 0, 5, 0, 10)
    elif pct <= 22:
        return _interp(pct, 5, 22, 10, 100)
    elif pct <= 45:
        return _interp(pct, 22, 45, 100, 55)
    else:
        return _clamp(_interp(pct, 45, 80, 55, 0))


def _ts_momentum(row: dict) -> float:
    """
    Return-based momentum: average of 1M, 3M, 6M scores.
    Uses MACD histogram as a supplement when returns are absent.
    """
    scores = []

    for key, lo, hi in [
        ("return_1m", -0.10, 0.10),
        ("return_3m", -0.20, 0.20),
        ("return_6m", -0.30, 0.30),
    ]:
        v = _safe(row.get(key), default=None)
        if v is not None:
            scores.append(_interp(v, lo, hi))

    macd_hist = _safe(row.get("macd_hist"), default=None)
    if macd_hist is not None:
        scores.append(100.0 if macd_hist > 0 else 0.0)

    return float(np.mean(scores)) if scores else _NEUTRAL


def compute_technical_score(row: dict) -> float:
    """
    Weighted blend of five technical sub-scores.

    Weights
    -------
    SMA trend   30%
    RSI         25%
    Momentum    20%
    Volume      15%
    Volatility  10%
    """
    ts = (
        _ts_sma_trend(row)   * 0.30
        + _ts_rsi(row)       * 0.25
        + _ts_momentum(row)  * 0.20
        + _ts_volume(row)    * 0.15
        + _ts_volatility(row)* 0.10
    )
    return round(_clamp(ts), 2)


# ═══════════════════════════════════════════════════════════════════
# FUNDAMENTAL SUB-SCORERS
# ═══════════════════════════════════════════════════════════════════

def _fs_roe(row: dict) -> float:
    """ROE (%): higher is better.  Negative → 0."""
    roe = _safe(row.get("roe"), default=None)
    if roe is None:
        return _NEUTRAL
    if roe < 0:
        return 0.0
    return _clamp(_interp(roe, 0, 30, 0, 100))


def _fs_pe(row: dict) -> float:
    """
    P/E: reasonable range scores best.
    NaN (missing / negative EPS) → neutral.
    Very high P/E → penalised.
    """
    pe = _safe(row.get("pe_ratio"), default=None)
    if pe is None:
        return _NEUTRAL
    if pe <= 0:
        return 10.0
    elif pe <= 8:
        return _interp(pe, 0, 8, 40, 80)
    elif pe <= 18:
        return _interp(pe, 8, 18, 80, 100)
    elif pe <= 30:
        return _interp(pe, 18, 30, 100, 60)
    elif pe <= 50:
        return _interp(pe, 30, 50, 60, 25)
    else:
        return _clamp(_interp(pe, 50, 100, 25, 0))


def _fs_pb(row: dict) -> float:
    """
    P/B: below book is cheap, very high is expensive.
    NaN (missing / negative BVPS) → neutral.
    """
    pb = _safe(row.get("pb_ratio"), default=None)
    if pb is None:
        return _NEUTRAL
    if pb <= 0:
        return 10.0
    elif pb <= 1:
        return _interp(pb, 0, 1, 60, 100)
    elif pb <= 3:
        return _interp(pb, 1, 3, 100, 65)
    elif pb <= 6:
        return _interp(pb, 3, 6, 65, 30)
    else:
        return _clamp(_interp(pb, 6, 15, 30, 0))


def _fs_dividend_yield(row: dict) -> float:
    """
    Dividend yield (%): moderate is best.
    Zero is neutral; very high may signal distress.
    """
    dy = _safe(row.get("dividend_yield"), default=None)
    if dy is None:
        return _NEUTRAL
    if dy <= 0:
        return 35.0
    elif dy <= 3:
        return _interp(dy, 0, 3, 35, 70)
    elif dy <= 7:
        return _interp(dy, 3, 7, 70, 100)
    elif dy <= 12:
        return _interp(dy, 7, 12, 100, 60)
    else:
        return _clamp(_interp(dy, 12, 25, 60, 10))


def _fs_debt_to_equity(row: dict) -> float:
    """Low D/E is safer and scores higher."""
    de = _safe(row.get("debt_to_equity"), default=None)
    if de is None:
        return _NEUTRAL
    if de < 0:
        return 5.0          # negative equity — very risky
    return _clamp(_interp(de, 0, 3, 100, 0))


def _fs_profit_margin(row: dict) -> float:
    """Net profit margin (%): higher is better; losses score 0."""
    pm = _safe(row.get("net_profit_margin"), default=None)
    if pm is None:
        return _NEUTRAL
    if pm < 0:
        return 0.0
    return _clamp(_interp(pm, 0, 30, 0, 100))


def _fs_payout_ratio(row: dict) -> float:
    """
    Payout ratio (%): 30–60% is healthy.
    Zero = company pays no dividend (neutral-ish).
    Over 100% = paying more than earnings (risky).
    """
    pr = _safe(row.get("payout_ratio"), default=None)
    if pr is None:
        return _NEUTRAL
    if pr <= 0:
        return 40.0
    elif pr <= 30:
        return _interp(pr, 0, 30, 40, 75)
    elif pr <= 60:
        return _interp(pr, 30, 60, 75, 100)
    elif pr <= 90:
        return _interp(pr, 60, 90, 100, 50)
    elif pr <= 100:
        return _interp(pr, 90, 100, 50, 20)
    else:
        return _clamp(_interp(pr, 100, 200, 20, 0))


def compute_fundamental_score(row: dict) -> float:
    """
    Weighted blend of seven fundamental sub-scores.

    When fundamental data is absent the data_quality_score
    penalty is applied before returning.

    Weights
    -------
    ROE              25%
    P/E              20%
    Profit margin    20%
    P/B              15%
    Debt/equity      10%
    Dividend yield    5%
    Payout ratio      5%
    """
    fs = (
        _fs_roe(row)             * 0.25
        + _fs_pe(row)            * 0.20
        + _fs_profit_margin(row) * 0.20
        + _fs_pb(row)            * 0.15
        + _fs_debt_to_equity(row)* 0.10
        + _fs_dividend_yield(row)* 0.05
        + _fs_payout_ratio(row)  * 0.05
    )

    # Scale down by data quality so missing fundamentals reduce score
    dq = _safe(row.get("data_quality_score"), default=70.0)
    dq_factor = _clamp(dq / 100.0, 0.0, 1.0)

    # Apply penalty: missing data drags score toward 50 (neutral)
    fs_adjusted = fs * dq_factor + _NEUTRAL * (1 - dq_factor)

    return round(_clamp(fs_adjusted), 2)


# ═══════════════════════════════════════════════════════════════════
# RISK SUB-SCORERS  (higher = riskier)
# ═══════════════════════════════════════════════════════════════════

def _rs_volatility(row: dict) -> float:
    """High volatility → high risk score."""
    vol = _safe(row.get("volatility_30d", row.get("volatility")), default=None)
    if vol is None:
        return _NEUTRAL
    pct = vol * 100
    return _clamp(_interp(pct, 5, 60, 0, 100))


def _rs_negative_eps(row: dict) -> float:
    """Negative or zero EPS is a risk signal."""
    eps = _safe(row.get("eps"), default=None)
    if eps is None:
        return 40.0
    return 80.0 if eps <= 0 else 0.0


def _rs_negative_equity(row: dict) -> float:
    """Negative equity is a serious risk signal."""
    eq = _safe(row.get("total_equity"), default=None)
    if eq is None:
        return 40.0
    return 90.0 if eq <= 0 else 0.0


def _rs_debt_burden(row: dict) -> float:
    """High D/E ratio contributes to risk score."""
    de = _safe(row.get("debt_to_equity"), default=None)
    if de is None:
        return _NEUTRAL
    if de < 0:
        return 90.0
    return _clamp(_interp(de, 0, 3, 0, 100))


def _rs_data_quality(row: dict) -> float:
    """Low data quality is a proxy for information risk."""
    dq = _safe(row.get("data_quality_score"), default=70.0)
    return _clamp(_interp(dq, 0, 100, 80, 0))


def _rs_payout_risk(row: dict) -> float:
    """Payout ratio above 100% signals unsustainable dividends."""
    pr = _safe(row.get("payout_ratio"), default=None)
    if pr is None:
        return 20.0
    if pr <= 80:
        return 0.0
    elif pr <= 100:
        return _interp(pr, 80, 100, 0, 50)
    else:
        return _clamp(_interp(pr, 100, 200, 50, 100))


def _rs_low_volume(row: dict) -> float:
    """Very low volume surge → potential illiquidity risk."""
    surge = _safe(row.get("volume_surge", row.get("volume_ratio")), default=None)
    if surge is None:
        return 20.0
    return _clamp(_interp(surge, 0.0, 0.5, 60, 0))


def compute_risk_score(row: dict) -> float:
    """
    Weighted blend of seven risk sub-scores.
    Output: 0 = very low risk, 100 = very high risk.

    Weights
    -------
    Negative equity   25%
    Negative EPS      20%
    Debt burden       20%
    Volatility        15%
    Data quality      10%
    Payout risk        5%
    Low volume         5%
    """
    rs = (
        _rs_negative_equity(row) * 0.25
        + _rs_negative_eps(row)  * 0.20
        + _rs_debt_burden(row)   * 0.20
        + _rs_volatility(row)    * 0.15
        + _rs_data_quality(row)  * 0.10
        + _rs_payout_risk(row)   * 0.05
        + _rs_low_volume(row)    * 0.05
    )
    return round(_clamp(rs), 2)


# ═══════════════════════════════════════════════════════════════════
# COMPOSITE SCORE
# ═══════════════════════════════════════════════════════════════════

def compute_composite_score(
    technical_score:    float,
    fundamental_score:  float,
    risk_score:         float,
) -> float:
    """
    composite = technical  * 0.40
              + fundamental * 0.40
              + (100 - risk) * 0.20
    """
    composite = (
        _safe(technical_score)   * _W_TECHNICAL
        + _safe(fundamental_score)  * _W_FUNDAMENTAL
        + (100.0 - _safe(risk_score)) * _W_RISK_INVERSE
    )
    return round(_clamp(composite), 2)


# ═══════════════════════════════════════════════════════════════════
# ROW-LEVEL SCORER  (technical + fundamental + risk in one call)
# ═══════════════════════════════════════════════════════════════════

def compute_score(row: dict) -> dict:
    """
    Compute all four scores for a single ticker row dict.

    Returns
    -------
    dict with keys:
        technical_score, fundamental_score, risk_score,
        composite_score, demo_score
    """
    tech  = compute_technical_score(row)
    fund  = compute_fundamental_score(row)
    risk  = compute_risk_score(row)
    comp  = compute_composite_score(tech, fund, risk)

    return {
        "technical_score":   tech,
        "fundamental_score": fund,
        "risk_score":        risk,
        "composite_score":   comp,
        "demo_score":        comp,   # backward-compat alias
    }


# ═══════════════════════════════════════════════════════════════════
# BATCH SCORER
# ═══════════════════════════════════════════════════════════════════

def score_all_tickers(indicator_rows: list[dict]) -> pd.DataFrame:
    """
    Score a list of indicator dicts (one per ticker) and return a
    flat DataFrame with all score columns appended.

    Parameters
    ----------
    indicator_rows : list of dicts from get_latest_indicators()

    Returns
    -------
    pd.DataFrame sorted by composite_score descending.
    """
    results = []
    for row in indicator_rows:
        scores   = compute_score(row)
        flat_row = {**row, **scores}
        results.append(flat_row)

    df = pd.DataFrame(results)

    if "composite_score" in df.columns:
        df.sort_values("composite_score", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df
