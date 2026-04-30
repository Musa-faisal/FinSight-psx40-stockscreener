"""
verdict.py
----------
Phase 5B — Maps composite_score to final_verdict and legacy verdict.

Tiers (final_verdict)
---------------------
  80–100    →  Strong Buy
  65–79.99  →  Buy
  50–64.99  →  Hold
  35–49.99  →  Weak
  0–34.99   →  Avoid

Backward compatibility
----------------------
  verdict            — same label as final_verdict  (legacy column)
  verdict_emoji      — emoji for the verdict
  verdict_color      — hex colour for charts
  verdict_rationale  — one-line explanation
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


# ═══════════════════════════════════════════════════════════════════
# VERDICT DATACLASS
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Verdict:
    label:      str
    emoji:      str
    color:      str
    rationale:  str


# ═══════════════════════════════════════════════════════════════════
# TIER TABLE  — ordered highest → lowest threshold
# ═══════════════════════════════════════════════════════════════════

_TIERS: list[tuple[float, Verdict]] = [
    (80.0, Verdict(
        label="Strong Buy",
        emoji="🟢",
        color="#00C853",
        rationale=(
            "Strong technical trend, solid fundamentals, and "
            "low risk profile — high-conviction entry signal."
        ),
    )),
    (65.0, Verdict(
        label="Buy",
        emoji="🔵",
        color="#2979FF",
        rationale=(
            "Positive factor tilt across technical and fundamental "
            "pillars with manageable risk."
        ),
    )),
    (50.0, Verdict(
        label="Hold",
        emoji="🟡",
        color="#FFD600",
        rationale=(
            "Mixed signals across pillars — "
            "no strong directional edge; monitor for confirmation."
        ),
    )),
    (35.0, Verdict(
        label="Weak",
        emoji="🟠",
        color="#FF6D00",
        rationale=(
            "Deteriorating technical or fundamental signals with "
            "elevated risk — exercise caution."
        ),
    )),
    (0.0, Verdict(
        label="Avoid",
        emoji="🔴",
        color="#D50000",
        rationale=(
            "Poor scores across multiple pillars and high risk — "
            "no entry signal present."
        ),
    )),
]


# ═══════════════════════════════════════════════════════════════════
# INTERNAL LOOKUP
# ═══════════════════════════════════════════════════════════════════

def _tier_for(score: float) -> Verdict:
    for threshold, verdict in _TIERS:
        if score >= threshold:
            return verdict
    return _TIERS[-1][1]


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API — scalar helpers
# ═══════════════════════════════════════════════════════════════════

def get_verdict(score: float | None) -> Verdict:
    """Return the Verdict object for a given composite score."""
    if score is None:
        return _tier_for(50.0)
    try:
        return _tier_for(float(score))
    except (TypeError, ValueError):
        return _tier_for(50.0)


def verdict_label(score: float | None) -> str:
    return get_verdict(score).label


def verdict_emoji(score: float | None) -> str:
    return get_verdict(score).emoji


def verdict_color(score: float | None) -> str:
    return get_verdict(score).color


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API — DataFrame enrichment
# ═══════════════════════════════════════════════════════════════════

def apply_verdicts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all verdict columns to a scored DataFrame.

    Score column priority
    ---------------------
    composite_score  >  demo_score

    Columns added
    -------------
    final_verdict       — Phase 5B primary verdict label
    verdict             — same as final_verdict  (legacy alias)
    verdict_emoji
    verdict_color
    verdict_rationale
    """
    # Determine which score column to read from
    score_col = (
        "composite_score" if "composite_score" in df.columns
        else "demo_score"
    )

    if score_col not in df.columns:
        raise ValueError(
            "DataFrame must contain 'composite_score' or 'demo_score'."
        )

    verdicts = df[score_col].apply(get_verdict)

    df = df.copy()
    df["final_verdict"]     = verdicts.apply(lambda v: v.label)
    df["verdict"]           = df["final_verdict"]          # legacy alias
    df["verdict_emoji"]     = verdicts.apply(lambda v: v.emoji)
    df["verdict_color"]     = verdicts.apply(lambda v: v.color)
    df["verdict_rationale"] = verdicts.apply(lambda v: v.rationale)

    return df