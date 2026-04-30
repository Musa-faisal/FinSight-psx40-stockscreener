"""
engine.py
---------
Central orchestrator for the PSX40 screener.

Phase 8 update
--------------
Added preset screener support via get_preset_names() and get_preset().
All existing public methods and signatures are unchanged.

Pipeline
--------
  1. Init DB
  2. Seed universe + OHLCV if needed
  3. Build technical indicator table
  4. Merge fundamental metrics          (Phase 5A)
  5. Re-score using merged row data     (Phase 5B)
  6. Add sector benchmark metrics       (Phase 7)
  7. Apply verdicts
  8. Cache results

Backward compatibility
----------------------
  demo_score  is still present (alias for composite_score).
  verdict     is still present (alias for final_verdict).
  All existing public methods and signatures are unchanged.
"""

from __future__ import annotations

import pandas as pd

from src.database.connection import init_db
from src.database.db_manager import (
    load_universe,
    load_prices,
    upsert_universe,
    upsert_prices,
    get_available_tickers,
    db_summary,
)
from src.ingestion.sample_data_generator import generate_all_sample_data
from src.ingestion.data_validator import validate_and_raise
from src.analysis.technical import compute_all_indicators, get_latest_indicators
from src.analysis.scoring_model import (
    score_all_tickers,
    compute_technical_score,
    compute_fundamental_score,
    compute_risk_score,
    compute_composite_score,
)
from src.analysis.verdict import apply_verdicts
from src.screener.presets import PRESET_NAMES, get_preset_names, apply_preset
from config.stock_universe import PSX40_UNIVERSE


# NOTE: src.analysis.sector_benchmark is imported lazily inside
# _run_sector_benchmarks() and get_sector_benchmarks() so that a
# missing or broken sector_benchmark module never prevents the rest
# of the engine from loading.


# ── Fundamental columns merged from Phase 5A ─────────────────────

_FUNDAMENTAL_MERGE_COLS = [
    "pe_ratio",
    "pb_ratio",
    "roe",
    "debt_to_equity",
    "dividend_yield",
    "net_profit_margin",
    "payout_ratio",
    "data_quality_score",
    "fundamental_notes",
    # raw financials also available for risk scoring
    "eps",
    "total_equity",
    "total_debt",
    "revenue",
    "net_profit",
]

# ── Preferred column order in the final screener DataFrame ───────

_FRONT_COLS = [
    "ticker", "name", "sector",
    "latest_close",
    "sma_20", "sma_50", "sma_200",
    "rsi_14",
    "macd_hist",
    "return_1m", "return_3m", "return_6m",
    "breakout_ratio",
    "volatility_30d", "volatility",
    "volume_surge", "volume_ratio",
    "avg_volume_20d",
    # scores
    "technical_score",
    "fundamental_score",
    "risk_score",
    "composite_score",
    "demo_score",
    # verdict
    "final_verdict",
    "verdict",
    "verdict_emoji",
    "verdict_color",
    "verdict_rationale",
    # fundamentals
    "pe_ratio", "pb_ratio", "roe",
    "debt_to_equity", "dividend_yield",
    "net_profit_margin", "payout_ratio",
    "data_quality_score", "fundamental_notes",
    # sector benchmarking (Phase 7)
    "sector_avg_composite_score", "composite_score_vs_sector",
    "sector_avg_risk_score",      "risk_score_vs_sector",
    "sector_value_label",
    "sector_avg_pe",    "sector_median_pe",    "pe_vs_sector",
    "sector_avg_pb",                           "pb_vs_sector",
    "sector_avg_roe",                          "roe_vs_sector",
    "sector_avg_dividend_yield",               "dividend_yield_vs_sector",
    "sector_stock_count",
]


class ScreenerEngine:
    """
    Orchestrates data seeding, indicator computation, three-pillar
    scoring, fundamental merging, sector benchmarking, and filtering
    for the PSX40 dashboard.
    """

    def __init__(self) -> None:
        self._screener_df:  pd.DataFrame = pd.DataFrame()
        self._price_cache:  dict[str, pd.DataFrame] = {}
        self._universe_df:  pd.DataFrame = pd.DataFrame()
        self._is_ready:     bool = False

    # ═══════════════════════════════════════════════════════════════
    # PUBLIC — INITIALISATION
    # ═══════════════════════════════════════════════════════════════

    def run(self, force_reseed: bool = False) -> None:
        """
        Full pipeline — seed → indicators → fundamentals → scoring
                      → sector benchmarks → verdicts.
        """
        print("[engine] Initialising database...")
        init_db()

        self._seed_if_needed(force=force_reseed)

        print("[engine] Loading universe...")
        self._universe_df = load_universe()

        print("[engine] Computing technical indicators...")
        technical_df = self._build_technical_indicator_table()

        print("[engine] Merging fundamental metrics...")
        merged_df = self._merge_fundamentals(technical_df)

        print("[engine] Running three-pillar scoring...")
        scored_df = self._run_full_scoring(merged_df)

        print("[engine] Applying sector benchmarks...")
        benchmarked_df = self._run_sector_benchmarks(scored_df)

        print("[engine] Applying verdicts...")
        self._screener_df = self._finalise_columns(
            apply_verdicts(benchmarked_df)
        )

        self._is_ready = True

        s = db_summary()
        print(
            f"[engine] Ready — "
            f"{s.get('tickers_with_prices', 0)} tickers | "
            f"{s['price_rows']:,} price rows | "
            f"{s.get('tickers_with_financials', 0)} with fundamentals"
        )

    # ═══════════════════════════════════════════════════════════════
    # PRIVATE — SEEDING
    # ═══════════════════════════════════════════════════════════════

    def _seed_if_needed(self, force: bool = False) -> None:
        tickers = get_available_tickers()
        if not force and tickers:
            print(
                f"[engine] DB has data for {len(tickers)} ticker(s) "
                f"— skipping seed."
            )
            return

        print("[engine] Seeding stock universe...")
        upsert_universe(PSX40_UNIVERSE)

        print("[engine] Generating sample OHLCV data...")
        ohlcv_df = generate_all_sample_data()
        validate_and_raise(ohlcv_df, source="sample_generator")
        upsert_prices(ohlcv_df)
        print("[engine] Seed complete.")

    # ═══════════════════════════════════════════════════════════════
    # PRIVATE — TECHNICAL INDICATOR TABLE
    # ═══════════════════════════════════════════════════════════════

    def _build_technical_indicator_table(self) -> pd.DataFrame:
        """
        For each ticker compute indicators, extract the latest row,
        run the Phase-3 technical-only scorer to seed score columns,
        and merge universe metadata.
        """
        tickers        = get_available_tickers()
        indicator_rows = []

        for ticker in tickers:
            df = load_prices(ticker)
            if df.empty or len(df) < 20:
                print(
                    f"  [engine] Skipping {ticker} — "
                    f"only {len(df)} row(s)"
                )
                continue

            df_ind = compute_all_indicators(df)
            self._price_cache[ticker] = df_ind

            latest           = get_latest_indicators(df_ind)
            latest["ticker"] = ticker
            indicator_rows.append(latest)

        if not indicator_rows:
            return pd.DataFrame()

        # Initial score pass (technical-only at this stage)
        scored_df = score_all_tickers(indicator_rows)

        # Merge universe metadata
        scored_df = scored_df.merge(
            self._universe_df[["ticker", "name", "sector"]],
            on="ticker",
            how="left",
        )

        return scored_df.reset_index(drop=True)

    # ═══════════════════════════════════════════════════════════════
    # PRIVATE — FUNDAMENTAL MERGE
    # ═══════════════════════════════════════════════════════════════

    def _merge_fundamentals(self, technical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Left-join fundamental metrics onto the technical table.
        Missing fundamentals produce NaN columns — never a crash.

        Uses the correct module name: src.analysis.fundamentals
        (NOT src.analysis.fundamental).
        """
        if technical_df.empty:
            return technical_df

        try:
            from src.analysis.fundamentals import build_fundamental_metrics  # noqa: PLC0415

            fund_df = build_fundamental_metrics()

            if fund_df.empty:
                print(
                    "[engine] No fundamental data — "
                    "fundamental_score will be penalised."
                )
                return self._add_empty_fundamental_cols(technical_df)

            # Columns to bring across (only those present in fund_df)
            available = [
                c for c in _FUNDAMENTAL_MERGE_COLS if c in fund_df.columns
            ]

            # Drop columns already in technical_df to avoid _x/_y clashes
            already_present = [
                c for c in available if c in technical_df.columns
            ]
            fund_slim = fund_df[["ticker"] + available].copy()
            if already_present:
                fund_slim = fund_slim.drop(columns=already_present)

            merged = technical_df.merge(fund_slim, on="ticker", how="left")

            for col in _FUNDAMENTAL_MERGE_COLS:
                if col not in merged.columns:
                    merged[col] = None

            n_matched = (
                int(merged["pe_ratio"].notna().sum())
                if "pe_ratio" in merged.columns
                else 0
            )
            print(
                f"[engine] Fundamental merge — "
                f"{n_matched}/{len(merged)} tickers have ratio data."
            )
            return merged.reset_index(drop=True)

        except Exception as exc:
            print(
                f"[engine] Warning: fundamental merge failed ({exc}). "
                f"Continuing with technical-only table."
            )
            return self._add_empty_fundamental_cols(technical_df)

    @staticmethod
    def _add_empty_fundamental_cols(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in _FUNDAMENTAL_MERGE_COLS:
            if col not in df.columns:
                df[col] = None
        return df

    # ═══════════════════════════════════════════════════════════════
    # PRIVATE — FULL THREE-PILLAR SCORING  (Phase 5B)
    # ═══════════════════════════════════════════════════════════════

    def _run_full_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Re-compute technical_score, fundamental_score, risk_score, and
        composite_score now that both technical indicators and fundamental
        metrics are available on every row.

        This overwrites the preliminary scores set during
        _build_technical_indicator_table().
        """
        if df.empty:
            return df

        df = df.copy()

        tech_scores  = []
        fund_scores  = []
        risk_scores  = []
        comp_scores  = []

        for _, row in df.iterrows():
            row_dict = row.to_dict()

            tech = compute_technical_score(row_dict)
            fund = compute_fundamental_score(row_dict)
            risk = compute_risk_score(row_dict)
            comp = compute_composite_score(tech, fund, risk)

            tech_scores.append(tech)
            fund_scores.append(fund)
            risk_scores.append(risk)
            comp_scores.append(comp)

        df["technical_score"]   = tech_scores
        df["fundamental_score"] = fund_scores
        df["risk_score"]        = risk_scores
        df["composite_score"]   = comp_scores
        df["demo_score"]        = df["composite_score"]  # backward compat

        df.sort_values("composite_score", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    # ═══════════════════════════════════════════════════════════════
    # PRIVATE — SECTOR BENCHMARKS  (Phase 7)
    # ═══════════════════════════════════════════════════════════════

    def _run_sector_benchmarks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply sector benchmark metrics to the scored DataFrame.

        Wrapped in a broad try/except so that any unexpected failure
        in sector_benchmark.py never crashes the screener — it simply
        continues without sector columns.
        """
        if df.empty:
            return df

        try:
            from src.analysis.sector_benchmark import apply_sector_benchmarks  # noqa: PLC0415

            return apply_sector_benchmarks(df)

        except Exception as exc:
            print(
                f"[engine] Warning: sector benchmarking failed ({exc}). "
                f"Continuing without sector benchmarks."
            )
            return df

    # ═══════════════════════════════════════════════════════════════
    # PRIVATE — COLUMN ORDERING
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _finalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder columns so priority columns appear first,
        followed by any remaining columns alphabetically.
        """
        front = [c for c in _FRONT_COLS if c in df.columns]
        rest  = sorted([c for c in df.columns if c not in front])
        return df[front + rest].reset_index(drop=True)

    # ═══════════════════════════════════════════════════════════════
    # PUBLIC — ACCESSORS  (all unchanged from Phase 5A)
    # ═══════════════════════════════════════════════════════════════

    def get_screener_df(self) -> pd.DataFrame:
        """Return the full scored and fundamental-enriched DataFrame."""
        self._check_ready()
        return self._screener_df.copy()

    def get_price_df(self, ticker: str) -> pd.DataFrame:
        """
        Return OHLCV + indicator DataFrame for a single ticker.
        Falls back to a fresh DB load + compute if not cached.
        """
        self._check_ready()
        ticker = ticker.upper()

        if ticker not in self._price_cache:
            df = load_prices(ticker)
            if df.empty:
                return pd.DataFrame()
            self._price_cache[ticker] = compute_all_indicators(df)

        return self._price_cache[ticker].copy()

    def get_sectors(self) -> list[str]:
        self._check_ready()
        if "sector" not in self._screener_df.columns:
            return []
        return sorted(self._screener_df["sector"].dropna().unique().tolist())

    def get_tickers(self) -> list[str]:
        self._check_ready()
        if "ticker" not in self._screener_df.columns:
            return []
        return sorted(self._screener_df["ticker"].tolist())

    def get_sector_benchmarks(self) -> pd.DataFrame:
        """Return one row per sector with Phase 7 benchmark metrics."""
        self._check_ready()
        try:
            from src.analysis.sector_benchmark import build_sector_metrics  # noqa: PLC0415

            return build_sector_metrics(self._screener_df).copy()
        except Exception as exc:
            print(
                f"[engine] Warning: get_sector_benchmarks failed ({exc})."
            )
            return pd.DataFrame()

    def get_preset_names(self) -> list[str]:
        """Return the list of available preset screener names."""
        return get_preset_names()

    def get_preset(self, preset_name: str) -> pd.DataFrame:
        """
        Return a filtered DataFrame for a named preset.

        Parameters
        ----------
        preset_name : str
            One of the names returned by get_preset_names().
        """
        self._check_ready()
        return apply_preset(self._screener_df, preset_name)

    def filter(
        self,
        sectors:   list[str] | None = None,
        min_score: float = 0.0,
        max_score: float = 100.0,
        verdicts:  list[str] | None = None,
        tickers:   list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Return a filtered slice of the screener DataFrame.

        Parameters
        ----------
        sectors   : keep only these sectors  (None = all)
        min_score : minimum composite_score inclusive
        max_score : maximum composite_score inclusive
        verdicts  : filter on final_verdict label  (None = all)
        tickers   : keep only these tickers  (None = all)
        """
        self._check_ready()
        df        = self._screener_df.copy()
        score_col = (
            "composite_score" if "composite_score" in df.columns
            else "demo_score"
        )
        verdict_col = (
            "final_verdict" if "final_verdict" in df.columns
            else "verdict"
        )

        if sectors:
            df = df[df["sector"].isin(sectors)]
        if tickers:
            df = df[df["ticker"].isin(tickers)]
        if verdicts:
            df = df[df[verdict_col].isin(verdicts)]
        if score_col in df.columns:
            df = df[
                df[score_col].fillna(0).between(
                    min_score, max_score, inclusive="both"
                )
            ]

        return df.reset_index(drop=True)

    def summary(self) -> dict:
        """Return a quick-stats dict for the dashboard KPI bar."""
        self._check_ready()
        df        = self._screener_df
        score_col = (
            "composite_score" if "composite_score" in df.columns
            else "demo_score"
        )
        return {
            "total_tickers": len(df),
            "sectors":       df["sector"].nunique()
                             if "sector" in df.columns else 0,
            "avg_score":     round(df[score_col].mean(), 1)
                             if score_col in df.columns else 0.0,
            "top_ticker":    df.iloc[0]["ticker"]  if not df.empty else "—",
            "top_score":     df.iloc[0][score_col] if not df.empty else 0.0,
        }

    # ═══════════════════════════════════════════════════════════════
    # PRIVATE — GUARD
    # ═══════════════════════════════════════════════════════════════

    def _check_ready(self) -> None:
        if not self._is_ready:
            raise RuntimeError(
                "ScreenerEngine is not ready. Call engine.run() first."
            )
