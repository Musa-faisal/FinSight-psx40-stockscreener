# PROJECT_SUMMARY.md — PSX40 Stock Screener

---

## Executive Summary

PSX40 Stock Screener is a full-stack Python analytics platform that simulates the core capabilities of a professional equity screening system for the Pakistan Stock Exchange (PSX). Built as a personal FinTech portfolio project, it covers the complete data lifecycle — from raw OHLCV ingestion through indicator calculation, multi-factor scoring, sector benchmarking, and portfolio analysis — and delivers results through both a Streamlit web dashboard and a Jupyter Notebook interface.

The project uses entirely synthetic data and is not financial advice. Its purpose is to demonstrate applied data engineering, quantitative analytics, and product-oriented software development skills in a realistic FinTech context.

---

## Problem Statement

Retail investors and analysts in Pakistan typically lack access to the kind of systematic, rules-based screening tools available on Bloomberg or Refinitiv. Building one from scratch raises several real engineering challenges:

- How do you reliably ingest and validate time-series market data?
- How do you design a scoring model that is transparent, reproducible, and configurable?
- How do you benchmark individual stocks against their sector peers in a statistically sound way?
- How do you present complex quantitative output to non-technical users?
- How do you support both interactive exploration (dashboard) and programmatic analysis (notebooks)?

This project addresses all five challenges using an open-source Python stack with no paid data dependencies.

---

## Solution

A modular, twelve-phase pipeline that cleanly separates concerns across ingestion, computation, storage, and presentation layers:

- A validated OHLCV ingestion layer supports both synthetic data generation and real CSV import
- A SQLite relational database stores the universe and price history with full ORM support
- A configurable scoring engine computes technical, fundamental, and risk sub-scores and combines them into a single composite score per stock
- A sector benchmarking module computes relative and z-score comparisons across peer groups
- A Streamlit dashboard provides interactive filtering, charting, and score breakdown for non-technical users
- All modules are independently importable in Jupyter Notebooks for analyst-style ad-hoc exploration

---

## Main Capabilities

| Capability | Description |
|------------|-------------|
| OHLCV Ingestion | Synthetic random-walk generator or CSV loader with data validation |
| Technical Indicators | SMA 20/50/200, RSI 14, MACD, Volume Ratio, Volatility, Drawdown, Returns |
| Fundamental Metrics | P/E, P/B, ROE, Dividend Yield, EPS, Payout Ratio |
| Three-Pillar Scoring | Composite 0–100 score from Technical, Fundamental, and Risk pillars |
| Verdict System | Five-tier label (Strong Buy → Avoid) with emoji, colour, and rationale |
| Sector Benchmarking | Per-stock relative differences and z-scores vs sector peers |
| Preset Screeners | Eight named filter profiles covering major investment styles |
| Analyst Report | Structured per-ticker narrative summary of all scoring factors |
| Portfolio Builder | Score-weighted or equal-weight portfolio construction with diversification checks |
| Simple Backtest | Rule-based entry/exit simulation with Sharpe ratio and drawdown reporting |
| Streamlit Dashboard | Interactive web UI with charts, screener table, KPI bar, score breakdown |
| Jupyter Workflow | All modules independently importable for programmatic analysis |

---

## Technical Architecture

The system follows a layered architecture with a single central orchestrator:

**Ingestion Layer** (`src/ingestion/`)
Handles all data input — synthetic generation, CSV loading, and validation. Decoupled from the database layer so sources can be swapped without changing downstream code.

**Database Layer** (`src/database/`)
SQLAlchemy 2.0 ORM over SQLite. Two tables: `stock_universe` (40-stock master list) and `stock_price` (OHLCV rows per ticker). WAL mode and foreign key enforcement are configured at connection time.

**Analysis Layer** (`src/analysis/`)
Independent modules for technical indicators (`technical.py`), fundamental metrics (`fundamentals.py`), scoring (`scoring_model.py`), verdicts (`verdict.py`), and sector benchmarking (`sector_benchmark.py`). Each module is independently testable.

**Screener Layer** (`src/screener/`)
The `ScreenerEngine` in `engine.py` orchestrates the full pipeline. Supporting modules provide preset screeners, per-ticker reports, portfolio construction, and backtesting.

**Presentation Layer** (`app/`)
The Streamlit dashboard reads from the engine's output DataFrame. No business logic lives in the UI layer.

All weights, window sizes, and thresholds are centralised in `config/settings.py` and can be adjusted without touching module code.

---

## Data Architecture

```
[Source]          [Validation]      [Storage]        [Compute]         [Output]
CSV / Generator → DataValidator  → SQLite (ORM)  → ScreenerEngine  → DataFrame
                                                   ├─ Indicators
                                                   ├─ Fundamentals
                                                   ├─ Scoring
                                                   ├─ Verdicts
                                                   └─ Benchmarks
```

- Storage uses SQLite with SQLAlchemy, making a future migration to PostgreSQL or MySQL straightforward — only the connection string needs to change
- All computation runs in-memory on pandas DataFrames after a single DB read; no repeated round-trips
- The output DataFrame is the single source of truth for the dashboard, reports, and portfolio tools

---

## Scoring Methodology

The composite score (0–100) is the project's analytical centrepiece. It is built from four weighted pillars:

**Technical Pillar (35%)** measures price trend health via SMA alignment, momentum returns, RSI, MACD, and 52-week breakout proximity. Stocks in a clear bullish trend with strong relative returns score highest.

**Fundamental Pillar (30%)** evaluates valuation attractiveness and profitability via P/E, P/B, ROE, and dividend yield. Lower valuation multiples paired with higher ROE score best.

**Risk Pillar (20%)** penalises high annualised volatility, large maximum drawdowns, and elevated downside deviation. Stable, low-drawdown stocks score highest.

**Volume Pillar (15%)** rewards consistent liquidity and unusual volume surges, which often precede significant price moves.

Each sub-factor is normalised to a 0–100 range before weighting. All weights are configurable in `settings.py` so the model can be recalibrated without code changes.

---

## Sector Benchmarking Explanation

The sector benchmarking module (`sector_benchmark.py`) adds contextual depth beyond absolute scores.

For each of six metrics (P/E, P/B, ROE, Dividend Yield, Composite Score, Risk Score), the module computes per-sector statistics — mean, standard deviation, and median — then calculates two relative measures for every stock:

- **`{metric}_vs_sector`** (relative difference): `(stock_value − sector_avg) / |sector_avg|`. A value of +0.15 means the stock is 15% above its sector average.
- **`{metric}_sector_z`** (z-score): `(stock_value − sector_avg) / sector_std`. Shows how many standard deviations above or below the sector peer group the stock sits.

Finally, a `sector_value_label` is assigned using the composite score z-score and PE z-score, ranging from "Sector Value Leader" (composite_z ≥ 1.0) down to "Weak vs Sector". Sectors with fewer than two valid data points return "Insufficient sector data" rather than unreliable statistics.

---

## Preset Screener Explanation

Eight ready-to-use screener profiles are implemented in `presets.py`. Each applies a set of threshold filters to the screener DataFrame and returns a sorted, filtered result:

- **Strong Buy Candidates** — top-conviction entries combining verdict, score, and risk filters
- **Undervalued Quality** — valuation below sector average with above-average profitability
- **Dividend Picks** — positive yield, sustainable payout ratio, manageable risk
- **Momentum Leaders** — strong technical score with confirmed positive price momentum
- **Low Risk Blue Chips** — low risk score paired with above-average composite score
- **Oversold Bounce** — RSI < 35 with sufficient technical quality to suggest a recovery
- **Avoid / High Risk** — negative screens for portfolio risk management
- **Sector Relative Value** — stocks whose sector benchmarks place them in the top value tiers

All presets handle missing columns gracefully and return an empty DataFrame rather than crashing when data is unavailable.

---

## Stock Report Explanation

`stock_report.py` generates a structured analyst-style summary for any individual ticker in the universe. The report includes all three scoring pillars with sub-factor breakdowns, the verdict with its rationale, the sector benchmark context showing how the stock compares to its peers, and a snapshot of the latest OHLCV and fundamental data.

This is designed to answer the question: *"Why does this stock have this score?"* — providing transparency into the quantitative model that is often missing from black-box screeners.

---

## Portfolio Builder and Backtest Explanation

**Portfolio Builder** (`portfolio_builder.py`):
Takes the screener output and constructs a model portfolio by selecting the top N stocks above a minimum score threshold. Two allocation methods are supported — equal weight and score-proportional weight. A sector cap constraint prevents over-concentration in any single sector. The output includes per-holding weights, scores, verdicts, and a portfolio-level weighted average score.

**Backtest** (`backtest.py`):
Applies a simple rule-based simulation to the synthetic OHLCV history. A stock is "bought" when its composite score crosses above an entry threshold and "sold" when the score falls below that threshold or a stop-loss percentage is breached. The engine tracks daily portfolio value and reports total return, annualised return, Sharpe ratio, maximum drawdown, number of trades, and win rate.

The backtest is explicitly a methodology demonstration. It uses synthetic prices, ignores transaction costs and slippage, and should not be used to draw conclusions about real investment performance.

---

## Challenges Solved

**1. Graceful degradation with partial data**
Every module is designed to handle missing columns and NaN values without crashing. Sector benchmarks degrade to "Insufficient sector data" rather than producing misleading statistics. Preset screeners return empty DataFrames rather than exceptions. This makes the system robust enough for demonstration with a sparse synthetic dataset.

**2. Avoiding duplicate column suffixes on repeated calls**
The sector benchmarking module drops all pre-existing benchmark columns before re-merging, preventing pandas `_x`/`_y` suffix collisions when `apply_sector_benchmarks` is called multiple times on the same DataFrame.

**3. Separating configuration from logic**
All weights, window sizes, and thresholds are centralised in `config/settings.py`. This means the scoring model can be recalibrated — e.g. increasing the weight of the fundamental pillar — without touching any analysis module code.

**4. Supporting two usage modes**
The same analytical modules work in both Streamlit (event-driven, cached) and Jupyter (linear, interactive). The engine is designed to be instantiated and queried independently of any UI layer.

**5. Clean ORM-to-DataFrame handoff**
The database layer returns raw ORM objects; `db_manager.py` converts them to DataFrames before passing to the analysis layer. This keeps SQLAlchemy concerns out of the pandas-based computation modules.

---

## Business and FinTech Relevance

This project mirrors the architecture of real equity analytics platforms at a smaller scale:

- **Data ingestion pipelines** are fundamental to every FinTech data product, from robo-advisors to risk systems
- **Factor scoring models** are the basis of systematic (quantitative) investing — used at hedge funds, asset managers, and index providers
- **Sector-relative benchmarking** is standard practice in equity research and portfolio attribution
- **Interactive dashboards** for screener output are the core product of platforms like Bloomberg, FactSet, and retail tools like Finviz
- **Backtesting frameworks** underpin every quantitative strategy evaluation pipeline

Building all of these from scratch in a cohesive, modular system demonstrates genuine understanding of how these components fit together — not just the ability to call a single library.

---

## What This Demonstrates Technically

- **Python data engineering**: multi-module pipeline design, pandas data transformation, NumPy-based financial calculations
- **Database engineering**: SQLAlchemy ORM, SQLite configuration, schema design, session management
- **Software architecture**: separation of concerns, configurable weights, graceful degradation, clean public APIs
- **FinTech domain knowledge**: equity screening, factor investing, technical analysis, fundamental analysis, risk metrics, sector benchmarking, portfolio construction, backtesting
- **Product thinking**: two usage surfaces (dashboard + notebook), recruiter-ready documentation, honest limitation disclosure

---

## Future Roadmap

| Priority | Improvement |
|----------|-------------|
| High | Replace synthetic data with real PSX OHLCV (scraper or `yfinance`) |
| High | Real fundamental data from company filings or a data vendor |
| Medium | Automated nightly data pipeline (APScheduler / Celery) |
| Medium | Watchlist and price alert system |
| Medium | Extended backtest with transaction costs, benchmark comparison |
| Medium | Optional PostgreSQL or MySQL backend (connection string change only — the SQLAlchemy ORM layer is already database-agnostic) |
| Low | User authentication for multi-user Streamlit deployment |
| Low | Automated pytest suite for all analysis modules |
| Low | Docker containerisation for consistent deployment |
