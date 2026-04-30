# 📈 PSX40 Stock Screener

A personal FinTech portfolio project — a full-stack stock screening and analytics platform for the Pakistan Stock Exchange (PSX), built with Python, SQLite, and Streamlit. Covers the complete data lifecycle: ingestion → indicators → scoring → reporting → portfolio analysis.

> ⚠️ **Demo project using synthetic (fake) data.**
> Does not connect to live PSX market feeds. Not financial advice.

---

## Table of Contents

- [Why This Project Matters](#why-this-project-matters)
- [Key Features](#key-features)
- [Completed Phases](#completed-phases)
- [Tech Stack](#tech-stack)
- [Architecture Overview](#architecture-overview)
- [Folder Structure](#folder-structure)
- [Data Pipeline](#data-pipeline)
- [Database Overview](#database-overview)
- [OHLCV Ingestion](#ohlcv-ingestion)
- [Fundamental Data Loading](#fundamental-data-loading)
- [Technical Indicators](#technical-indicators)
- [Fundamental Ratios](#fundamental-ratios)
- [Three-Pillar Scoring Methodology](#three-pillar-scoring-methodology)
- [Risk Model](#risk-model)
- [Verdict Model](#verdict-model)
- [Sector Benchmarking](#sector-benchmarking)
- [Preset Screeners](#preset-screeners)
- [Single-Stock Analyst Report](#single-stock-analyst-report)
- [Portfolio Builder](#portfolio-builder)
- [Simple Backtest](#simple-backtest)
- [Streamlit Dashboard Guide](#streamlit-dashboard-guide)
- [Jupyter Notebook Workflow](#jupyter-notebook-workflow)
- [Installation](#installation)
- [Initialising the Database](#initialising-the-database)
- [Importing CSV / Kaggle Data](#importing-csv--kaggle-data)
- [Running the Streamlit App](#running-the-streamlit-app)
- [Running Portfolio & Backtest in Jupyter](#running-portfolio--backtest-in-jupyter)
- [SQLite Setup](#sqlite-setup)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Disclaimer](#disclaimer)
- [Resume Bullet Points](#resume-bullet-points)

---

## Why This Project Matters

Most retail investors in Pakistan lack access to professional-grade screening tools. This project demonstrates how modern Python data engineering and analytics can close that gap — even with a synthetic dataset. It covers the same architectural concerns found in production FinTech systems:

- Structured data ingestion and validation
- Relational database management with an ORM
- Reproducible, weight-configurable scoring models
- Interactive dashboards for non-technical stakeholders
- Portfolio construction and basic quantitative backtesting

The project is intentionally scoped as a learning and portfolio demonstration. It is not connected to live data and does not constitute financial advice.

---

## Key Features

- **40-stock PSX universe** spanning 10 sectors
- **Auto-generated synthetic OHLCV data** via random walk for 10 active tickers
- **Technical indicators**: SMA 20/50/200, RSI 14, MACD, Volume Ratio, Volatility, Drawdown
- **Fundamental metrics**: P/E, P/B, ROE, Dividend Yield, EPS, Payout Ratio
- **Three-pillar composite score (0–100)**: Technical · Fundamental · Risk
- **Five-tier verdict system**: Strong Buy → Buy → Hold → Weak → Avoid
- **Sector benchmarking**: per-stock relative differences and z-scores vs sector peers
- **Eight named preset screeners**: Momentum Leaders, Strong Buy Candidates, Dividend Picks, and more
- **Single-stock analyst report**: full narrative breakdown per ticker
- **Portfolio builder**: weight-based portfolio construction with diversification checks
- **Simple backtest**: score-threshold entry rules, drawdown tracking, Sharpe ratio
- **Streamlit dashboard**: interactive candlestick charts, screener table, KPI bar, score breakdown
- **Jupyter Notebook workflow**: all modules importable for ad-hoc analysis

---

## Completed Phases

| Phase | Title | Status |
|-------|-------|--------|
| 1 | Project setup, folder structure, DB models | ✅ Complete |
| 2 | OHLCV ingestion (synthetic generator + CSV loader) | ✅ Complete |
| 3 | Technical indicators (SMA, RSI, MACD, volume, volatility) | ✅ Complete |
| 4 | Fundamental metrics (P/E, P/B, ROE, dividend yield, EPS) | ✅ Complete |
| 5 | Three-pillar scoring model (Technical · Fundamental · Risk) | ✅ Complete |
| 6 | Verdict system (composite score → label + emoji + rationale) | ✅ Complete |
| 7 | Sector benchmarking (relative differences, z-scores, value labels) | ✅ Complete |
| 8 | Preset screeners (eight named filter profiles) | ✅ Complete |
| 9 | Single-stock analyst report (per-ticker narrative output) | ✅ Complete |
| 10 | Portfolio builder + simple backtest | ✅ Complete |
| 11 | Optional MySQL support | ⏭️ Skipped |
| 12 | Final documentation polish | ✅ Complete |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Dashboard | Streamlit 1.35 |
| Database | SQLite 3 (via SQLAlchemy 2.0 ORM) |
| Data processing | pandas, numpy |
| Charts | Plotly 5.22 |
| Environment config | python-dotenv |
| Notebook support | Jupyter |

---

## Architecture Overview

```
CSV / Synthetic Generator
        │
        ▼
  Data Validator
        │
        ▼
   SQLite Database  ◄──────────────────────────────┐
        │                                           │
        ▼                                           │
  Screener Engine (engine.py)                       │
   ├─ Technical Indicators                          │
   ├─ Fundamental Metrics                           │
   ├─ Three-Pillar Scoring                          │
   ├─ Risk Model                                    │
   ├─ Verdict Assignment                            │
   ├─ Sector Benchmarking                           │
   └─ Preset Screeners                              │
        │                                           │
        ├──► Streamlit Dashboard ───────────────────┘
        ├──► Single-Stock Report
        ├──► Portfolio Builder
        └──► Backtest Engine
```

The `ScreenerEngine` is the central orchestrator. All modules are independently importable for notebook use.

---

## Folder Structure

```
psx40-screener/
│
├── app/
│   └── streamlit_app.py        # Streamlit dashboard UI
│
├── config/
│   ├── settings.py             # App-wide config and indicator parameters
│   └── stock_universe.py       # 40-stock PSX universe definition
│
├── data/
│   └── psx40.db                # SQLite database (auto-created on first run)
│
├── src/
│   ├── analysis/
│   │   ├── technical.py        # SMA, RSI, MACD, volume ratio, volatility
│   │   ├── scoring_model.py    # Three-pillar composite score (0–100)
│   │   ├── verdict.py          # Verdict label, emoji, colour, rationale
│   │   ├── fundamentals.py     # P/E, P/B, ROE, dividend yield, EPS
│   │   └── sector_benchmark.py # Sector-level stats, relative diffs, z-scores
│   │
│   ├── database/
│   │   ├── connection.py       # SQLAlchemy engine + session factory
│   │   ├── db_manager.py       # Read/write helpers
│   │   └── models.py           # ORM table definitions
│   │
│   ├── ingestion/
│   │   ├── sample_data_generator.py   # Synthetic OHLCV via random walk
│   │   ├── csv_loader.py              # Load OHLCV from CSV
│   │   └── data_validator.py          # Validates data before DB write
│   │
│   └── screener/
│       ├── engine.py           # Master orchestrator
│       ├── presets.py          # Eight named filter presets
│       ├── stock_report.py     # Single-stock analyst report
│       ├── portfolio_builder.py # Weight-based portfolio construction
│       └── backtest.py         # Simple score-based backtest
│
├── run_app.py                  # App launch script
├── requirements.txt
├── .env.example
└── README.md
```

---

## Data Pipeline

```
1. Stock Universe Seed
   └─ 40 tickers with sector tags loaded into StockUniverse table

2. OHLCV Ingestion
   └─ Synthetic random-walk data (or CSV) → validated → StockPrice table

3. Indicator Calculation
   └─ SMA 20/50/200, RSI 14, MACD, Volume Ratio, Volatility, Drawdown
      calculated per ticker from StockPrice rows

4. Fundamental Loading
   └─ P/E, P/B, ROE, Dividend Yield, EPS attached per ticker

5. Scoring
   └─ Technical score + Fundamental score + Risk score
      → weighted composite score (0–100)

6. Verdict Assignment
   └─ Composite score → label, emoji, colour, rationale

7. Sector Benchmarking
   └─ Per-metric sector averages, std devs, medians
      → relative differences and z-scores per stock

8. Export
   └─ Screener DataFrame available to dashboard, reports, portfolio builder
```

---

## Database Overview

The project uses **SQLite** as its default — and currently only — database backend. No external database server is required. The database file is created automatically at `data/psx40.db` on first run.

### Tables

| Table | Purpose |
|-------|---------|
| `stock_universe` | 40-stock master list: ticker, name, sector |
| `stock_price` | Daily OHLCV rows per ticker |

### ORM

SQLAlchemy 2.0 is used for all database interactions. The `connection.py` module configures:

- WAL journal mode for improved concurrency
- Foreign key enforcement
- A session factory (`SessionLocal`) used by all DB helpers

### Resetting the Database

```bash
rm data/psx40.db        # Mac / Linux
del data\psx40.db       # Windows
python run_app.py       # Recreates and reseeds automatically
```

---

## OHLCV Ingestion

Two ingestion paths are supported:

**Synthetic data (default)**
`sample_data_generator.py` generates 365 days of OHLCV via a geometric random walk seeded per ticker. Suitable for development and demonstration with no external data dependency.

**CSV ingestion**
`csv_loader.py` accepts a CSV file with columns: `ticker, date, open, high, low, close, volume`. The file is validated by `data_validator.py` before writing to the database.

### Kaggle / External CSV

1. Download a PSX OHLCV dataset from Kaggle or another source.
2. Rename/reformat columns to match: `ticker, date, open, high, low, close, volume`.
3. Run:

```python
from src.ingestion.csv_loader import load_csv_to_db
load_csv_to_db("path/to/your_data.csv")
```

---

## Fundamental Data Loading

`fundamentals.py` provides per-ticker fundamental metrics. In the current demo phase these are synthetic values seeded deterministically per ticker. The module exposes a clean API to be swapped for real scraped or API-sourced data in a future phase.

Metrics included: `pe_ratio`, `pb_ratio`, `roe`, `dividend_yield`, `eps`, `payout_ratio`.

---

## Technical Indicators

Calculated by `technical.py` from raw OHLCV rows:

| Indicator | Description |
|-----------|-------------|
| SMA 20 / 50 / 200 | Simple moving averages |
| RSI 14 | Relative Strength Index (14-period) |
| MACD | MACD line, signal line, histogram (12/26/9) |
| Volume Ratio | Latest volume vs 20-day average |
| Annualised Volatility | 30-day rolling std dev, annualised |
| Max Drawdown | Peak-to-trough drawdown over 252-day window |
| Downside Deviation | Downside volatility over 30-day window |
| Return 1M / 3M / 6M | Price returns over 21 / 63 / 126 trading days |
| 52-Week High | Highest close over last 252 trading days |

---

## Fundamental Ratios

Provided by `fundamentals.py`:

| Ratio | Description |
|-------|-------------|
| P/E Ratio | Price-to-Earnings |
| P/B Ratio | Price-to-Book |
| ROE | Return on Equity |
| Dividend Yield | Annual dividend as % of price |
| EPS | Earnings Per Share |
| Payout Ratio | Dividends paid as % of earnings |

---

## Three-Pillar Scoring Methodology

`scoring_model.py` produces a single composite score (0–100) for each stock by combining three independently computed pillars. All weights are configurable in `config/settings.py`.

### Pillar Weights

| Pillar | Default Weight |
|--------|---------------|
| Technical | 35% |
| Fundamental | 30% |
| Risk | 20% |
| Volume | 15% |

### Technical Sub-factors

- Price above SMA 20 (20%)
- SMA 20 above SMA 50 (20%)
- SMA 50 above SMA 200 (20%)
- 1-month return (15%)
- 3-month return (15%)
- 6-month return (10%)

### Momentum Sub-factors

- RSI score — penalises overbought (>70) and oversold (<30) conditions
- MACD score — signal line crossover signal
- Breakout score — proximity to 52-week high

### Risk Sub-factors

- Volatility score — lower annualised volatility scores higher
- Drawdown score — smaller max drawdown scores higher
- Downside deviation score — lower downside risk scores higher

### Volume Sub-factors

- Average volume score — rewards consistently liquid names
- Volume surge score — rewards unusual volume spikes

---

## Risk Model

The risk pillar produces a separate `risk_score` (0–100, lower = riskier). It combines:

- Annualised volatility (40% weight)
- Maximum drawdown over the past year (35% weight)
- Downside deviation (25% weight)

The risk score feeds both the composite score and the preset screeners (e.g. "Low Risk Blue Chips" filters `risk_score ≤ 40`).

---

## Verdict Model

`verdict.py` maps the composite score to a five-tier verdict:

| Score Range | Verdict | Emoji | Colour |
|-------------|---------|-------|--------|
| 80 – 100 | Strong Buy | 🟢 | #00C853 |
| 65 – 79 | Buy | 🔵 | #2979FF |
| 50 – 64 | Hold | 🟡 | #FFD600 |
| 35 – 49 | Weak | 🟠 | #FF6D00 |
| 0 – 34 | Avoid | 🔴 | #D50000 |

Each verdict also carries a one-line rationale text used in the analyst report.

---

## Sector Benchmarking

`sector_benchmark.py` provides two public functions:

**`build_sector_metrics(df)`** — returns one row per sector containing:
- Stock count and valid-data counts per metric
- Sector average, standard deviation, and median (PE only) for six metrics

**`apply_sector_benchmarks(df)`** — merges sector stats back onto every stock row and adds:
- `{metric}_vs_sector` — fractional relative difference from sector average
- `{metric}_sector_z` — z-score vs sector peers
- `sector_value_label` — one of: Sector Value Leader / Attractive vs Sector / In Line with Sector / Expensive vs Sector / Weak vs Sector / Insufficient sector data

Sectors with fewer than two valid data points for a given metric return `NaN` rather than unreliable statistics. Zero standard deviation returns z-score = 0.0 safely.

---

## Preset Screeners

`presets.py` provides eight named filters applied to the full screener DataFrame:

| Preset | Logic Summary |
|--------|--------------|
| Strong Buy Candidates | `final_verdict == "Strong Buy"` and `composite_score ≥ 70` |
| Undervalued Quality | P/E and P/B below sector average, ROE above sector average |
| Dividend Picks | Positive dividend yield, payout ratio ≤ 100%, manageable risk |
| Momentum Leaders | `technical_score ≥ 65`, positive 1M and 3M returns, price > SMA 50 |
| Low Risk Blue Chips | `risk_score ≤ 40` and `composite_score ≥ 55` |
| Oversold Bounce | `rsi_14 < 35` and `technical_score ≥ 40` |
| Avoid / High Risk | `verdict == "Avoid"` or `risk_score ≥ 75` or `composite_score < 40` |
| Sector Relative Value | `sector_value_label` in top two positive tiers, P/E below sector |

All presets degrade gracefully when optional columns are absent — no crashes on partial data.

---

## Single-Stock Analyst Report

`stock_report.py` generates a structured per-ticker report containing:

- Stock identity: ticker, name, sector
- Latest OHLCV snapshot
- Full technical indicator summary
- Fundamental ratio summary
- Three-pillar score breakdown with sub-factor detail
- Verdict label, emoji, and rationale
- Sector benchmark context (vs-sector and z-score highlights)

Reports can be printed to the terminal, rendered in a Jupyter cell, or exported as a text/markdown file.

```python
from src.screener.stock_report import generate_report
report = generate_report("ENGRO")
print(report)
```

---

## Portfolio Builder

`portfolio_builder.py` constructs a model portfolio from the screener output:

- Accepts a target number of holdings and an allocation method (equal weight or score-weighted)
- Applies minimum score thresholds and sector diversification constraints
- Returns a portfolio DataFrame with: ticker, weight, composite score, verdict, sector
- Exports a summary of total expected yield and weighted average score

```python
from src.screener.portfolio_builder import build_portfolio
portfolio = build_portfolio(n=10, method="score_weighted", min_score=55)
print(portfolio)
```

---

## Simple Backtest

`backtest.py` runs a rule-based historical simulation on the synthetic OHLCV data:

- **Entry rule**: buy when composite score crosses above a configurable threshold
- **Exit rule**: sell when score drops below threshold or stop-loss is triggered
- **Metrics reported**: total return, annualised return, max drawdown, Sharpe ratio, number of trades, win rate

> The backtest uses synthetic price data and a simplified scoring model. Results have no predictive value and are provided for methodology demonstration only.

```python
from src.screener.backtest import run_backtest
results = run_backtest(ticker="ENGRO", entry_threshold=60, stop_loss_pct=0.08)
print(results)
```

---

## Streamlit Dashboard Guide

### Starting the app

```bash
python run_app.py
```

Then open: `http://localhost:8501`

### Sidebar controls

| Control | Function |
|---------|----------|
| Quick Preset | Apply a named screener filter |
| Apply Preset button | Confirm and execute the selected preset |
| Sector | Filter table to one or more sectors |
| Demo Score Range | Min/max score slider |
| Verdict | Filter by one or more verdict labels |
| Price Chart Ticker | Select the stock for the candlestick chart |

### Main area sections

| Section | Content |
|---------|---------|
| KPI bar | Total stocks, sectors covered, average score, top-ranked ticker |
| Screener Table | All filtered stocks with indicators, scores, and verdicts |
| Price Chart | Interactive candlestick + SMA 20/50 overlay + RSI 14 panel |
| Score Breakdown | Expandable three-pillar sub-score detail for selected ticker |

### Optional launch flags

```bash
python run_app.py --port 8080        # Use a different port
python run_app.py --no-browser       # Do not auto-open browser
python run_app.py --debug            # Enable debug logging
```

---

## Jupyter Notebook Workflow

All modules are independently importable. A typical notebook session:

```python
# 1. Run the full engine pipeline
from src.screener.engine import ScreenerEngine
engine = ScreenerEngine()
engine.run()
df = engine.get_screener_df()

# 2. Apply a preset
from src.screener.presets import apply_preset
momentum = apply_preset(df, "Momentum Leaders")

# 3. Generate a stock report
from src.screener.stock_report import generate_report
print(generate_report("ENGRO"))

# 4. Build a portfolio
from src.screener.portfolio_builder import build_portfolio
portfolio = build_portfolio(n=10, method="score_weighted")

# 5. Run a backtest
from src.screener.backtest import run_backtest
results = run_backtest(ticker="ENGRO", entry_threshold=60)

# 6. Sector benchmarking
from src.analysis.sector_benchmark import build_sector_benchmarks
sector_summary, stock_sector = build_sector_benchmarks()
```

---

## Installation

### Requirements

| Tool | Minimum Version |
|------|----------------|
| Python | 3.11+ |
| pip | 23+ |

### Step-by-step

**1. Clone or unzip the project**

```bash
git clone https://github.com/your-username/psx40-screener.git
cd psx40-screener
```

**2. Create a virtual environment**

```bash
# Mac / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Create `.env` file**

```bash
cp .env.example .env        # Mac / Linux
copy .env.example .env      # Windows
```

**5. Create package `__init__.py` files**

```bash
# Mac / Linux
mkdir -p src/database src/ingestion src/analysis src/screener app config
touch src/__init__.py src/database/__init__.py src/ingestion/__init__.py
touch src/analysis/__init__.py src/screener/__init__.py
touch app/__init__.py config/__init__.py
```

---

## Initialising the Database

The database is created automatically on first launch:

```bash
python run_app.py
```

On first run, the app will:
1. Create `data/psx40.db`
2. Seed the 40-stock universe
3. Generate synthetic OHLCV data for 10 tickers
4. Calculate all indicators and scores
5. Load the dashboard (approximately 5–10 seconds)

Subsequent launches are instant.

---

## Importing CSV / Kaggle Data

1. Prepare a CSV with columns: `ticker, date, open, high, low, close, volume`
2. Run:

```python
from src.ingestion.csv_loader import load_csv_to_db
load_csv_to_db("data/your_file.csv")
```

The validator will check for required columns, non-negative prices, and valid date formats before writing.

---

## Running the Streamlit App

```bash
python run_app.py
```

Open your browser to `http://localhost:8501`.

---

## Running Portfolio & Backtest in Jupyter

```bash
jupyter notebook
```

Open `notebooks/analysis.ipynb` (or create a new notebook) and follow the [Jupyter Notebook Workflow](#jupyter-notebook-workflow) section above.

---

## SQLite Setup

No installation or configuration of an external database server is needed. SQLite is bundled with Python 3.

The database file lives at `data/psx40.db`. All table creation, seeding, and querying is handled automatically by the application.

**Key SQLite configuration** (applied via SQLAlchemy event listener):

```
PRAGMA journal_mode = WAL      -- Better read/write concurrency
PRAGMA foreign_keys = ON       -- Enforce referential integrity
PRAGMA synchronous = NORMAL    -- Balanced durability and speed
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
Run `python run_app.py` from the project root directory, not from a subfolder.

**`ModuleNotFoundError: No module named 'streamlit'`**
Activate your virtual environment: `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows).

**`sqlite3.OperationalError: no such table`**
Ensure `src/database/models.py` exists and defines `StockUniverse` and `StockPrice`. Re-run `python run_app.py` to recreate tables.

**Dashboard loads but screener table is empty**
Click **Apply Preset → All Stocks** in the sidebar to clear all active filters.

**Port 8501 already in use**
Use a different port: `python run_app.py --port 8502`

**Preset returns an empty DataFrame**
The preset's required columns (e.g. `rsi_14`, `dividend_yield`) may not have been calculated if data is missing. Verify the engine ran successfully by checking the terminal output.

**Resetting the database**
```bash
rm data/psx40.db    # Mac / Linux
del data\psx40.db   # Windows
python run_app.py   # Reseeds automatically
```

---

## Limitations

- **Synthetic data only.** All OHLCV and fundamental figures are randomly generated. No real PSX market data is fetched.
- **No live data feed.** There is no scheduler, scraper, or API integration for real-time updates.
- **Simplified fundamentals.** P/E, P/B, ROE, and dividend yield are static placeholder values — not computed from actual financial statements.
- **Basic backtest.** The backtest uses simplified entry/exit rules with no transaction costs, slippage, or realistic order execution.
- **Single-user SQLite.** SQLite is appropriate for local development. It is not suitable for concurrent multi-user production environments without modification.
- **No user authentication.** There is no login system or per-user data isolation.
- **No automated testing suite.** Unit and integration tests are not yet included.

---

## Future Improvements

- **Phase 12+: Real data ingestion** — PSX scraper or third-party API (e.g. Yahoo Finance via `yfinance`) for live OHLCV
- **Automated scheduling** — APScheduler or Celery for nightly data updates
- **Real fundamental data** — Company financial statements via scraped filings or a data vendor API
- **Watchlist and alerts** — Per-user stock watchlists with email/SMS price alerts
- **Extended backtesting** — Transaction costs, position sizing, benchmark comparison, rolling-window analysis
- **Optional PostgreSQL / MySQL support** — For multi-user or cloud-hosted deployments, the SQLAlchemy-based data layer can be migrated to PostgreSQL or MySQL by changing the connection string in `settings.py`. This migration has not been implemented in the current version.
- **User authentication** — Session-based login for the Streamlit app or migration to a framework supporting auth (FastAPI + React, Django)
- **Automated test suite** — pytest-based unit tests for scoring, benchmarking, and ingestion modules
- **Docker deployment** — Containerise the app for consistent environment management

---

## Disclaimer

This project is a personal portfolio and learning exercise. It uses entirely synthetic (randomly generated) data. It does not connect to any real market data source. Nothing in this project constitutes financial advice, investment recommendations, or a solicitation to buy or sell any security. All scores, verdicts, and reports are for technical demonstration purposes only.

---

## Resume Bullet Points

```
• Built a full-stack stock screening platform in Python (Streamlit, SQLAlchemy, pandas,
  Plotly) covering data ingestion, indicator calculation, scoring, sector benchmarking,
  and portfolio analysis across a 40-stock synthetic PSX universe.

• Designed a configurable three-pillar scoring model (technical, fundamental, risk)
  with weighted sub-factors, a five-tier verdict system, and eight preset screeners —
  mirroring quantitative factor-investing frameworks used in FinTech analytics.

• Engineered a SQLite-backed data pipeline using SQLAlchemy 2.0 ORM with WAL mode,
  foreign key enforcement, and a validated OHLCV ingestion layer supporting both
  synthetic data generation and CSV import.

• Built an interactive Streamlit dashboard with candlestick charts, RSI panels,
  live score filtering, and a sector benchmarking breakdown — delivering analytics
  insights to non-technical stakeholders without requiring code access.

• Implemented a score-based quantitative backtest with Sharpe ratio, max drawdown,
  win rate reporting, and a weight-based portfolio builder with sector diversification
  constraints and score-weighted allocation.
```
