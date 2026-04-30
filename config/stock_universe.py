# config/stock_universe.py
# ─────────────────────────────────────────────────────────────────────────────
# PSX40 Universe — 40 liquid, investable blue-chip tickers on the
# Pakistan Stock Exchange (PSX), grouped by sector.
#
# base_price : approximate mid-2024 price (PKR) used for sample-data seeding
#              and fallback simulations only — not a live quote.
# ─────────────────────────────────────────────────────────────────────────────

PSX40_UNIVERSE = [

    # ── Banking (6) ──────────────────────────────────────────────────────────
    # Largest, most liquid commercial and Islamic banks by market-cap
    {"ticker": "HBL",     "name": "Habib Bank Ltd",               "sector": "Banking",    "base_price": 155.0},
    {"ticker": "UBL",     "name": "United Bank Ltd",              "sector": "Banking",    "base_price": 195.0},
    {"ticker": "MCB",     "name": "MCB Bank Ltd",                 "sector": "Banking",    "base_price": 210.0},
    {"ticker": "MEBL",    "name": "Meezan Bank Ltd",              "sector": "Banking",    "base_price": 175.0},
    {"ticker": "ABL",     "name": "Allied Bank Ltd",              "sector": "Banking",    "base_price": 118.0},
    {"ticker": "BAFL",    "name": "Bank Alfalah Ltd",             "sector": "Banking",    "base_price":  48.0},

    # ── Oil & Gas (6) ─────────────────────────────────────────────────────────
    # Upstream E&P, downstream marketing, and gas transmission majors
    {"ticker": "PSO",     "name": "Pakistan State Oil",           "sector": "Oil & Gas",  "base_price": 285.0},
    {"ticker": "OGDC",    "name": "Oil & Gas Dev. Company",       "sector": "Oil & Gas",  "base_price": 175.0},
    {"ticker": "PPL",     "name": "Pakistan Petroleum Ltd",       "sector": "Oil & Gas",  "base_price": 115.0},
    {"ticker": "MARI",    "name": "Mari Petroleum Company Ltd",   "sector": "Oil & Gas",  "base_price": 1950.0},
    {"ticker": "POC",     "name": "Pakistan Oilfields Ltd",       "sector": "Oil & Gas",  "base_price":  470.0},
    {"ticker": "SNGP",    "name": "Sui Northern Gas Pipelines",   "sector": "Oil & Gas",  "base_price":  62.0},

    # ── Fertilizer (4) ───────────────────────────────────────────────────────
    {"ticker": "EFERT",   "name": "Engro Fertilizers Ltd",        "sector": "Fertilizer", "base_price":  92.0},
    {"ticker": "FFC",     "name": "Fauji Fertilizer Company",     "sector": "Fertilizer", "base_price": 110.0},
    {"ticker": "FFBL",    "name": "Fauji Fertilizer Bin Qasim",   "sector": "Fertilizer", "base_price":  28.0},
    {"ticker": "FATIMA",  "name": "Fatima Fertilizer Company",    "sector": "Fertilizer", "base_price":  35.0},

    # ── Cement (5) ───────────────────────────────────────────────────────────
    {"ticker": "LUCK",    "name": "Lucky Cement Ltd",             "sector": "Cement",     "base_price": 880.0},
    {"ticker": "CHCC",    "name": "Cherat Cement Company",        "sector": "Cement",     "base_price": 155.0},
    {"ticker": "DGKC",    "name": "DG Khan Cement Company",       "sector": "Cement",     "base_price":  95.0},
    {"ticker": "PIOC",    "name": "Pioneer Cement Ltd",           "sector": "Cement",     "base_price":  82.0},
    {"ticker": "MLCF",    "name": "Maple Leaf Cement Factory",    "sector": "Cement",     "base_price":  38.0},

    # ── Power / IPPs (4) ─────────────────────────────────────────────────────
    # Independent power producers; high dividend yield, defensive
    {"ticker": "HUBC",    "name": "Hub Power Company Ltd",        "sector": "Power",      "base_price":  90.0},
    {"ticker": "KAPCO",   "name": "Kot Addu Power Company",       "sector": "Power",      "base_price":  38.0},
    {"ticker": "NCPL",    "name": "Nishat Chunian Power Ltd",     "sector": "Power",      "base_price":  36.0},
    {"ticker": "KEL",     "name": "K-Electric Ltd",               "sector": "Power",      "base_price":   4.5},

    # ── Technology (3) ───────────────────────────────────────────────────────
    {"ticker": "SYS",     "name": "Systems Ltd",                  "sector": "Technology", "base_price": 420.0},
    {"ticker": "TRG",     "name": "TRG Pakistan Ltd",             "sector": "Technology", "base_price": 130.0},
    {"ticker": "NETSOL",  "name": "NetSol Technologies Ltd",      "sector": "Technology", "base_price": 185.0},

    # ── Automobile (3) ───────────────────────────────────────────────────────
    {"ticker": "INDU",    "name": "Indus Motor Company",          "sector": "Automobile", "base_price": 1650.0},
    {"ticker": "ATLH",    "name": "Atlas Honda Ltd",              "sector": "Automobile", "base_price":  530.0},
    {"ticker": "PSMC",    "name": "Pak Suzuki Motor Company",     "sector": "Automobile", "base_price":  285.0},

    # ── Consumer / FMCG (4) ──────────────────────────────────────────────────
    {"ticker": "NESTLE",  "name": "Nestle Pakistan Ltd",          "sector": "FMCG",       "base_price": 6800.0},
    {"ticker": "COLG",    "name": "Colgate-Palmolive Pakistan",   "sector": "FMCG",       "base_price": 2550.0},
    {"ticker": "UNILEVER","name": "Unilever Pakistan Ltd",        "sector": "FMCG",       "base_price": 2100.0},
    {"ticker": "UNITY",   "name": "Unity Foods Ltd",              "sector": "FMCG",       "base_price":   22.0},

    # ── Pharmaceuticals (3) ──────────────────────────────────────────────────
    {"ticker": "SEARL",   "name": "The Searle Company Ltd",       "sector": "Pharma",     "base_price": 210.0},
    {"ticker": "FEROZ",   "name": "Ferozsons Laboratories Ltd",   "sector": "Pharma",     "base_price": 490.0},
    {"ticker": "GLAXO",   "name": "GlaxoSmithKline Pakistan",     "sector": "Pharma",     "base_price": 175.0},

    # ── Chemicals / Conglomerates (2) ────────────────────────────────────────
    {"ticker": "ENGRO",   "name": "Engro Corporation Ltd",        "sector": "Chemicals",  "base_price": 275.0},
    {"ticker": "ICI",     "name": "ICI Pakistan Ltd",             "sector": "Chemicals",  "base_price": 660.0},
]

# ── Quick-access helpers ──────────────────────────────────────────────────────

# Flat list of all tickers — useful for data fetching loops
PSX40_TICKERS: list[str] = [s["ticker"] for s in PSX40_UNIVERSE]

# Sector → [tickers] mapping
import collections as _collections
PSX40_BY_SECTOR: dict[str, list[str]] = _collections.defaultdict(list)
for _s in PSX40_UNIVERSE:
    PSX40_BY_SECTOR[_s["sector"]].append(_s["ticker"])
PSX40_BY_SECTOR = dict(PSX40_BY_SECTOR)  # convert to plain dict

# Ticker → metadata lookup
PSX40_META: dict[str, dict] = {s["ticker"]: s for s in PSX40_UNIVERSE}
