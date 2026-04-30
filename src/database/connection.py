"""
connection.py
-------------
SQLAlchemy engine + session factory for the PSX40 screener.

Streamlit Cloud fix
-------------------
The repo mount (/mount/src/) is read-only on Streamlit Cloud.
We always write the SQLite database to /tmp/, which is writable
on every platform (local, Docker, Streamlit Cloud).
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# ── Resolve a writable DB path ────────────────────────────────────

def _resolve_db_path() -> str:
    """
    Return a writable absolute path for the SQLite file.

    Priority:
      1. $DATABASE_URL env-var  (explicit override, e.g. for Postgres)
      2. /tmp/psx40_screener.db (always writable — Streamlit Cloud safe)
    """
    explicit = os.environ.get("DATABASE_URL", "").strip()
    if explicit:
        return explicit          # caller supplied a full SQLAlchemy URL

    # Default: /tmp is writable on every platform
    return "sqlite:////tmp/psx40_screener.db"


_DB_URL = _resolve_db_path()

# Ensure local data directory exists when running locally with a file path
if _DB_URL.startswith("sqlite:///"):
    _db_file = _DB_URL.replace("sqlite:///", "")
    _db_dir  = os.path.dirname(_db_file)
    if _db_dir:
        os.makedirs(_db_dir, exist_ok=True)


# ── Declarative base ──────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Engine ────────────────────────────────────────────────────────

engine = create_engine(
    _DB_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)


@event.listens_for(engine, "connect")
def set_sqlite_pragmas(dbapi_conn, connection_record):
    """Apply SQLite performance and integrity settings on every connection."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")    # Better concurrency
    cursor.execute("PRAGMA foreign_keys=ON;")     # Enforce FK constraints
    cursor.execute("PRAGMA synchronous=NORMAL;")  # Balanced speed/safety
    cursor.close()


# ── Session factory ───────────────────────────────────────────────

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


def get_session():
    """Yield a SQLAlchemy session and ensure it is closed after use."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Create all tables defined on Base if they do not already exist."""
    from src.database.models import StockPrice, StockUniverse  # noqa: F401

    Base.metadata.create_all(bind=engine)
    print(f"[connection] DB initialised → {_DB_URL}")
