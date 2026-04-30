import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from config.settings import DB_URL, DB_PATH

# Ensure data directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


class Base(DeclarativeBase):
    pass


# SQLite engine with performance pragmas
engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)


@event.listens_for(engine, "connect")
def set_sqlite_pragmas(dbapi_conn, connection_record):
    """Apply SQLite performance and integrity settings on every connection."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")   # Better concurrency
    cursor.execute("PRAGMA foreign_keys=ON;")    # Enforce FK constraints
    cursor.execute("PRAGMA synchronous=NORMAL;") # Balanced speed/safety
    cursor.close()


# Session factory
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


def init_db():
    """Create all tables defined on Base if they do not already exist."""
    from src.database.models import StockPrice, StockUniverse  # noqa: F401
    Base.metadata.create_all(bind=engine)
