"""
SQLite price history store.

Every scan the bot calls save_snapshot() for each market.
After a week of running, momentum_7d and price_volatility become real signals
instead of approximations.

Schema:
    price_snapshots(ticker, ts, price, volume_24h, spread_pct)

Connection hygiene: every function opens a connection and explicitly closes
it in a finally block. sqlite3's context manager only handles transactions,
not connection lifetime — leaving it to garbage collection caused [Errno 24]
Too many open files when scanning 150 markets × 3 DB calls per market.
"""

import os
import sqlite3
from datetime import datetime, timezone, timedelta

import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "price_history.db")


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, timeout=10)


def init_db():
    conn = _connect()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS price_snapshots (
                ticker     TEXT    NOT NULL,
                ts         INTEGER NOT NULL,
                price      REAL    NOT NULL,
                volume_24h REAL,
                spread_pct REAL,
                PRIMARY KEY (ticker, ts)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ticker_ts ON price_snapshots(ticker, ts)"
        )
        conn.commit()
    finally:
        conn.close()


def save_snapshot(ticker: str, price: float, volume_24h: float, spread_pct: float):
    ts = int(datetime.now(timezone.utc).timestamp())
    conn = _connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO price_snapshots VALUES (?,?,?,?,?)",
            (ticker, ts, price, volume_24h, spread_pct),
        )
        conn.commit()
    finally:
        conn.close()


def get_history(ticker: str, days: int = 7) -> pd.DataFrame:
    """All snapshots for this ticker in the last N days."""
    cutoff = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT ts, price, volume_24h FROM price_snapshots "
            "WHERE ticker=? AND ts>=? ORDER BY ts",
            (ticker, cutoff),
        ).fetchall()
        return pd.DataFrame(rows, columns=["ts", "price", "volume_24h"])
    finally:
        conn.close()


def get_price_n_days_ago(ticker: str, days: int = 7) -> float | None:
    """Closest stored price to exactly N days ago."""
    target = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT price FROM price_snapshots "
            "WHERE ticker=? ORDER BY ABS(ts - ?) LIMIT 1",
            (ticker, target),
        ).fetchone()
        return float(row[0]) if row else None
    finally:
        conn.close()


def get_volume_n_days_ago(ticker: str, days: int = 1) -> float | None:
    """Closest stored volume_24h to N days ago."""
    target = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT volume_24h FROM price_snapshots "
            "WHERE ticker=? AND volume_24h IS NOT NULL ORDER BY ABS(ts - ?) LIMIT 1",
            (ticker, target),
        ).fetchone()
        return float(row[0]) if row else None
    finally:
        conn.close()


def cleanup_old(days: int = 30):
    """Drop snapshots older than N days to keep the DB from growing unbounded."""
    cutoff = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    conn = _connect()
    try:
        conn.execute("DELETE FROM price_snapshots WHERE ts < ?", (cutoff,))
        conn.commit()
    finally:
        conn.close()


# Initialise on import
init_db()
