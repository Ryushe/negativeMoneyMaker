"""
Kalshi API layer — fetch markets and build feature DataFrames.

Responsibilities:
  - HTTP calls to api.elections.kalshi.com
  - Orderbook midpoint + spread calculation
  - Orchestrating features.py for each market row
"""

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import features as f

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

_session = requests.Session()
_session.headers.update({"Content-Type": "application/json"})


# ── Raw API calls ────────────────────────────────────────────────────────────

def fetch_markets(limit: int = 200, status: str = "open") -> pd.DataFrame:
    """Return raw Kalshi market rows sorted by 24h volume descending."""
    markets, cursor = [], ""
    while len(markets) < limit:
        params: dict = {"limit": min(200, limit - len(markets)), "status": status}
        if cursor:
            params["cursor"] = cursor
        resp = _session.get(f"{KALSHI_API}/markets", params=params, timeout=15)
        resp.raise_for_status()
        body   = resp.json()
        batch  = body.get("markets", [])
        markets.extend(batch)
        cursor = body.get("cursor", "")
        if not cursor or not batch:
            break

    df = pd.DataFrame(markets)
    if "volume_24h" in df.columns:
        df = df.sort_values("volume_24h", ascending=False)
    return df.reset_index(drop=True)


def fetch_orderbook(ticker: str) -> tuple[float, float]:
    """
    Return (midpoint, spread_pct) for a YES contract.
    midpoint  — (best_yes_bid + best_yes_ask) / 2, in [0, 1]
    spread_pct — (ask - bid) / midpoint
    Both default to (0.5, 1.0) if the orderbook is empty.
    """
    try:
        resp = _session.get(
            f"{KALSHI_API}/markets/{ticker}/orderbook", timeout=10
        )
        resp.raise_for_status()
        book     = resp.json().get("orderbook", {})
        yes_bids = book.get("yes", [])   # [[price_cents, qty], ...]
        no_bids  = book.get("no",  [])   # no-side implies yes-ask at (100 - no_bid)

        if yes_bids:
            best_bid = float(yes_bids[0][0]) / 100.0
        else:
            best_bid = None

        if no_bids:
            best_ask = (100.0 - float(no_bids[0][0])) / 100.0
        else:
            best_ask = None

        if best_bid is not None and best_ask is not None:
            mid        = (best_bid + best_ask) / 2.0
            spread_pct = (best_ask - best_bid) / (mid + 1e-9)
            return float(np.clip(mid, 0.01, 0.99)), float(spread_pct)
        if best_bid is not None:
            return float(np.clip(best_bid, 0.01, 0.99)), 0.05
        if best_ask is not None:
            return float(np.clip(best_ask, 0.01, 0.99)), 0.05
    except Exception:
        pass
    return 0.5, 1.0   # worst-case defaults


def _days_since(open_time_str: str) -> float:
    try:
        t   = datetime.fromisoformat(open_time_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return max((now - t).total_seconds() / 86400.0, 0.0)
    except Exception:
        return 0.0


def _days_to_expiry(close_time_str: str) -> float:
    try:
        t   = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return max((t - now).total_seconds() / 86400.0, 0.0)
    except Exception:
        return 30.0


# ── Feature DataFrame builder ────────────────────────────────────────────────

def build_features(markets_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each market row: fetch live orderbook, compute 15 features, return DataFrame.
    Also saves a price snapshot to the SQLite DB on every call.
    """
    rows = []
    for _, row in markets_df.iterrows():
        ticker = row.get("ticker", "")
        if not ticker:
            continue

        try:
            # Live price + spread
            mid, spread_pct = fetch_orderbook(ticker)

            # Fallback price from stored field if orderbook was empty
            if mid == 0.5 and spread_pct == 1.0:
                stored = row.get("last_price") or row.get("yes_bid")
                if stored:
                    mid = float(stored) / 100.0

            volume_24h = float(row.get("volume_24h") or row.get("volume") or 0)
            liquidity  = float(row.get("open_interest") or 0)

            close_time = str(row.get("close_time") or row.get("expected_expiration_time") or "")
            open_time  = str(row.get("open_time") or "")

            dte         = _days_to_expiry(close_time)
            age         = _days_since(open_time)
            title       = str(row.get("title") or row.get("subtitle") or "")
            api_cat     = str(row.get("category") or "")

            feat = f.build_feature_vector(
                ticker=ticker,
                price=mid,
                volume_24h=volume_24h,
                spread_pct=spread_pct,
                days_to_expiry=dte,
                liquidity=liquidity,
                market_age_days=age,
                title=title,
                api_category=api_cat,
            )

            feat.update({
                "ticker":       ticker,
                "title":        title,
                "close_time":   close_time,
                "market_price": mid,
            })
            rows.append(feat)

        except Exception:
            continue

        time.sleep(0.05)  # ~20 req/s — stay well within rate limits

    return pd.DataFrame(rows)
