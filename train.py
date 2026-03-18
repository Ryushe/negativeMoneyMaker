"""
Offline training on resolved Kalshi markets.

Usage:
    python train.py
    python train.py --markets 500
"""

import argparse

import numpy as np
import pandas as pd
import requests

import model as m
from data import _days_to_expiry
from features import FEATURE_COLS, classify

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"


def fetch_resolved(limit: int) -> pd.DataFrame:
    markets, cursor = [], ""
    while len(markets) < limit:
        params: dict = {"limit": min(200, limit - len(markets)), "status": "settled"}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{KALSHI_API}/markets", params=params, timeout=20)
        resp.raise_for_status()
        body   = resp.json()
        batch  = body.get("markets", [])
        markets.extend(batch)
        cursor = body.get("cursor", "")
        if not cursor or not batch:
            break
    return pd.DataFrame(markets)


def label(row) -> int | None:
    result = str(row.get("result") or "").lower()
    if result == "yes":
        return 1
    if result == "no":
        return 0
    lp = float(row.get("last_price") or row.get("yes_bid", -1))
    if lp >= 95:
        return 1
    if lp <= 5:
        return 0
    return None


def build_row(row) -> dict | None:
    lbl = label(row)
    if lbl is None:
        return None
    try:
        price      = float(row.get("last_price") or row.get("yes_bid", 50)) / 100.0
        volume_24h = float(row.get("volume_24h") or row.get("volume") or 0)
        liquidity  = float(row.get("open_interest") or 0)
        close_time = str(row.get("close_time") or "")
        dte        = _days_to_expiry(close_time) if close_time else 0.0
        title      = str(row.get("title") or row.get("subtitle") or "")
        api_cat    = str(row.get("category") or "")
        cat        = classify(title, api_cat)

        # History-based features default to 0 for historical data
        # (DB won't have old snapshots). Model learns to use them once live.
        return {
            "price":            float(np.clip(price, 0.01, 0.99)),
            "volume_24h":       volume_24h,
            "days_to_expiry":   dte,
            "liquidity":        liquidity,
            "spread_pct":       0.0,   # not available in resolved market dump
            "momentum_7d":      0.0,
            "price_volatility": 0.0,
            "volume_accel":     0.0,
            "market_age_days":  0.0,
            **cat,
            "label":            lbl,
        }
    except Exception:
        return None


def main(n_markets: int):
    print(f"[train] Fetching {n_markets} resolved Kalshi markets …")
    raw  = fetch_resolved(n_markets)
    print(f"[train] Got {len(raw)} raw markets")

    rows = [r for r in (build_row(row) for _, row in raw.iterrows()) if r]
    df   = pd.DataFrame(rows)
    yes  = int(df["label"].sum())
    no   = int((df["label"] == 0).sum())
    print(f"[train] Labelled: {len(df)}  (YES={yes}, NO={no})")

    if len(df) < 30:
        print("[train] Need at least 30 labelled samples.")
        return

    X = df[FEATURE_COLS].values
    y = df["label"].values

    trained = m.train(X, y)

    acc = (trained.predict(X) == y).mean()
    print(f"[train] In-sample accuracy: {acc:.1%}  (run backtest.py for out-of-sample)")

    imp = m.feature_importance(trained)
    if imp:
        print("\n  Feature importances:")
        for feat, val in sorted(imp.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(val * 40)
            print(f"    {feat:<20} {bar}  {val:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", type=int, default=500)
    args = parser.parse_args()
    main(args.markets)
