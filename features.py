"""
Feature engineering — 15 features vs the original 5.

New vs old:
  OLD: price, volume_24h, momentum_7d*, days_to_expiry, liquidity
       (* was broken — used current bid as "prior price")

  NEW: all of the above, plus:
    spread_pct      — bid-ask spread as % of midpoint (market efficiency signal)
    momentum_7d     — real 7-day price delta from SQLite history
    price_volatility— std dev of daily prices over 7d from history
    volume_accel    — is volume growing or shrinking vs yesterday?
    market_age_days — older markets tend to be more efficiently priced
    cat_*           — one-hot market category (politics/sports/economics/…)

Category features matter because different market types have different base
resolution rates and different levels of crowd wisdom. Politics markets
tend to be heavily bet and efficient; niche entertainment markets less so.

Note on history-dependent features (momentum_7d, price_volatility, volume_accel):
  These are 0.0 until the DB has enough history. The model will learn to use
  them once you've been running for a week. Re-run train.py then to pick them up.
"""

import numpy as np
import db

# ── Feature column order — must match exactly what the model was trained on ──
FEATURE_COLS = [
    "price",
    "volume_24h",
    "days_to_expiry",
    "liquidity",
    "spread_pct",
    "momentum_7d",
    "price_volatility",
    "volume_accel",
    "market_age_days",
    "cat_politics",
    "cat_sports",
    "cat_economics",
    "cat_financials",
    "cat_entertainment",
    "cat_other",
]

# Keywords used to classify market titles into categories
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "politics": [
        "election", "president", "congress", "senate", "house", "vote",
        "democrat", "republican", "governor", "primary", "ballot", "legislation",
        "supreme court", "impeach", "cabinet", "mayor", "poll",
    ],
    "sports": [
        "nfl", "nba", "mlb", "nhl", "ncaa", "super bowl", "world series",
        "championship", "playoff", "soccer", "football", "basketball",
        "baseball", "hockey", "tennis", "golf", "ufc", "fight", "match",
        "tournament", "olympics", "world cup",
    ],
    "economics": [
        "gdp", "inflation", "fed", "federal reserve", "interest rate",
        "unemployment", "cpi", "ppi", "recession", "jobs report",
        "consumer price", "rate hike", "rate cut", "fomc",
    ],
    "financials": [
        "s&p", "nasdaq", "dow", "stock", "bitcoin", "crypto", "ethereum",
        "etf", "ipo", "earnings", "market cap", "shares",
    ],
    "entertainment": [
        "oscar", "grammy", "emmy", "tony", "award", "movie", "film",
        "music", "celebrity", "box office", "album", "chart",
    ],
}


def classify(title: str, api_category: str = "") -> dict[str, float]:
    """Return a one-hot category dict. Defaults to cat_other if no match."""
    text = (title + " " + api_category).lower()
    result = {f"cat_{k}": 0.0 for k in _CATEGORY_KEYWORDS}
    result["cat_other"] = 1.0
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            result[f"cat_{cat}"] = 1.0
            result["cat_other"] = 0.0
            break
    return result


def build_feature_vector(
    ticker: str,
    price: float,
    volume_24h: float,
    spread_pct: float,
    days_to_expiry: float,
    liquidity: float,
    market_age_days: float,
    title: str,
    api_category: str = "",
) -> dict[str, float]:
    """
    Compute the full 15-feature vector for one market.
    Saves a price snapshot to the DB every time it's called (live mode).
    """
    # ── DB-dependent features ────────────────────────────────────────────────
    hist = db.get_history(ticker, days=7)

    if len(hist) >= 3:
        price_7d = db.get_price_n_days_ago(ticker, days=7) or price
        momentum_7d    = (price - price_7d) / (price_7d + 1e-9)
        price_volatility = float(hist["price"].std())
    else:
        # Not enough history yet — defaults to 0 (model trained with same defaults)
        momentum_7d      = 0.0
        price_volatility = 0.0

    vol_yesterday = db.get_volume_n_days_ago(ticker, days=1)
    if vol_yesterday and vol_yesterday > 0:
        volume_accel = (volume_24h - vol_yesterday) / (vol_yesterday + 1e-9)
    else:
        volume_accel = 0.0

    # ── Category ─────────────────────────────────────────────────────────────
    cat = classify(title, api_category)

    # ── Save snapshot so next scan has history ────────────────────────────────
    db.save_snapshot(ticker, price, volume_24h, spread_pct)

    return {
        "price":            float(np.clip(price, 0.01, 0.99)),
        "volume_24h":       float(volume_24h),
        "days_to_expiry":   float(days_to_expiry),
        "liquidity":        float(liquidity),
        "spread_pct":       float(np.clip(spread_pct, 0.0, 1.0)),
        "momentum_7d":      float(np.clip(momentum_7d, -1.0, 1.0)),
        "price_volatility": float(price_volatility),
        "volume_accel":     float(np.clip(volume_accel, -5.0, 5.0)),
        "market_age_days":  float(market_age_days),
        **cat,
    }


def as_array(feature_dict: dict) -> np.ndarray:
    """Convert a feature dict to a numpy array in the correct column order."""
    return np.array([feature_dict[col] for col in FEATURE_COLS], dtype=float)
