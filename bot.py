"""
Live trading bot for Kalshi.

Key improvements over v1:
  - Fee-adjusted expected value replaces the bare 50% discount rule
  - Kelly criterion position sizing (quarter-Kelly by default)
  - 15-feature model (real momentum, volatility, spread, category)
  - DB cleanup runs every 24h to keep price_history.db lean
"""

import base64
import datetime
import os
import time

import numpy as np
import requests
from dotenv import load_dotenv

import data as d
import db
import model as m
from features import FEATURE_COLS, as_array
from metrics import PerformanceLog, PositionTracker
from risk import RiskGuard

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
MIN_CONFIDENCE   = float(os.getenv("MIN_CONFIDENCE",    "0.70"))
ENTRY_DISCOUNT   = float(os.getenv("ENTRY_DISCOUNT",    "0.50"))
EXIT_TARGET      = float(os.getenv("EXIT_TARGET",       "0.90"))
EXPIRY_EXIT_DAYS = float(os.getenv("EXPIRY_EXIT_DAYS",  "7"))
MIN_EDGE         = float(os.getenv("MIN_EDGE",          "0.03"))
FEE_RATE         = float(os.getenv("KALSHI_FEE_RATE",   "0.07"))
KELLY_FRACTION   = float(os.getenv("KELLY_FRACTION",    "0.25"))
MAX_BANKROLL     = float(os.getenv("MAX_BANKROLL_USD",  "1000"))
MAX_POSITION_USD = float(os.getenv("MAX_POSITION_USD",  "50"))
RISK_FREE_RATE   = float(os.getenv("RISK_FREE_RATE",    "0.05"))
POLL_INTERVAL    = int(os.getenv("POLL_INTERVAL_SEC",   "120"))

KALSHI_KEY_ID     = os.getenv("KALSHI_API_KEY_ID",     "")
KALSHI_KEY_SECRET = os.getenv("KALSHI_API_KEY_SECRET", "")
PAPER_MODE        = not (KALSHI_KEY_ID and KALSHI_KEY_SECRET)

KALSHI_API = "https://trading.kalshi.com/trade-api/v2"


# ── Edge calculations ─────────────────────────────────────────────────────────

def expected_value(market_price: float, model_prob: float) -> float:
    """
    Fee-adjusted expected value per dollar invested.

    If YES: collect (1 - market_price), pay fee on winnings.
    If NO : lose market_price.

    EV = p * (1 - market_price) * (1 - fee) - (1 - p) * market_price
    """
    net_win  = (1.0 - market_price) * (1.0 - FEE_RATE)
    net_loss = market_price
    return model_prob * net_win - (1.0 - model_prob) * net_loss


def kelly_size(market_price: float, model_prob: float) -> float:
    """
    Quarter-Kelly position size in USD.

    b = net win per dollar staked (after fees)
    f* = (b*p - q) / b   where q = 1 - p
    size = f* * KELLY_FRACTION * MAX_BANKROLL, capped at MAX_POSITION_USD
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0
    b = (1.0 - market_price) * (1.0 - FEE_RATE) / market_price
    if b <= 0:
        return 0.0
    q      = 1.0 - model_prob
    f_star = (b * model_prob - q) / b
    if f_star <= 0:
        return 0.0
    return float(min(f_star * KELLY_FRACTION * MAX_BANKROLL, MAX_POSITION_USD))


# ── Signal gates ──────────────────────────────────────────────────────────────

def should_buy(market_price: float, model_prob: float) -> tuple[bool, float]:
    """
    Returns (enter: bool, size_usd: float).

    Three gates must all pass:
      1. Model confidence >= MIN_CONFIDENCE
      2. Market price <= model_prob * ENTRY_DISCOUNT  (2× undervalued)
      3. Fee-adjusted EV >= MIN_EDGE
    """
    if model_prob < MIN_CONFIDENCE:
        return False, 0.0
    if market_price > model_prob * ENTRY_DISCOUNT:
        return False, 0.0
    ev = expected_value(market_price, model_prob)
    if ev < MIN_EDGE:
        return False, 0.0
    size = kelly_size(market_price, model_prob)
    if size < 1.0:
        return False, 0.0
    return True, size


def should_sell(market_price: float, model_prob: float, days_left: float) -> tuple[bool, str]:
    if market_price >= model_prob * EXIT_TARGET:
        return True, "target reached"
    if days_left <= EXPIRY_EXIT_DAYS:
        return True, "near expiry"
    return False, ""


# ── Order execution ───────────────────────────────────────────────────────────

def _signed_headers(method: str, path: str) -> dict:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    ts  = str(int(time.time() * 1000))
    msg = (ts + method.upper() + path).encode()
    key = serialization.load_pem_private_key(KALSHI_KEY_SECRET.encode(), password=None)
    sig = key.sign(msg, padding.PKCS1v15(), hashes.SHA256())
    return {
        "KALSHI-ACCESS-KEY":       KALSHI_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "Content-Type":            "application/json",
    }


def place_order(ticker: str, side: str, size_usd: float, price: float) -> bool:
    if PAPER_MODE:
        ev = expected_value(price, price)   # illustrative — use actual model_prob in caller
        print(f"  [PAPER] {side.upper()} {ticker}  @ {price:.3f}  size=${size_usd:.2f}")
        return True
    try:
        price_cents = max(1, min(99, round(price * 100)))
        count       = max(1, round(size_usd / price))
        path        = "/portfolio/orders"
        payload     = {
            "ticker":        ticker,
            "action":        "buy",
            "side":          side,
            "type":          "limit",
            "count":         count,
            "yes_price":     price_cents if side == "yes" else (100 - price_cents),
            "expiration_ts": int(time.time()) + 300,
        }
        resp = requests.post(
            f"{KALSHI_API}{path}",
            json=payload,
            headers=_signed_headers("POST", path),
            timeout=10,
        )
        resp.raise_for_status()
        order_id = resp.json().get("order", {}).get("id", "?")
        print(f"  [ORDER] {side.upper()} {ticker} @ {price:.3f} × {count}  id={order_id}")
        return True
    except Exception as e:
        print(f"  [ERROR] Order failed for {ticker}: {e}")
        return False


def close_position(ticker: str, price: float, size_usd: float) -> bool:
    """Close a YES position by buying NO at the equivalent price."""
    no_price = 1.0 - price
    return place_order(ticker, "no", size_usd, no_price)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    rf_model = m.load()
    if rf_model is None:
        print("[bot] No trained model found. Run  python train.py  first.")
        return

    perf_log  = PerformanceLog(risk_free_rate=RISK_FREE_RATE)
    positions: dict[str, tuple[PositionTracker, float]] = {}  # ticker → (tracker, size_usd)
    guard     = RiskGuard()

    mode  = "PAPER" if PAPER_MODE else "LIVE"
    print(f"\n{'='*60}")
    print(f"  Kalshi RF Bot  |  {mode}  |  poll {POLL_INTERVAL}s")
    print(f"  MIN_CONFIDENCE={MIN_CONFIDENCE}  MIN_EDGE={MIN_EDGE}  FEE={FEE_RATE:.0%}")
    print(f"  Kelly fraction={KELLY_FRACTION}  Max bankroll=${MAX_BANKROLL:.0f}  Cap=${MAX_POSITION_USD:.0f}")
    print(f"{'='*60}\n")

    # Print feature importances so we can see what the model is actually using
    importance = m.feature_importance(rf_model)
    if importance:
        top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        print("  Top-5 features by importance:")
        for feat, imp in top:
            print(f"    {feat:<20} {imp:.3f}")
        print()

    last_cleanup = time.time()

    while True:
        try:
            now = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] Scanning markets …")

            raw_df  = d.fetch_markets(limit=150)
            feat_df = d.build_features(raw_df)

            if feat_df.empty:
                print("  No feature data — sleeping.")
                time.sleep(POLL_INTERVAL)
                continue

            for idx, row in feat_df.iterrows():
                ticker       = row["ticker"]
                market_price = float(row["market_price"])
                days_left    = float(row["days_to_expiry"])
                feat_vec     = as_array({col: row[col] for col in FEATURE_COLS})
                model_prob   = m.predict_proba(rf_model, feat_vec)

                tracker, pos_size = positions.get(ticker, (None, 0.0))
                if tracker:
                    tracker.update(market_price)

                # ── EXIT (always allowed — circuit breaker never blocks exits) ──
                sell, reason = should_sell(market_price, model_prob, days_left)
                if tracker and sell:
                    print(f"  SELL [{reason}]  {row['title'][:60]}")
                    print(f"    exit={market_price:.3f}  model={model_prob:.3f}")
                    close_position(ticker, market_price, pos_size)
                    trade = tracker.close(market_price)
                    # Convert log return → dollar P&L so the risk guard tracks real $
                    import math
                    pnl_usd = pos_size * (math.exp(trade["log_return"]) - 1.0)
                    guard.record_trade(pnl_usd)
                    perf_log.record(trade)
                    sign = "+" if trade["log_return"] > 0 else ""
                    print(f"    return={sign}{trade['log_return']:.4f}  "
                          f"P&L=${pnl_usd:+.2f}  "
                          f"MAE={trade['mae']:.4f}  MFE={trade['mfe']:.4f}\n")
                    del positions[ticker]

                # ── ENTRY (blocked if any circuit breaker is tripped) ─────────
                elif not tracker:
                    allowed, block_reason = guard.can_open(len(positions))
                    if not allowed:
                        # Print once per scan cycle, not once per market
                        continue
                    enter, size = should_buy(market_price, model_prob)
                    if enter:
                        ev = expected_value(market_price, model_prob)
                        print(f"  BUY  {row['title'][:60]}")
                        print(f"    market={market_price:.3f}  model={model_prob:.3f}  "
                              f"EV={ev:+.3f}  size=${size:.2f}")
                        if place_order(ticker, "yes", size, market_price):
                            positions[ticker] = (
                                PositionTracker(ticker, market_price, model_prob),
                                size,
                            )
                        print()

            # ── Periodic summary ──────────────────────────────────────────────
            if perf_log.trades:
                perf_log.print_summary()

            allowed, block_reason = guard.can_open(len(positions))
            risk_line = f"  Risk: {guard.status()}"
            if not allowed:
                risk_line += f"\n  *** ENTRIES HALTED — {block_reason} ***"
            print(risk_line)
            print(f"  Open positions: {len(positions)}  |  sleeping {POLL_INTERVAL}s …\n")

            # DB cleanup once per day
            if time.time() - last_cleanup > 86400:
                db.cleanup_old(days=30)
                last_cleanup = time.time()

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print("\n[bot] Stopped.")
            perf_log.print_summary()
            break
        except Exception as e:
            print(f"[bot] Error: {e} — retrying in 60s")
            time.sleep(60)


if __name__ == "__main__":
    run()
