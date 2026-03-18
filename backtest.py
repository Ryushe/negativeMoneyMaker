"""
Walk-forward backtest — 70% train / 30% simulate.

Improvements over v1:
  - Kalshi fees deducted from winning trades
  - Reports EV-filtered trade count alongside raw win rate
  - Breaks out results by market category

Usage:
    python backtest.py
    python backtest.py --markets 1000 --fee 0.07 --edge 0.03
"""

import argparse

import numpy as np
import pandas as pd

import model as m
from features import FEATURE_COLS
from metrics import log_return, sharpe_ratio
from train import fetch_resolved, build_row

MIN_CONFIDENCE = 0.70
ENTRY_DISCOUNT = 0.50


def fee_adjusted_return(market_price: float, resolved_yes: bool, fee_rate: float) -> float:
    """Log return after Kalshi takes its fee from winnings."""
    if resolved_yes:
        payout = 1.0 - fee_rate * (1.0 - market_price)
        return log_return(market_price, payout)
    else:
        return log_return(market_price, 0.01)   # lost stake


def ev(market_price: float, model_prob: float, fee_rate: float) -> float:
    net_win  = (1.0 - market_price) * (1.0 - fee_rate)
    net_loss = market_price
    return model_prob * net_win - (1.0 - model_prob) * net_loss


def main(n_markets: int, fee_rate: float, min_edge: float):
    print(f"[backtest] Fetching {n_markets} resolved markets …")
    raw  = fetch_resolved(n_markets)
    rows = [r for r in (build_row(row) for _, row in raw.iterrows()) if r]
    df   = pd.DataFrame(rows)
    print(f"[backtest] Usable: {len(df)}")

    if len(df) < 60:
        print("[backtest] Need at least 60 samples.")
        return

    split    = int(len(df) * 0.70)
    train_df = df.iloc[:split].copy()
    test_df  = df.iloc[split:].copy()

    X_train, y_train = train_df[FEATURE_COLS].values, train_df["label"].values
    X_test,  y_test  = test_df[FEATURE_COLS].values,  test_df["label"].values

    print(f"[backtest] Train={len(train_df)}  Test={len(test_df)}")

    fitted = m.build_model()
    fitted.fit(X_train, y_train)

    trades, skipped = [], 0
    for i in range(len(X_test)):
        market_price = float(test_df.iloc[i]["price"])
        model_prob   = m.predict_proba(fitted, X_test[i])
        actual       = int(y_test[i])

        # Gate 1: confidence
        if model_prob < MIN_CONFIDENCE:
            skipped += 1
            continue

        # Gate 2: discount rule
        if market_price > model_prob * ENTRY_DISCOUNT:
            skipped += 1
            continue

        # Gate 3: fee-adjusted EV
        trade_ev = ev(market_price, model_prob, fee_rate)
        if trade_ev < min_edge:
            skipped += 1
            continue

        lr = fee_adjusted_return(market_price, actual == 1, fee_rate)
        trades.append({
            "market_price": market_price,
            "model_prob":   model_prob,
            "ev":           trade_ev,
            "actual":       actual,
            "log_return":   lr,
            "win":          actual == 1,
            # category (any cat_ column that is 1)
            "category":     next(
                (col[4:] for col in FEATURE_COLS if col.startswith("cat_") and test_df.iloc[i][col] == 1.0),
                "other",
            ),
        })

    if not trades:
        print("[backtest] No trades triggered — loosen MIN_EDGE or MIN_CONFIDENCE.")
        return

    tdf      = pd.DataFrame(trades)
    win_rate = tdf["win"].mean()
    sr       = sharpe_ratio(tdf["log_return"].tolist())
    sr_label = "excellent" if sr > 2 else ("good" if sr >= 1 else "bad")

    print("\n── Backtest Results (fee-adjusted) ───────────────────")
    print(f"  Trades triggered : {len(tdf)}  (skipped {skipped})")
    print(f"  Win rate         : {win_rate:.1%}")
    print(f"  Sharpe ratio     : {sr:.3f}  [{sr_label}]")
    print(f"  Mean log return  : {tdf['log_return'].mean():.4f}")
    print(f"  Total log return : {tdf['log_return'].sum():.4f}")
    print(f"  Mean EV at entry : {tdf['ev'].mean():.4f}")

    # Per-category breakdown
    print("\n  By category:")
    for cat, grp in tdf.groupby("category"):
        cat_sr = sharpe_ratio(grp["log_return"].tolist()) if len(grp) > 1 else 0.0
        print(f"    {cat:<14} trades={len(grp):3d}  win={grp['win'].mean():.0%}  SR={cat_sr:.2f}")

    print("──────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", type=int, default=500)
    parser.add_argument("--fee",     type=float, default=0.07, help="Kalshi fee rate (default 0.07)")
    parser.add_argument("--edge",    type=float, default=0.03, help="Min EV threshold (default 0.03)")
    args = parser.parse_args()
    main(args.markets, args.fee, args.edge)
