"""
Performance metrics — Phases 5, 6, 7.

  - Log returns (Phase 6)
  - Sharpe Ratio (Phase 5)
  - MAE / MFE tracking (Phase 7)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


def log_return(p0: float, p1: float) -> float:
    """Phase 6: ln(P1/P0) — additive over time, symmetric for big moves."""
    return float(np.log(p1 / (p0 + 1e-9)))


def sharpe_ratio(log_returns: list[float], risk_free_rate: float = 0.05) -> float:
    """
    Phase 5: SR = (Rp - Rf) / σ
      SR < 1  → bad
      SR 1-2  → good
      SR > 2  → excellent
    """
    arr = np.array(log_returns)
    if len(arr) < 2 or arr.std() == 0:
        return 0.0
    # Annualise assuming ~365 trades/year (prediction markets are fast)
    daily_rf = risk_free_rate / 365.0
    excess   = arr - daily_rf
    return float(excess.mean() / excess.std() * np.sqrt(365))


@dataclass
class PositionTracker:
    """
    Tracks a single open position and computes MAE/MFE on close.

    MAE (Maximum Adverse Excursion)  — deepest drawdown before close
    MFE (Maximum Favorable Excursion) — highest peak before close
    """
    market_id:     str
    entry_price:   float
    model_prob:    float
    entry_time:    pd.Timestamp = field(default_factory=pd.Timestamp.now)
    _min_price:    float = field(init=False)
    _max_price:    float = field(init=False)

    def __post_init__(self):
        self._min_price = self.entry_price
        self._max_price = self.entry_price

    def update(self, current_price: float):
        self._min_price = min(self._min_price, current_price)
        self._max_price = max(self._max_price, current_price)

    def close(self, exit_price: float) -> dict:
        pnl = log_return(self.entry_price, exit_price)
        mae = log_return(self.entry_price, self._min_price)  # negative = loss
        mfe = log_return(self.entry_price, self._max_price)  # positive = gain
        return {
            "market_id":    self.market_id,
            "entry_price":  self.entry_price,
            "exit_price":   exit_price,
            "model_prob":   self.model_prob,
            "log_return":   pnl,
            "mae":          mae,
            "mfe":          mfe,
            "left_on_table": mfe - pnl,  # how much we missed vs peak
        }


class PerformanceLog:
    """Accumulates closed trade records and computes aggregate metrics."""

    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.trades: list[dict] = []

    def record(self, trade: dict):
        self.trades.append(trade)

    def summary(self) -> dict:
        if not self.trades:
            return {}
        df = pd.DataFrame(self.trades)
        wins = (df["log_return"] > 0).sum()
        sr   = sharpe_ratio(df["log_return"].tolist(), self.risk_free_rate)
        return {
            "total_trades": len(df),
            "win_rate":     wins / len(df),
            "sharpe_ratio": sr,
            "mean_return":  df["log_return"].mean(),
            "mean_mae":     df["mae"].mean(),
            "mean_mfe":     df["mfe"].mean(),
            "mean_left_on_table": df["left_on_table"].mean(),
        }

    def print_summary(self):
        s = self.summary()
        if not s:
            print("[metrics] No closed trades yet.")
            return
        sr = s["sharpe_ratio"]
        sr_label = "excellent" if sr > 2 else ("good" if sr >= 1 else "bad")
        print("\n── Performance Summary ──────────────────")
        print(f"  Trades      : {s['total_trades']}")
        print(f"  Win rate    : {s['win_rate']:.1%}")
        print(f"  Sharpe ratio: {sr:.3f}  [{sr_label}]")
        print(f"  Mean return : {s['mean_return']:.4f} (log)")
        print(f"  Mean MAE    : {s['mean_mae']:.4f}")
        print(f"  Mean MFE    : {s['mean_mfe']:.4f}")
        print(f"  Left on table (avg): {s['mean_left_on_table']:.4f}")
        print("─────────────────────────────────────────\n")
