# Kalshi RF Trading Bot  (v2)

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# fill in KALSHI_API_KEY_ID and KALSHI_API_KEY_SECRET for live trading
```

---

## Quick Start

```bash
python train.py --markets 500    # train on resolved markets
python backtest.py --markets 1000  # check win rate + Sharpe before risking money
python bot.py                    # run (paper mode until API keys are set)
```

---

## What changed in v2

### 1. Features: 5 → 15
| Feature | v1 | v2 |
|---------|----|----|
| `price` | ✓ | ✓ |
| `volume_24h` | ✓ | ✓ |
| `momentum_7d` | broken (used current bid as "prior") | real 7d delta from SQLite history |
| `days_to_expiry` | ✓ | ✓ |
| `liquidity` | ✓ | ✓ |
| `spread_pct` | ✗ | bid-ask spread ÷ midpoint |
| `price_volatility` | ✗ | std dev of daily prices over 7d |
| `volume_accel` | ✗ | volume change vs yesterday |
| `market_age_days` | ✗ | days since market opened |
| `cat_politics/sports/…` | ✗ | one-hot category (5 categories) |

### 2. Entry logic: 1 gate → 3 gates
```
v1:  market_price <= model_prob * 0.50

v2:  model_prob >= MIN_CONFIDENCE (0.70)          ← same
     market_price <= model_prob * ENTRY_DISCOUNT  ← same
     fee_adjusted_EV >= MIN_EDGE (0.03)           ← NEW
```

The EV gate accounts for Kalshi's ~7% fee on winnings. Without it you can
pass the price gate but still be unprofitable after fees.

### 3. Position sizing: flat → Kelly criterion
```
v1:  always bet $50

v2:  f* = (b*p - q) / b          (Kelly fraction)
     size = f* × 0.25 × $1000   (quarter-Kelly × bankroll)
     size = min(size, $50)       (hard cap)
```
Quarter-Kelly means you're betting 25% of the mathematically optimal amount.
This is deliberately conservative — full Kelly is theoretically optimal but
causes extreme drawdowns in practice.

### 4. Fees modeled throughout
All return calculations now deduct Kalshi's fee from winning trades.
`backtest.py` accepts `--fee` and `--edge` arguments to test sensitivity.

### 5. Price history database
`db.py` maintains a local SQLite file (`price_history.db`).
Every scan saves a snapshot. After ~7 days of running, `momentum_7d`,
`price_volatility`, and `volume_accel` become real signals.
Re-run `train.py` after a week to let the model learn them.

---

## File map

```
bot.py          — live/paper trading loop
train.py        — trains Random Forest on resolved markets
backtest.py     — 70/30 walk-forward backtest with fees
model.py        — RF model (200 trees, calibrated)
data.py         — Kalshi API calls
features.py     — 15-feature vector + category classifier
db.py           — SQLite price history (momentum, volatility)
metrics.py      — log returns, Sharpe ratio, MAE/MFE
requirements.txt
.env.example
```

---

## Config reference

| Variable | Default | Meaning |
|----------|---------|---------|
| `MIN_CONFIDENCE` | 0.70 | Skip if model < 70% |
| `ENTRY_DISCOUNT` | 0.50 | Price must be ≤ model × 50% |
| `EXIT_TARGET` | 0.90 | Sell when price ≥ model × 90% |
| `EXPIRY_EXIT_DAYS` | 7 | Force-exit N days before close |
| `MIN_EDGE` | 0.03 | Min fee-adjusted EV per dollar |
| `KALSHI_FEE_RATE` | 0.07 | Kalshi's ~7% fee on winnings |
| `KELLY_FRACTION` | 0.25 | Fraction of full Kelly (0.25 = conservative) |
| `MAX_BANKROLL_USD` | 1000 | Total capital Kelly sizes against |
| `MAX_POSITION_USD` | 50 | Hard cap per trade |
| `POLL_INTERVAL_SEC` | 120 | Seconds between scans |

---

## Getting Kalshi API keys

1. Log into kalshi.com → Profile → API
2. Create a key pair (download the PEM private key)
3. Set `KALSHI_API_KEY_ID` and `KALSHI_API_KEY_SECRET` in `.env`
   - For `KALSHI_API_KEY_SECRET`: paste the full PEM contents as a single line
     with `\n` replacing actual newlines, or use a multi-line env var.

Leave blank to run in paper mode (signals printed, no orders placed).
