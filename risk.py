"""
Two-layer circuit breaker — standard prop-desk risk pattern.

Layer 1 — Daily loss limit (hard stop)
    If the day's realised P&L drops below MAX_DAILY_LOSS_USD the bot
    stops opening new positions for the rest of the calendar day.
    Existing positions can still be closed. Resets automatically at midnight.

Layer 2 — Consecutive loss cool-off (soft stop)
    After N losses in a row the bot pauses for COOLOFF_MINUTES before
    resuming. This catches short-term model drift or correlated bad luck
    without killing the session entirely.

Layer 3 — Max concurrent positions
    Caps how many open trades the bot can hold at once, limiting
    correlated-event exposure.

Why these three specifically:
    - Daily limit   → detects "model is broken today" or a code bug
    - Consecutive   → detects "market regime shifted right now"
    - Position cap  → limits damage from one correlated shock hitting
                      multiple open positions simultaneously

All three only block new entries. Exits always go through.
"""

import datetime
import os
from dataclasses import dataclass, field


# ── Config (read once at import) ─────────────────────────────────────────────
MAX_DAILY_LOSS_USD    = float(os.getenv("MAX_DAILY_LOSS_USD",    "15"))
MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3"))
COOLOFF_MINUTES       = int(os.getenv("COOLOFF_MINUTES",         "60"))
MAX_OPEN_POSITIONS    = int(os.getenv("MAX_OPEN_POSITIONS",       "10"))


@dataclass
class RiskGuard:
    """
    Stateful circuit breaker. One instance lives for the lifetime of the bot.
    Call record_trade() after every close, can_open() before every entry.
    """

    # ── Daily tracking ────────────────────────────────────────────────────────
    _day:          datetime.date    = field(default_factory=datetime.date.today)
    _daily_pnl:    float            = 0.0   # realised USD P&L today

    # ── Consecutive loss tracking ─────────────────────────────────────────────
    _consecutive:  int              = 0
    _cooloff_until: datetime.datetime | None = None

    def _roll_day_if_needed(self):
        """Reset daily P&L counter at midnight."""
        today = datetime.date.today()
        if today != self._day:
            self._day       = today
            self._daily_pnl = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def record_trade(self, pnl_usd: float):
        """
        Call after every position close.
        pnl_usd — realised dollar profit/loss for that trade (negative = loss).
        """
        self._roll_day_if_needed()
        self._daily_pnl += pnl_usd

        if pnl_usd < 0:
            self._consecutive += 1
            if self._consecutive >= MAX_CONSECUTIVE_LOSSES:
                resume = datetime.datetime.now() + datetime.timedelta(minutes=COOLOFF_MINUTES)
                self._cooloff_until = resume
                print(
                    f"\n  [RISK] {MAX_CONSECUTIVE_LOSSES} consecutive losses — "
                    f"cool-off until {resume.strftime('%H:%M:%S')}  "
                    f"(layer 2)\n"
                )
        else:
            self._consecutive = 0   # reset streak on any win

    def can_open(self, open_position_count: int) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        Call before placing any new entry order.
        Exits are never blocked — only entries.
        """
        self._roll_day_if_needed()

        # ── Layer 1: daily loss limit ────────────────────────────────────────
        if self._daily_pnl <= -MAX_DAILY_LOSS_USD:
            return False, (
                f"Daily loss limit hit  "
                f"(today: ${self._daily_pnl:.2f} / limit: -${MAX_DAILY_LOSS_USD:.2f})  "
                f"[layer 1 — resets midnight]"
            )

        # ── Layer 2: consecutive loss cool-off ───────────────────────────────
        if self._cooloff_until and datetime.datetime.now() < self._cooloff_until:
            remaining = (self._cooloff_until - datetime.datetime.now()).seconds // 60
            return False, (
                f"Cool-off active — {remaining}m remaining  "
                f"(after {MAX_CONSECUTIVE_LOSSES} consecutive losses)  [layer 2]"
            )
        elif self._cooloff_until and datetime.datetime.now() >= self._cooloff_until:
            self._cooloff_until  = None
            self._consecutive    = 0
            print("  [RISK] Cool-off expired — resuming entries.\n")

        # ── Layer 3: max concurrent positions ────────────────────────────────
        if open_position_count >= MAX_OPEN_POSITIONS:
            return False, (
                f"Max open positions reached  "
                f"({open_position_count}/{MAX_OPEN_POSITIONS})  [layer 3]"
            )

        return True, "ok"

    def status(self) -> str:
        """One-line summary for the bot's periodic print."""
        self._roll_day_if_needed()
        cooloff_str = ""
        if self._cooloff_until and datetime.datetime.now() < self._cooloff_until:
            remaining   = (self._cooloff_until - datetime.datetime.now()).seconds // 60
            cooloff_str = f"  COOLOFF {remaining}m"
        return (
            f"day P&L: ${self._daily_pnl:+.2f} / -${MAX_DAILY_LOSS_USD:.2f}  "
            f"streak: {self._consecutive} losses{cooloff_str}"
        )
