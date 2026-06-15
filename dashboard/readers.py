"""File readers for the dashboard — recent signals, backtest summary, archive health.

Reuses the Phase 2 building blocks (review_signals / backtest_signals) and the check_archive
parquet-load pattern. Backtest + health read the whole archive, so results are cached briefly.
"""
from __future__ import annotations

import glob
import time

import pandas as pd

from config import CONFIG
import backtest_signals as bt
import review_signals as rv

_cache: dict = {}


def _memo(key: tuple, ttl: float, fn):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    val = fn()
    _cache[key] = (now, val)
    return val


def recent_signals(symbol: str, limit: int = 100) -> list[dict]:
    sigs = rv._load_signals(symbol)
    sigs.sort(key=lambda s: s.get("bar_epoch", 0), reverse=True)
    return sigs[:limit]


def backtest_summary(symbol: str, payout: float | None = None, duration_bars: int | None = None) -> dict:
    key = ("bt", symbol, payout, duration_bars)
    return _memo(key, 10.0, lambda: bt.run_backtest(symbol, payout=payout, duration_bars=duration_bars))


def health(symbol: str) -> dict:
    return _memo(("health", symbol), 8.0, lambda: _health(symbol))


def _health(symbol: str) -> dict:
    files = sorted(glob.glob(str(CONFIG.tick_dir / symbol / "*.parquet")))
    sig_files = glob.glob(str(CONFIG.signal_dir / symbol / "*.jsonl"))
    sig_count = sum(sum(1 for _ in open(f, encoding="utf-8")) for f in sig_files)
    if not files:
        return {"symbol": symbol, "ticks": 0, "signals": sig_count, "last_tick_age_s": None,
                "coverage_pct": None, "gaps": None, "live": False}
    ep = (pd.concat([pd.read_parquet(f, columns=["epoch"]) for f in files], ignore_index=True)
          ["epoch"].drop_duplicates().sort_values().to_numpy())
    diffs = pd.Series(ep).diff().dropna()
    gaps = int((diffs > 5).sum())
    missing = int((diffs[diffs > 5] - 1).sum()) if gaps else 0
    span = int(ep[-1] - ep[0]) or 1
    age = time.time() - int(ep[-1])
    return {"symbol": symbol, "ticks": int(ep.size), "signals": sig_count,
            "last_tick_age_s": round(age, 1),
            "coverage_pct": round(100 * (1 - missing / span), 2),
            "gaps": gaps, "live": age < 120}
