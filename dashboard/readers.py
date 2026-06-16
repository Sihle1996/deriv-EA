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


def archive_candles(symbol: str, granularity: int, count: int = 2000) -> list[dict]:
    """OHLC candles resampled from the TICK ARCHIVE (historical), for the chart's 'archive' view —
    so backfilled ATS value lines/entries (which live in the archived period) render in-window.
    Cached briefly; the live feed serves the real-time chart instead."""
    return _memo(("arch", symbol, granularity, count), 30.0,
                 lambda: _archive_candles(symbol, granularity, count))


def _archive_candles(symbol: str, granularity: int, count: int) -> list[dict]:
    ep, px = rv._load_ticks(symbol)
    if ep is None:
        return []
    s = pd.Series(px, index=pd.to_datetime(ep, unit="s", utc=True))
    ohlc = s.resample(f"{int(granularity)}s").ohlc().dropna().iloc[-count:]
    return [{"time": int(t.timestamp()), "open": float(r.open), "high": float(r.high),
             "low": float(r.low), "close": float(r.close)} for t, r in ohlc.iterrows()]


def ats_overlay(symbol: str, limit: int = 300) -> dict:
    """ATS Master Pattern overlay for the chart: the HTF (15m) value lines and the LTF (1m) pullback
    ENTRY markers, read from data/signals_ats/. Value lines are drawn as horizontal segments from
    each contraction's bar to the next; entries as arrows. Display only — NO trading."""
    sigs = rv._load_signals(symbol, signal_dir=CONFIG.ats_signal_dir)
    sigs.sort(key=lambda s: s.get("bar_epoch", 0))
    htf, ltf = CONFIG.ats_htf, CONFIG.ats_ltf
    value_lines = [{"epoch": s["bar_epoch"], "value_line": s["value_line"], "tf": s["timeframe"]}
                   for s in sigs if s.get("phase") == "contraction"
                   and s.get("value_line") is not None and s.get("timeframe") in (htf, ltf)]
    entries = [{"bar_epoch": s["bar_epoch"], "direction": s.get("direction"),
                "price": s.get("price_at_signal"), "tf": s["timeframe"],
                "value_line": s.get("value_line"), "htf_bias": s.get("htf_bias")}
               for s in sigs if s.get("phase") == "entry"]
    return {"symbol": symbol, "htf": htf, "ltf": ltf,
            "value_lines": value_lines[-limit:], "entries": entries[-limit:],
            "funnel": _ats_funnel(sigs, htf, ltf)}


def _ats_funnel(sigs: list[dict], htf: str, ltf: str) -> dict:
    """ATS funnel counts — shows WHERE the chain collapses (contraction → breakout → pullback →
    entry) and WHY entries are gated (no HTF bias vs counter-bias), without touching any rule."""
    from collections import Counter
    c = Counter((s.get("timeframe"), s.get("phase")) for s in sigs)
    blocked = [s for s in sigs if s.get("phase") == "entry_blocked"]
    no_bias = sum(1 for s in blocked if s.get("htf_bias") in (None, "none"))
    return {
        "htf_contractions": c.get((htf, "contraction"), 0),
        "htf_breakouts": c.get((htf, "breakout"), 0),
        "ltf_contractions": c.get((ltf, "contraction"), 0),
        "ltf_breakouts": c.get((ltf, "breakout"), 0),
        "pullback_candidates": c.get((ltf, "entry"), 0) + c.get((ltf, "entry_blocked"), 0),
        "entries": c.get((ltf, "entry"), 0),
        "blocked_no_bias": no_bias,
        "blocked_counter": len(blocked) - no_bias,
    }


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
