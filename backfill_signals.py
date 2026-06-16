"""Regenerate the COMPLETE ATS signal set offline from the gap-free tick archive.

The live detector in main.py can miss signals for candles that closed during a network outage
(those closes are never delivered live). But every tick is archived gap-free, so the authoritative
signal set can always be rebuilt from the ticks. This replays the exact same store + ATS engine over
the whole tick archive and writes any signals not already logged — deduped on (tf, bar_epoch, phase)
— so your review set is complete and reproducible no matter what the network did.

Run:  python backfill_signals.py --symbol stpRNG
      python backfill_signals.py --symbol stpRNG --dry-run   # report only (read-only, safe anytime)

NOTE: for a clean canonical write, run this with main.py STOPPED (two processes appending the same
JSONL could interleave). Candles here are resampled from the archived ticks; live detection uses
Deriv's native candles — they match to within resample fidelity.
"""
from __future__ import annotations

import argparse

import pandas as pd

from ats_signals import AtsEngine
from candles import MultiTimeframeStore
from config import CONFIG
from storage import SignalStore
import review_signals as rv


def build_candles(epochs, prices) -> list[dict]:
    df = pd.DataFrame({"price": prices}, index=pd.to_datetime(epochs, unit="s", utc=True))
    ohlc = df["price"].resample("1min").ohlc().dropna()
    return [{"open_time": int(t.timestamp()), "open": float(r.open), "high": float(r.high),
             "low": float(r.low), "close": float(r.close)} for t, r in ohlc.iterrows()]


def replay(symbol: str, candles: list[dict]) -> list:
    """Replay candles through the store + ATS engine exactly as main.py does (upsert, act on candle
    close). Returns the full list of SignalRecords the ATS detector would have produced."""
    store = MultiTimeframeStore(symbol, CONFIG.timeframes, base_granularity=CONFIG.base_granularity,
                                signal_timeframes=CONFIG.all_signal_timeframes,
                                signal_params=CONFIG.view_params())
    tf_seconds = {tf: int(pd.Timedelta(CONFIG.timeframes[tf]).total_seconds())
                  for tf in CONFIG.all_signal_timeframes}
    engine = AtsEngine(symbol, CONFIG.ats_signal_params(), CONFIG.ats_ladder,
                       tf_seconds, CONFIG.ats_signal_version, CONFIG.ats_params_hash())
    records, prev = [], None
    for c in candles:
        _, is_new = store.upsert(c)
        if is_new and prev is not None:
            records.extend(engine.on_snapshot(store.snapshot(c["close"], c["open_time"])))
        prev = c
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", default=None, help="symbol (positional)")
    ap.add_argument("--symbol", dest="symbol_flag", default=None, help="symbol (flag form)")
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing (read-only)")
    args = ap.parse_args()
    symbol = args.symbol_flag or args.symbol or CONFIG.symbol

    epochs, prices = rv._load_ticks(symbol)
    if epochs is None:
        raise SystemExit(f"no tick archive under {CONFIG.tick_dir / symbol}")
    candles = build_candles(epochs, prices)
    records = replay(symbol, candles)
    print(f"symbol: {symbol}   ticks: {epochs.size:,}   candles: {len(candles)}   "
          f"ATS signals regenerated: {len(records)}")
    print("(tip: run check_archive.py first - gaps in the tick archive become holes here too)")

    store = SignalStore(CONFIG.ats_signal_dir, symbol, CONFIG.signal_flush_every)
    if args.dry_run:
        new = 0
        seen_by_date: dict[str, set] = {}
        for r in records:
            d = SignalStore._utc_date(r.bar_epoch)
            seen = seen_by_date.setdefault(d, store._load_seen(d))
            key = (r.timeframe, r.bar_epoch, r.phase)
            if key not in seen:
                new += 1
                seen.add(key)
        print(f"DRY RUN: {new} new signals would be added, {len(records) - new} already logged.")
        return

    added = sum(store.append(r) for r in records)
    store.close()
    print(f"wrote {added} new signals, skipped {len(records) - added} already logged "
          f"-> signals_ats/{symbol}/")


if __name__ == "__main__":
    main()
