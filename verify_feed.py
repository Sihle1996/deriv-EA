"""Phase 1.5 feed-quality gate.

Static, one-shot checks that the data spine is trustworthy before any strategy work:

  1. Boundary alignment  — every base (1m) candle opens on an exact minute boundary.
  2. Gap detection       — consecutive base candles are exactly `base_granularity` apart.
  3. Resample fidelity   — our resampled 5m bars equal Deriv's NATIVE 5m candles, OHLC for OHLC.
  4. Tick archive        — today's Parquet reloads, has no duplicate epochs, and is ordered.

The 24h *stability* soak (no unrecovered disconnects, no missing candles, flat memory) is done
by running main.py for a day and watching the logs — it is not part of this static check.

Run:  python verify_feed.py
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import pandas as pd

from candles import _AGG, _OHLC_COLS
from config import CONFIG
from deriv_client import DerivClient

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("verify")

REL_TOL = 1e-6  # relative tolerance for OHLC float comparison


def _candles_to_df(candles: list[dict]) -> pd.DataFrame:
    idx = pd.to_datetime([int(c["epoch"]) for c in candles], unit="s", utc=True)
    df = pd.DataFrame(
        {k: [float(c[k]) for c in candles] for k in _OHLC_COLS},
        index=idx,
    )
    return df.sort_index()


def check_alignment(base: pd.DataFrame, granularity: int) -> bool:
    bad = [t for t in base.index if int(t.timestamp()) % granularity != 0]
    ok = not bad
    log.info("[%s] boundary alignment: %d/%d candles aligned to %ds",
             "PASS" if ok else "FAIL", len(base) - len(bad), len(base), granularity)
    return ok


def check_gaps(base: pd.DataFrame, granularity: int) -> bool:
    epochs = [int(t.timestamp()) for t in base.index]
    gaps = [
        (epochs[i - 1], epochs[i], epochs[i] - epochs[i - 1])
        for i in range(1, len(epochs))
        if epochs[i] - epochs[i - 1] != granularity
    ]
    ok = not gaps
    log.info("[%s] gap detection: %d gaps over %d candles",
             "PASS" if ok else "FAIL", len(gaps), len(base))
    for a, b, d in gaps[:5]:
        log.info("      gap: %ss between %s and %s",
                 d, datetime.fromtimestamp(a, tz=timezone.utc), datetime.fromtimestamp(b, tz=timezone.utc))
    return ok


def check_resample(base: pd.DataFrame, native5: pd.DataFrame) -> bool:
    grouped = base.resample("5min")
    agg = grouped.agg(_AGG)
    counts = grouped.size()
    full = agg[counts == 5]  # only fully-formed 5m bars (all five 1m candles present)
    common = full.index.intersection(native5.index)
    if len(common) == 0:
        log.info("[WARN] resample fidelity: no overlapping fully-formed 5m bars to compare")
        return True
    mismatches = []
    for t in common:
        for col in _OHLC_COLS:
            ours, theirs = full.at[t, col], native5.at[t, col]
            if abs(ours - theirs) > REL_TOL * max(1.0, abs(theirs)):
                mismatches.append((t, col, ours, theirs))
    ok = not mismatches
    log.info("[%s] resample fidelity: %d/%d 5m bars match native, %d field mismatches",
             "PASS" if ok else "FAIL", len(common) - len({m[0] for m in mismatches}), len(common), len(mismatches))
    for t, col, ours, theirs in mismatches[:5]:
        log.info("      mismatch @ %s %s: ours=%.6f native=%.6f", t, col, ours, theirs)
    return ok


def check_tick_archive() -> bool:
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    path = CONFIG.tick_dir / CONFIG.symbol / f"{today}.parquet"
    if not path.exists():
        log.info("[SKIP] tick archive: %s not found — run main.py first to collect ticks", path)
        return True
    df = pd.read_parquet(path)
    dupes = int(df["epoch"].duplicated().sum())
    ordered = df["epoch"].is_monotonic_increasing
    ok = dupes == 0 and ordered
    log.info("[%s] tick archive: %d ticks, %d duplicate epochs, ordered=%s (%s)",
             "PASS" if ok else "FAIL", len(df), dupes, ordered, path.name)
    return ok


async def main() -> None:
    cfg = CONFIG

    client = DerivClient(cfg)
    results: dict[str, bool] = {}

    async def on_connect(c: DerivClient) -> None:
        # Candle history is public on legacy v3 — no authorize needed for these checks.
        log.info("connected; fetching public history for %s ...\n", cfg.symbol)

        res1 = await c.send({
            "ticks_history": cfg.symbol, "style": "candles",
            "granularity": cfg.base_granularity, "count": 300, "end": "latest",
        })
        res5 = await c.send({
            "ticks_history": cfg.symbol, "style": "candles",
            "granularity": cfg.base_granularity * 5, "count": 60, "end": "latest",
        })
        base = _candles_to_df(res1["candles"])
        native5 = _candles_to_df(res5["candles"])

        results["alignment"] = check_alignment(base, cfg.base_granularity)
        results["gaps"] = check_gaps(base, cfg.base_granularity)
        results["resample"] = check_resample(base, native5)
        results["tick_archive"] = check_tick_archive()

        await c.close()

    await client.run(on_connect)

    log.info("\n%s", "=" * 50)
    passed = sum(results.values())
    for name, ok in results.items():
        log.info("  %-14s %s", name, "PASS" if ok else "FAIL")
    log.info("%s\n%d/%d checks passed", "=" * 50, passed, len(results))
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
