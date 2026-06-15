"""Audit the accumulated tick archive for continuity (gaps).

verify_feed.py checks the LIVE feed; this checks the PERSISTED record across all daily Parquet
files for a symbol — the source of truth for future backtests. Step Index ticks arrive ~1/s, so
any spacing > a few seconds is a gap (an outage the reconnect backfill should have filled).

Run:  python check_archive.py            # default symbol from config
      python check_archive.py R_50       # or any symbol
"""
from __future__ import annotations

import glob
import sys
from datetime import datetime, timezone

import pandas as pd

from config import CONFIG

GAP_THRESHOLD_S = 5  # spacing above this counts as a gap (well above the ~1s/2s tick cadence)


def _utc(epoch: int) -> str:
    return datetime.fromtimestamp(int(epoch), timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    symbol = sys.argv[1] if len(sys.argv) > 1 else CONFIG.symbol
    files = sorted(glob.glob(str(CONFIG.tick_dir / symbol / "*.parquet")))
    if not files:
        raise SystemExit(f"no tick files under {CONFIG.tick_dir / symbol}")

    df = (
        pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        .drop_duplicates("epoch")
        .sort_values("epoch")
        .reset_index(drop=True)
    )
    n = len(df)
    e0, e1 = int(df["epoch"].iloc[0]), int(df["epoch"].iloc[-1])
    span = e1 - e0
    diffs = df["epoch"].diff().dropna()
    gaps = [(int(df["epoch"][i - 1]), int(df["epoch"][i]), int(diffs[i]))
            for i in diffs.index if diffs[i] > GAP_THRESHOLD_S]
    missing = sum(d - 1 for _, _, d in gaps)  # approx ticks lost (1/s assumption)

    print(f"symbol:        {symbol}")
    print(f"files:         {len(files)}  ({', '.join(f.split(chr(92))[-1] for f in files)})")
    print(f"ticks on disk: {n:,}   monotonic={df['epoch'].is_monotonic_increasing}   "
          f"dup_epochs={int(df['epoch'].duplicated().sum())}")
    print(f"span:          {_utc(e0)} -> {_utc(e1)} UTC  ({span / 3600:.2f} h)")
    print(f"tick spacing:  median={diffs.median():.1f}s  max={int(diffs.max())}s")
    print(f"gaps > {GAP_THRESHOLD_S}s:    {len(gaps)}  (~{missing} ticks missing, "
          f"{100 * (1 - missing / span):.2f}% coverage)" if span else "")
    for a, b, d in sorted(gaps, key=lambda g: -g[2])[:15]:
        print(f"   {_utc(a)} -> {_utc(b)}   {d}s")
    if not gaps:
        print("   none - archive is continuous.")


if __name__ == "__main__":
    main()
