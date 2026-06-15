"""TickStore — append every raw tick to a daily Parquet file.

Ticks are the archive of record: candles are derivable from ticks, but not the reverse. By
keeping every tick now, any future timeframe or pattern definition (contraction, expansion,
value line, master pattern) can be re-tested against the exact market we saw — no need to wait
months to re-collect data.

Layout: data/ticks/<symbol>/<UTC-date>.parquet  (one file per symbol per day).
Buffer in memory, flush every N ticks / on UTC-date rollover / on shutdown. Each flush appends
by rewriting the day's file with the concatenated rows (fine at this volume; a row-group append
engine can replace it later if needed).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger("deriv.storage")

_COLS = ["epoch", "quote", "symbol"]


class TickStore:
    def __init__(self, tick_dir: Path, symbol: str, flush_every: int = 100):
        self.dir = Path(tick_dir) / symbol
        self.dir.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol
        self.flush_every = max(1, flush_every)
        self._buffer: list[dict] = []
        self._buffer_date: str | None = None  # UTC date currently buffered
        self.total = 0

    @staticmethod
    def _utc_date(epoch: int) -> str:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")

    def append(self, epoch: int, quote: float) -> None:
        date = self._utc_date(epoch)
        # On UTC-date rollover, flush the previous day before buffering the new one.
        if self._buffer_date is not None and date != self._buffer_date:
            self.flush()
        self._buffer_date = date
        self._buffer.append({"epoch": int(epoch), "quote": float(quote), "symbol": self.symbol})
        self.total += 1
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def append_many(self, ticks: list[tuple[int, float]]) -> int:
        """Append a batch of (epoch, quote) ticks (used for reconnect gap-backfill). Reuses
        append() so date-rollover/flush logic still applies. Duplicate epochs are dropped at
        flush time, so overlap with already-stored or live ticks is harmless. Returns count."""
        for epoch, quote in ticks:
            self.append(int(epoch), float(quote))
        return len(ticks)

    def flush(self) -> None:
        if not self._buffer or self._buffer_date is None:
            return
        path = self.dir / f"{self._buffer_date}.parquet"
        new_rows = pd.DataFrame(self._buffer, columns=_COLS)
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows
        # Idempotent against reconnect replays: a tick is identified by its epoch.
        combined = combined.drop_duplicates(subset="epoch", keep="last").sort_values("epoch")
        combined.to_parquet(path, index=False)
        log.debug("flushed %d ticks -> %s (%d total in file)", len(self._buffer), path.name, len(combined))
        self._buffer.clear()

    def close(self) -> None:
        self.flush()


class SignalStore:
    """Append Phase 2 signal records to a daily JSONL file: data/signals/<symbol>/<UTC-date>.jsonl.

    JSONL (not Parquet) on purpose: signals are low-volume, heterogeneous (nullable fields differ
    by phase), and the Phase 2 mandate is that they be reviewable BY HAND (cat/jq). Deduped on
    (timeframe, bar_epoch, phase) — both within the session and against the day file already on
    disk — so reconnect/restart replays never double-log. Write-through (flush_every=1) by default
    so an ungraceful kill can't lose a (rare) signal."""

    def __init__(self, signal_dir: Path, symbol: str, flush_every: int = 1):
        self.dir = Path(signal_dir) / symbol
        self.dir.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol
        self.flush_every = max(1, flush_every)
        self._buffer: list[dict] = []
        self._date: str | None = None
        self._seen: set[tuple] = set()  # (timeframe, bar_epoch, phase) keys for the current date
        self.total = 0

    @staticmethod
    def _utc_date(epoch: int) -> str:
        return datetime.fromtimestamp(int(epoch), tz=timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _key(rec: dict) -> tuple:
        return (rec.get("timeframe"), rec.get("bar_epoch"), rec.get("phase"))

    def _load_seen(self, date: str) -> set[tuple]:
        path = self.dir / f"{date}.jsonl"
        seen: set[tuple] = set()
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        seen.add(self._key(json.loads(line)))
                    except json.JSONDecodeError:
                        continue
        return seen

    def append(self, record) -> bool:
        """Append one SignalRecord (or dict). Returns False if it was a duplicate (skipped)."""
        rec = record.to_dict() if hasattr(record, "to_dict") else dict(record)
        date = self._utc_date(rec["bar_epoch"])
        if self._date is None or date != self._date:
            self.flush()                       # flush prior day before rolling over
            self._date = date
            self._seen = self._load_seen(date)  # survive restarts: dedup vs what's already on disk
        key = self._key(rec)
        if key in self._seen:
            return False
        self._seen.add(key)
        self._buffer.append(rec)
        self.total += 1
        if len(self._buffer) >= self.flush_every:
            self.flush()
        return True

    def flush(self) -> None:
        if not self._buffer or self._date is None:
            return
        path = self.dir / f"{self._date}.jsonl"
        with open(path, "a", encoding="utf-8") as f:  # JSONL appends cleanly; dedup happened pre-write
            for rec in self._buffer:
                f.write(json.dumps(rec) + "\n")
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
