"""Phase 1 data spine: connect to Deriv DEMO, stream ticks + base candles, resample higher
timeframes in-process, persist every tick, and print a MarketSnapshot on each candle close.

NO trading. NO dashboard. Demo-only — a real-money token aborts on startup by design.

Run:  python main.py   (after putting a DEMO token in .env)
Stop: Ctrl-C           (flushes the tick buffer to Parquet)
"""
from __future__ import annotations

import asyncio
import glob
import logging
import time

import pandas as pd

from ats_signals import AtsEngine
from candles import MultiTimeframeStore
from config import CONFIG
from deriv_client import DerivClient
from storage import SignalStore, TickStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("deriv.main")


class Spine:
    def __init__(self, cfg):
        self.cfg = cfg
        # The store computes a TFView (ATR + inside-bar contraction box) for the ATS HTF + LTF.
        self.store = MultiTimeframeStore(
            cfg.symbol, cfg.timeframes, cfg.max_base_rows,
            base_granularity=cfg.base_granularity,
            signal_timeframes=cfg.all_signal_timeframes,
            signal_params=cfg.view_params(),
        )
        self.tick_store = TickStore(cfg.tick_dir, cfg.symbol, cfg.tick_flush_every)
        # ATS Master Pattern detection + logging (NO trading). tf_seconds maps each timeframe label
        # to its length in seconds so the engine can stamp bar_close_epoch (the look-ahead firewall).
        tf_seconds = {
            tf: int(pd.Timedelta(cfg.timeframes[tf]).total_seconds())
            for tf in cfg.all_signal_timeframes
        }
        self.ats_engine = AtsEngine(
            cfg.symbol, cfg.ats_signal_params(), cfg.ats_htf, cfg.ats_ltf, tf_seconds,
            cfg.ats_signal_version, cfg.ats_params_hash(),
        )
        self.ats_store = SignalStore(cfg.ats_signal_dir, cfg.symbol, cfg.signal_flush_every)
        self.last_quote: float | None = None
        self.last_epoch: int | None = None
        self.tick_count = 0
        self._symbols_printed = False

    def seed_last_epoch_from_archive(self) -> None:
        """Make process RESTARTS gap-free too: seed last_epoch from the newest archived tick so
        the first connect backfills the downtime (reusing the reconnect path). Only seeds if the
        archive is recent enough for backfill to plausibly cover the gap; otherwise starts fresh."""
        files = sorted(glob.glob(str(self.cfg.tick_dir / self.cfg.symbol / "*.parquet")))
        if not files:
            return
        try:
            last_ep = int(pd.read_parquet(files[-1], columns=["epoch"])["epoch"].max())
        except Exception:
            log.exception("could not read archive to seed last_epoch")
            return
        age = time.time() - last_ep
        if 0 < age <= self.cfg.backfill_count:
            self.last_epoch = last_ep
            log.info("archive resumes %.0fs ago -> first connect will backfill the restart gap", age)
        elif age > self.cfg.backfill_count:
            log.info("archive last tick is %.0fs old (> backfill window); starting fresh", age)

    # -- runs on every (re)connect; re-establishes auth + subscriptions --------------
    async def on_connect(self, client: DerivClient) -> None:
        # Capture the gap boundary BEFORE live ticks resume (handle() bumps last_epoch the
        # instant the new subscription delivers a tick). If last_epoch is set, this is a
        # reconnect and [gap_start+1, now] is the window we missed.
        gap_start = self.last_epoch
        is_reconnect = gap_start is not None

        if self.cfg.authenticate:
            # Account access (later phases). Legacy v3 authorize needs a CLASSIC token; a
            # pat_ token will fail here with InvalidToken — that's expected, see config.py.
            auth = await client.authorize(self.cfg.api_token)
            if not auth.get("is_virtual"):
                # HARD BLOCK — never run against real money.
                raise SystemExit(
                    "REAL ACCOUNT BLOCKED — demo-only. Use a virtual/demo account token."
                )
            log.info(
                "authorized: loginid=%s is_virtual=%s balance=%s %s",
                auth.get("loginid"), auth.get("is_virtual"), auth.get("balance"), auth.get("currency"),
            )
        else:
            # Phase 1: public market data only — no token, no account access, nothing to trade.
            log.info("UNAUTHENTICATED data-only mode: public market data on %s (no account access)",
                     self.cfg.symbol)
        if not self._symbols_printed:
            await self._print_step_symbols(client)
            self._symbols_printed = True
        # Resume live FIRST (minimise any new gap), then backfill the outage window.
        await client.subscribe_candles(self.cfg.symbol, self.cfg.base_granularity, self.cfg.history_count)
        await client.subscribe_ticks(self.cfg.symbol)
        log.info(
            "subscribed: %ds base candles + ticks on %s; higher TFs resampled in-process",
            self.cfg.base_granularity, self.cfg.symbol,
        )
        if is_reconnect:
            await self._backfill_ticks(client, gap_start)

    async def _backfill_ticks(self, client: DerivClient, gap_start: int) -> None:
        """After a reconnect/restart, refetch ticks missed during the outage so the archive stays
        gap-free. Candles self-heal via the history reload; only the live tick stream has holes.

        Subtlety: Deriv's ticks_history 'latest' LAGS the live stream by several seconds, so a
        single fetch leaves a seam between where history ends and where the live subscription
        began. We therefore loop — fetching, then waiting for history to catch up — until the
        backfilled range meets the first live tick (`target`). Live ticks from `target` onward are
        already captured by the subscription, so backfill only needs to reach target-1."""
        # Wait briefly for the first live tick so we know where the live range starts.
        for _ in range(20):
            if self.last_epoch is not None and self.last_epoch > gap_start:
                break
            await asyncio.sleep(0.25)
        target = self.last_epoch  # first live tick epoch; backfill must reach this to close the seam

        cursor, total, capped = gap_start, 0, False
        for _ in range(40):  # bounded; history advances ~1 tick/s, lag is normally < 30s
            try:
                hist = await client.history_ticks(self.cfg.symbol, cursor + 1, self.cfg.backfill_count)
            except Exception:
                log.exception("tick backfill request failed (after epoch %s)", cursor)
                break
            times = hist.get("times", []) or []
            prices = hist.get("prices", []) or []
            limit = target if target is not None else (1 << 62)
            batch = [(int(e), float(p)) for e, p in zip(times, prices) if cursor < int(e) < limit]
            if batch:
                total += self.tick_store.append_many(batch)
                cursor = batch[-1][0]
                capped = capped or len(batch) >= self.cfg.backfill_count
            if target is None or cursor >= target - 1:
                break
            await asyncio.sleep(1.0)  # let the lagging history endpoint catch up, then fetch the rest

        if total:
            log.info("reconnect backfill: spliced %d ticks, reached epoch %s (gap after %s, live from %s)%s",
                     total, cursor, gap_start, target,
                     "  [WARN: hit backfill_count cap]" if capped else "")
        else:
            log.info("reconnect backfill: nothing to splice after epoch %s", gap_start)

    async def _print_step_symbols(self, client: DerivClient) -> None:
        try:
            symbols = await client.active_symbols()
        except Exception:
            log.exception("active_symbols failed")
            return
        step = [
            s for s in symbols
            if "step" in (str(s.get("display_name", "")) + str(s.get("symbol", ""))).lower()
        ]
        log.info("Step-related symbols (%d found):", len(step))
        for s in step:
            log.info("  %-10s %s", s.get("symbol"), s.get("display_name"))
        chosen = next((s for s in symbols if s.get("symbol") == self.cfg.symbol), None)
        if chosen:
            log.info("--> using %s (%s)", chosen.get("symbol"), chosen.get("display_name"))
        else:
            log.warning(
                "configured symbol %r not found in active_symbols — pick one from the list above",
                self.cfg.symbol,
            )

    # -- single sync message handler (runs on the read loop) -------------------------
    def handle(self, msg: dict) -> None:
        mt = msg.get("msg_type")
        if mt == "candles":
            self.store.load_history(msg["candles"])
            log.info("loaded %d base candles", len(msg["candles"]))
            # Print a snapshot, but do NOT run signals on the history reload — that would replay a
            # burst of stale historical signals on every reconnect (dedup would catch them, but
            # skipping is cleaner). Detection happens only on live candle close below.
            self._print_snapshot(self.store.snapshot(self.last_quote, self.last_epoch))
        elif mt == "ohlc":
            _, is_new = self.store.upsert(msg["ohlc"])
            if is_new:  # a base candle just closed -> one clean snapshot per minute
                snap = self.store.snapshot(self.last_quote, self.last_epoch)  # build ONCE
                self._print_snapshot(snap)
                self._run_signals(snap)
        elif mt == "tick":
            tick = msg["tick"]
            self.last_quote = float(tick["quote"])
            self.last_epoch = int(tick["epoch"])
            self.tick_store.append(self.last_epoch, self.last_quote)
            self.tick_count += 1
            if self.tick_count % 30 == 0:  # prove the stream is live without spamming
                log.info("ticks=%d last=%.5f saved=%d", self.tick_count, self.last_quote, self.tick_store.total)
        elif msg.get("error") and msg.get("msg_type") not in ("candles", "ohlc", "tick"):
            log.error("API error: %s", msg["error"])

    def _run_signals(self, snap) -> None:
        """Feed the closed-bar snapshot to the ATS detector and LOG any signals. No trading."""
        for rec in self.ats_engine.on_snapshot(snap):
            if self.ats_store.append(rec):
                arrow = {"up": "^", "down": "v"}.get(rec.direction or "", "")
                vl = f"{rec.value_line:.5f}" if rec.value_line is not None else "-"
                log.info("ATS %s %s %s%s  value=%s price=%.5f bias=%s",
                         rec.timeframe, rec.phase, rec.direction or "-", arrow,
                         vl, rec.price_at_signal, rec.htf_bias or "-")

    def _print_snapshot(self, snap) -> None:
        if not snap.frames:
            return
        parts = [
            f"{tf} O{b.open:.5f} H{b.high:.5f} L{b.low:.5f} C{b.close:.5f}[{b.bars}]"
            for tf, b in snap.frames.items()
        ]
        price = f"{snap.tick_price:.5f}" if snap.tick_price is not None else "-"
        log.info("SNAP %s tick=%s | %s", snap.symbol, price, "   ".join(parts))


async def main() -> None:
    cfg = CONFIG
    if cfg.authenticate and not cfg.api_token:
        raise SystemExit("DERIV_AUTHENTICATE=true but no DERIV_API_TOKEN set in .env.")
    spine = Spine(cfg)
    spine.seed_last_epoch_from_archive()  # restart gap-free (within the backfill window)
    client = DerivClient(cfg)
    client.add_handler(spine.handle)
    try:
        await client.run(spine.on_connect)
    finally:
        spine.tick_store.close()
        spine.ats_store.close()
        log.info("flushed; %d ticks archived, %d ATS signals logged this session",
                 spine.tick_store.total, spine.ats_store.total)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("shutdown requested")
