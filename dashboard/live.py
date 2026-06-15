"""Live market feed for the dashboard — one Deriv subscription per symbol, fanned out to WS clients.

Read-only viewer: reuses DerivClient + MultiTimeframeStore exactly like main.py's wire-up, but does
NOT persist (the collector bots already archive every tick). Each LiveFeed opens its own public
Deriv connection (no token) and broadcasts tick + forming-candle events to subscribed browsers.
"""
from __future__ import annotations

import asyncio
import logging

from candles import MultiTimeframeStore
from deriv_client import DerivClient

log = logging.getLogger("dashboard.live")


class LiveFeed:
    def __init__(self, symbol: str, cfg):
        self.symbol = symbol
        self.cfg = cfg
        # signal_timeframes=() -> no indicator views computed; this is a pure candle viewer.
        self.store = MultiTimeframeStore(symbol, cfg.timeframes, base_granularity=cfg.base_granularity)
        self.client = DerivClient(cfg)
        self.client.add_handler(self._handle)
        self.subscribers: set[asyncio.Queue] = set()
        self.last_tick: dict | None = None
        self.connected = False
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            await self.client.run(self._on_connect)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("live feed %s crashed", self.symbol)

    async def _on_connect(self, client: DerivClient) -> None:
        await client.subscribe_candles(self.symbol, self.cfg.base_granularity, self.cfg.history_count)
        await client.subscribe_ticks(self.symbol)
        self.connected = True
        log.info("dashboard live feed subscribed: %s", self.symbol)

    # runs on the client read loop (same event loop as FastAPI) -> put_nowait is safe
    def _handle(self, msg: dict) -> None:
        mt = msg.get("msg_type")
        if mt == "candles":
            self.store.load_history(msg["candles"])
        elif mt == "ohlc":
            o = msg["ohlc"]
            self.store.upsert(o)
            self._broadcast({"type": "candle", "bar": {
                "time": int(o["open_time"]), "open": float(o["open"]), "high": float(o["high"]),
                "low": float(o["low"]), "close": float(o["close"])}})
        elif mt == "tick":
            t = msg["tick"]
            self.last_tick = {"price": float(t["quote"]), "epoch": int(t["epoch"])}
            self._broadcast({"type": "tick", **self.last_tick})

    def _broadcast(self, event: dict) -> None:
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # slow client — drop the frame rather than block the read loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subscribers.discard(q)

    def candles(self, tf: str, count: int) -> list[dict]:
        tf = tf if tf in self.cfg.timeframes else "1m"
        df = self.store.frame(tf).tail(count)
        return [{"time": int(ts.timestamp()), "open": float(r.open), "high": float(r.high),
                 "low": float(r.low), "close": float(r.close)} for ts, r in df.iterrows()]

    async def stop(self) -> None:
        await self.client.close()
        if self._task:
            self._task.cancel()
