"""Minimal async Deriv WebSocket client (raw `websockets`, no SDK).

Transparent by design: every request/response is plain JSON over one socket. A supervisor
loop keeps the connection alive across drops with exponential backoff, and re-runs the
caller's `on_connect` hook each (re)connect — that hook re-authorizes and re-subscribes,
so subscriptions self-heal after a disconnect. This is what lets the spine survive a 24h soak.

Protocol notes:
- One-shot replies are correlated to requests by an incrementing `req_id`.
- Subscriptions stream many messages sharing the original `req_id`; the first is returned by
  `send()`, and every message (including the first) is also dispatched to stream handlers.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Awaitable, Callable

import websockets
from websockets.exceptions import ConnectionClosed

log = logging.getLogger("deriv.client")

Handler = Callable[[dict], None]
OnConnect = Callable[["DerivClient"], Awaitable[None]]


class DerivError(Exception):
    """A Deriv API error reply ({'error': {'code', 'message'}})."""

    def __init__(self, err: dict):
        self.code = err.get("code")
        self.message = err.get("message")
        super().__init__(f"{self.code}: {self.message}")


class DerivClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._req_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._handlers: list[Handler] = []
        self._stop = False

    # -- handler registration --------------------------------------------------------
    def add_handler(self, handler: Handler) -> None:
        """Register a sync callable invoked for every inbound message. Keep it fast —
        it runs on the read loop; heavy/blocking work would stall message reading."""
        self._handlers.append(handler)

    # -- supervisor ------------------------------------------------------------------
    async def run(self, on_connect: OnConnect) -> None:
        """Connect-and-serve forever. On each (re)connect, await on_connect(self) to
        (re)establish auth + subscriptions, then pump messages until the socket drops."""
        delay = self.cfg.reconnect_base_delay
        while not self._stop:
            try:
                async with websockets.connect(
                    self.cfg.ws_url,
                    ping_interval=self.cfg.ping_interval,
                    ping_timeout=self.cfg.ping_timeout,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    delay = self.cfg.reconnect_base_delay  # reset backoff after a good connect
                    log.info("connected: %s", self.cfg.ws_url)
                    # The read loop MUST run concurrently with on_connect: on_connect calls
                    # authorize()/subscribe(), which await reply futures that only the read loop
                    # can resolve. Starting it first avoids a deadlock (authorize would otherwise
                    # hang until its send-timeout and surface as a spurious TimeoutError).
                    reader = asyncio.create_task(self._read_loop(ws))
                    try:
                        await on_connect(self)
                        await reader  # block until the socket drops
                    finally:
                        if not reader.done():
                            reader.cancel()
                            try:
                                await reader
                            except (asyncio.CancelledError, Exception):
                                pass
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                log.warning("connection lost (%s)", e.__class__.__name__)
            except DerivError:
                raise  # auth/permission errors are fatal — don't spin reconnecting
            except Exception:
                log.exception("unexpected supervisor error")
            finally:
                self._fail_pending(ConnectionError("socket closed"))
                self._ws = None

            if self._stop:
                break
            log.info("reconnecting in %.1fs", delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, self.cfg.reconnect_max_delay)

    async def _read_loop(self, ws) -> None:
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    log.error("non-JSON message: %r", raw[:200])
                    continue
                self._dispatch(msg)
        finally:
            # Socket ended: unblock anyone awaiting a reply (e.g. authorize during on_connect)
            # instead of letting them hang until their send-timeout.
            self._fail_pending(ConnectionError("socket closed during read"))

    def _dispatch(self, msg: dict) -> None:
        rid = msg.get("req_id")
        if rid is not None:
            fut = self._pending.pop(rid, None)
            if fut is not None and not fut.done():
                if msg.get("error"):
                    fut.set_exception(DerivError(msg["error"]))
                else:
                    fut.set_result(msg)
        for handler in self._handlers:
            try:
                handler(msg)
            except Exception:
                log.exception("handler raised on msg_type=%s", msg.get("msg_type"))

    def _fail_pending(self, exc: Exception) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()

    # -- request/response ------------------------------------------------------------
    async def send(self, payload: dict, timeout: float = 20.0) -> dict:
        if self._ws is None:
            raise ConnectionError("not connected")
        self._req_id += 1
        rid = self._req_id
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        await self._ws.send(json.dumps({**payload, "req_id": rid}))
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(rid, None)

    # -- typed helpers ---------------------------------------------------------------
    async def authorize(self, token: str) -> dict:
        res = await self.send({"authorize": token})
        return res["authorize"]

    async def active_symbols(self, kind: str = "brief") -> list[dict]:
        res = await self.send({"active_symbols": kind, "product_type": "basic"})
        return res["active_symbols"]

    async def subscribe_candles(self, symbol: str, granularity: int, count: int) -> dict:
        return await self.send(
            {
                "ticks_history": symbol,
                "style": "candles",
                "granularity": granularity,
                "count": count,
                "end": "latest",
                "subscribe": 1,
            }
        )

    async def subscribe_ticks(self, symbol: str) -> dict:
        return await self.send({"ticks": symbol, "subscribe": 1})

    async def history_ticks(self, symbol: str, start: int, count: int) -> dict:
        """One-shot tick history (no subscribe) from `start` epoch to now, for gap backfill.
        Returns the `history` dict: {"prices": [...], "times": [...]} (may be empty)."""
        res = await self.send({
            "ticks_history": symbol,
            "style": "ticks",
            "start": int(start),
            "end": "latest",
            "count": count,
        })
        return res.get("history", {}) or {}

    async def close(self) -> None:
        self._stop = True
        if self._ws is not None:
            await self._ws.close()
