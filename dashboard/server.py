"""FastAPI dashboard backend (read-only). REST over file readers + WS over the live feed.

Run:  .venv\\Scripts\\python -m uvicorn dashboard.server:app --port 8000
Then run the Vite dev server in dashboard/web (npm run dev), which proxies /api and /ws here.
NO trading endpoints — this is a research viewer.
"""
from __future__ import annotations

import glob
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config import CONFIG
from dashboard import readers
from dashboard.live import LiveFeed

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("dashboard.server")

feeds: dict[str, LiveFeed] = {}


def _dashboard_symbols() -> list[str]:
    """Symbols to display: those with a tick archive, plus the configured default."""
    syms: list[str] = []
    base = CONFIG.tick_dir
    if base.exists():
        for d in sorted(base.iterdir()):
            if d.is_dir() and not d.name.startswith("_") and glob.glob(str(d / "*.parquet")):
                syms.append(d.name)
    if CONFIG.symbol not in syms:
        syms.insert(0, CONFIG.symbol)
    return syms or [CONFIG.symbol]


@asynccontextmanager
async def lifespan(app: FastAPI):
    for sym in _dashboard_symbols():
        feed = LiveFeed(sym, CONFIG)
        feed.start()
        feeds[sym] = feed
        log.info("started live feed: %s", sym)
    yield
    for feed in feeds.values():
        await feed.stop()


app = FastAPI(title="Deriv Research Dashboard", lifespan=lifespan)


@app.get("/api/symbols")
def api_symbols():
    return [{"symbol": s, "live": f.connected} for s, f in feeds.items()]


# Chart timeframe ladder -> Deriv candle granularity (seconds). Display-only; detection is 1m/5m.
GRANULARITY = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400}


@app.get("/api/timeframes")
def api_timeframes():
    return list(GRANULARITY.keys())


@app.get("/api/candles")
async def api_candles(symbol: str, tf: str = "1m", count: int = 500):
    f = feeds.get(symbol)
    if not f:
        return JSONResponse({"error": "unknown symbol"}, status_code=404)
    g = GRANULARITY.get(tf)
    if g is None:
        return JSONResponse({"error": f"unsupported tf {tf}"}, status_code=400)
    return await f.history_candles(g, count)


@app.get("/api/signals")
def api_signals(symbol: str, limit: int = 100):
    return readers.recent_signals(symbol, limit)


@app.get("/api/ats")
def api_ats(symbol: str):
    """ATS Master Pattern overlay: HTF value lines + LTF pullback entries (display only)."""
    return readers.ats_overlay(symbol)


@app.get("/api/backtest")
def api_backtest(symbol: str, payout: float | None = None, duration_bars: int | None = None):
    return readers.backtest_summary(symbol, payout, duration_bars)


@app.get("/api/health")
def api_health(symbol: str):
    return readers.health(symbol)


@app.websocket("/ws")
async def ws(websocket: WebSocket, symbol: str):
    await websocket.accept()
    feed = feeds.get(symbol)
    if not feed:
        await websocket.close(code=1008)
        return
    q = feed.subscribe()
    try:
        if feed.last_tick:
            await websocket.send_json({"type": "tick", **feed.last_tick})
        while True:
            await websocket.send_json(await q.get())
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        feed.unsubscribe(q)


# In production, serve the built frontend (after `npm run build` in dashboard/web).
_dist = Path(__file__).resolve().parent / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="static")
