"""Multi-timeframe candle store + the MarketSnapshot struct.

We subscribe to ONE base candle stream and resample every higher timeframe from it in-process.
That means fewer live subscriptions and perfectly aligned bars (a 5m bar is exactly the five
1m bars it spans). MarketSnapshot is the ONLY surface downstream (Phase 2) strategy code should
consume — it must never touch pandas or websocket messages directly.

Caveat carried from the plan: on a CSPRNG synthetic the timeframes are statistically
self-similar. Multi-timeframe views organise logic and trade frequency; they do not stack
independent predictive edges. This store is neutral infrastructure.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger("deriv.candles")

_OHLC_COLS = ["open", "high", "low", "close"]
_AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}


@dataclass(frozen=True)
class Bar:
    time: pd.Timestamp  # candle open time (UTC)
    open: float
    high: float
    low: float
    close: float
    bars: int           # how many base candles make up this timeframe's history


@dataclass(frozen=True)
class TFView:
    """Per-timeframe indicator VALUES for the signal detector to consume — plain floats only,
    so the detector never touches pandas. Built from CLOSED bars (no forming bar). Indicator
    fields are None until enough closed bars exist (the warm-up gate). See _compute_view."""
    tf: str
    closed_bar_epoch: int       # open-time epoch of the last CLOSED bar (the detection bar)
    close: float
    high: float
    low: float
    n_bars: int                 # closed bars available (warm-up gate)
    atr: float | None = None
    band_width: float | None = None       # dimensionless volatility measure (see _compute_view)
    bw_threshold: float | None = None      # contraction quantile of band_width over vol_lookback
    bw_percentile: float | None = None     # rank of current band_width in [0,1] within vol_lookback
    bbw_zscore: float | None = None        # (band_width - mean) / std over vol_lookback
    range_high: float | None = None        # high over the last contraction_range_bars closed bars
    range_low: float | None = None         # low  over the last contraction_range_bars closed bars

    @property
    def warm(self) -> bool:
        return self.band_width is not None and self.bw_percentile is not None


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    tick_price: float | None
    tick_epoch: int | None
    frames: dict[str, Bar]      # latest bar per timeframe, e.g. "1m"/"5m"/"15m" (Phase 1 display)
    views: dict[str, TFView]    # indicator views for the signal timeframes (Phase 2 detector)


class MultiTimeframeStore:
    def __init__(
        self,
        symbol: str,
        timeframes: dict[str, str],
        max_base_rows: int = 5000,
        base_granularity: int = 60,
        signal_timeframes: tuple[str, ...] = (),
        signal_params: dict | None = None,
    ):
        self.symbol = symbol
        self.timeframes = timeframes
        self.max_base_rows = max_base_rows
        self.base_granularity = base_granularity
        self.signal_timeframes = signal_timeframes
        self.signal_params = signal_params or {}
        self._base = pd.DataFrame(columns=_OHLC_COLS, dtype="float64")
        self._base.index = pd.DatetimeIndex([], tz="UTC", name="open_time")

    # -- ingestion -------------------------------------------------------------------
    def load_history(self, candles: list[dict]) -> None:
        """Seed the base frame from the initial `candles` history array."""
        if not candles:
            return
        idx = pd.to_datetime([int(c["epoch"]) for c in candles], unit="s", utc=True)
        df = pd.DataFrame(
            {
                "open": [float(c["open"]) for c in candles],
                "high": [float(c["high"]) for c in candles],
                "low": [float(c["low"]) for c in candles],
                "close": [float(c["close"]) for c in candles],
            },
            index=idx,
        )
        df.index.name = "open_time"
        self._base = df[~df.index.duplicated(keep="last")].sort_index()
        self._trim()

    def upsert(self, ohlc: dict) -> tuple[pd.Timestamp, bool]:
        """Upsert the forming candle from an `ohlc` update.

        Returns (open_time, is_new) where is_new is True when a brand-new base candle began —
        i.e. the previous candle just closed. Callers print/act on candle close, not every tick.
        """
        t = pd.to_datetime(int(ohlc["open_time"]), unit="s", utc=True)
        is_new = t not in self._base.index
        self._base.loc[t] = [
            float(ohlc["open"]),
            float(ohlc["high"]),
            float(ohlc["low"]),
            float(ohlc["close"]),
        ]
        if is_new:
            self._base.sort_index(inplace=True)
            self._trim()
        return t, is_new

    def _trim(self) -> None:
        if len(self._base) > self.max_base_rows:
            self._base = self._base.iloc[-self.max_base_rows :]

    # -- views -----------------------------------------------------------------------
    def frame(self, tf: str) -> pd.DataFrame:
        """Resample the base frame to timeframe `tf`. The base label resamples to itself.
        NOTE: the last row may be a still-forming bar — use closed_frame() for detection."""
        rule = self.timeframes[tf]
        return self._base.resample(rule).agg(_AGG).dropna()

    def _expected_count(self, tf: str) -> int:
        """How many base candles make one closed bar of `tf` (e.g. 5m over 1m base -> 5)."""
        rule = self.timeframes[tf]
        return max(1, int(round(pd.Timedelta(rule) / pd.Timedelta(seconds=self.base_granularity))))

    def closed_frame(self, tf: str) -> pd.DataFrame:
        """Resample to `tf` keeping only FULLY CLOSED bars. Drops the still-forming base candle
        (last base row) first, then keeps only higher-TF groups that have all their constituent
        base bars — so the detector never sees a partial/mutating bar."""
        if len(self._base) == 0:
            return self._base
        base_closed = self._base.iloc[:-1]  # drop the forming base candle
        rule = self.timeframes[tf]
        grouped = base_closed.resample(rule)
        agg = grouped.agg(_AGG)
        counts = grouped.size()
        return agg[counts >= self._expected_count(tf)].dropna()

    @property
    def base(self) -> pd.DataFrame:
        return self._base

    def snapshot(self, tick_price: float | None, tick_epoch: int | None) -> MarketSnapshot:
        frames: dict[str, Bar] = {}
        for tf in self.timeframes:
            df = self.frame(tf)
            if df.empty:
                continue
            last = df.iloc[-1]
            frames[tf] = Bar(
                time=df.index[-1],
                open=float(last["open"]),
                high=float(last["high"]),
                low=float(last["low"]),
                close=float(last["close"]),
                bars=len(df),
            )
        views: dict[str, TFView] = {}
        for tf in self.signal_timeframes:
            view = _compute_view(self.closed_frame(tf), tf, self.signal_params)
            if view is not None:
                views[tf] = view
        return MarketSnapshot(self.symbol, tick_price, tick_epoch, frames, views)


def _compute_view(df: pd.DataFrame, tf: str, p: dict) -> TFView | None:
    """Build a TFView (plain floats) from a CLOSED-bar OHLC frame. Returns None if empty.
    Indicator fields stay None until enough closed bars exist (warm-up gate). All pandas lives
    here so the detector downstream only ever sees floats.

    Volatility measure = Bollinger band width: (2*bb_std*stdev(close,w)) / SMA(close,w) — it is
    dimensionless, so 1m and 5m are directly comparable. Contraction is judged by where the
    current band width sits within its own recent distribution (percentile + z-score)."""
    if df is None or df.empty:
        return None
    close, high, low = df["close"], df["high"], df["low"]
    n = len(df)
    epoch = int(df.index[-1].timestamp())
    view = dict(tf=tf, closed_bar_epoch=epoch, close=float(close.iloc[-1]),
                high=float(high.iloc[-1]), low=float(low.iloc[-1]), n_bars=n)

    atr_period = int(p.get("atr_period", 14))
    bb_window = int(p.get("bb_window", 20))
    bb_std = float(p.get("bb_std", 2.0))
    vol_lookback = int(p.get("vol_lookback", 100))
    range_bars = int(p.get("contraction_range_bars", 20))
    # Need a full lookback window of *defined* band-width values (band width is NaN for the first
    # bb_window-1 bars), plus enough bars for ATR.
    min_bars = max(atr_period + 1, bb_window + vol_lookback)
    if n < min_bars:
        return TFView(**view)  # not warm yet — floats present, indicators None

    # ATR (Wilder ~ EMA with alpha=1/period, adjust=False).
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
                   axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1 / atr_period, adjust=False).mean().iloc[-1])

    # Bollinger band width series, then its recent distribution.
    sma = close.rolling(bb_window).mean()
    sd = close.rolling(bb_window).std(ddof=0)
    bw_series = ((2 * bb_std * sd) / sma).dropna()
    recent = bw_series.iloc[-vol_lookback:]
    bw_t = float(bw_series.iloc[-1])
    r_mean, r_std = float(recent.mean()), float(recent.std(ddof=0))

    view.update(
        atr=atr,
        band_width=bw_t,
        bw_threshold=float(recent.quantile(p.get("contraction_pct", 0.20))),
        bw_percentile=float((recent <= bw_t).mean()),
        bbw_zscore=((bw_t - r_mean) / r_std) if r_std > 0 else 0.0,
        range_high=float(high.iloc[-range_bars:].max()),
        range_low=float(low.iloc[-range_bars:].min()),
    )
    return TFView(**view)
