# Deriv ATS "Forex Master Pattern" research bot — full brief + source

> Paste this whole file into another model to bring it fully up to speed. The narrative is first;
> **every source file is appended verbatim at the bottom** under "## Source code".

## One line
A Python research harness that streams Deriv market data, detects the TradeATS "Forex Master
Pattern" structurally, **logs** trade signals (no live trading), and **rigorously tests whether those
signals have any real predictive edge** — fronted by a read-only React dashboard.

## Core philosophy (the whole point)
Primary markets are **Deriv synthetic indices** (Step Index, Volatility 50…), which are
**CSPRNG-generated** — provably random, no order flow/news/liquidity. So on synthetics **no chart
pattern can have predictive edge by construction**, and fixed-payout contracts carry a structural
house edge (negative EV). The bot is therefore an **honest measurement instrument**, not a profit
machine: it tries to *falsify* the strategy. Expected/correct answer on synthetics = "no edge."
Real markets (FX, Gold, NAS100) are collected in parallel as the only place edge is even possible.
**No real-money execution code exists.**

## Market & API
- Deriv **legacy WebSocket API** (`wss://ws.derivws.com/websockets/v3`); market data
  (ticks/candles/active_symbols) is **public — no token**. Auth only needed for account calls (none yet).
- New Deriv accounts mint `pat_` tokens for a *newer* Options API that the legacy `authorize` rejects;
  irrelevant while data is public. A future trading phase would add the new OTP auth flow.
- Hand-written async client (raw `websockets`): req_id correlation, concurrent read loop, auto-reconnect
  + backoff, resubscribe-on-reconnect, and **MarketIsClosed treated as transient** (real-market bots
  survive weekend/session closes instead of crashing).

## Data spine
- Per symbol: one **1-minute base candle stream + raw tick stream**; higher TFs (5m/15m/1h) are
  **resampled in-process** with pandas.
- **Every tick persisted** to daily Parquet (`data/ticks/<symbol>/<date>.parquet`) — ticks are the
  archive of record, so any rule can be re-tested.
- **Gap-free archive:** reconnect/restart backfills the missed window via `ticks_history` (looping to
  beat the history endpoint's lag) so there are no holes. 24h soak: 100% coverage, 0 gaps.

## ATS detector (`ats_signals.py`, `ats_v3`) — run on candle close, pure floats
- **Contraction:** swing-pivot compression (pivot-high lower than prior AND pivot-low higher than
  prior). Freezes a **value line = mean of closes inside the box** (Fair Market Value), plus box hi/lo.
- **Expansion:** close beyond a box boundary (breakout / liquidity sweep).
- **Entry** (two modes): `value_fade` (default) = enter *against* the spike, target = value line;
  `continuation` = enter on the pull-back *to* the value line, in the breakout direction.
- **Timeframe ladder `1h→15m→5m→1m`:** each TF's entries are **gated by the next-higher TF's bias**
  (which side of *its* value line the higher-TF close is). Counter-trend entries are logged as
  `entry_blocked` (not traded). Top rung = bias/context only. One entry per contraction block.
- Each entry stores a structural **`stop_ref`** (expansion swing extreme) for bracket testing.

## Signals & logging
JSONL per phase (`data/signals_ats/<symbol>/<date>.jsonl`) carrying everything to recompute outcomes
(prices, value line, box, stop_ref, HTF bias, episode id, params hash, version). `backfill_signals.py`
regenerates the complete authoritative set from the gap-free tick archive (deduped, reproducible).

## Validation pipeline (treats any "edge" as a null hypothesis to disprove)
- `review_signals.py` — forward return / MFE / MAE / first-touch vs a random-entry null; look-ahead
  firewall (ticks after bar close only) + gap rejection.
- `backtest_signals.py` — Rise/Fall binary P&L vs null (break-even win rate 51.3%).
- `bracket_backtest.py` — the **structural CFD bracket** the method actually uses (SL just beyond the
  expansion extreme, TP1=value line w/ partial + break-even runner, TP2=box far side); walks real ticks
  for first-touch fills; **R-multiple** P&L vs a same-geometry random-location null; configurable cost.
- `validate_signals.py` — permutation test (p-value), walk-forward OOS, **PBO via CSCV**, deflated/
  expected-max Sharpe, family-wise (Bonferroni) correction. `--quick` skips the heavy sweep.
- `verify_feed.py` / `verify_ats.py` / `verify_validation.py` — unit tests on known-truth inputs.

## Dashboard (`dashboard/`, read-only, no trading)
FastAPI backend + React/TS/Vite + TradingView lightweight-charts. Three chart modes: **live** (own
public Deriv feed), **archive** (resampled from recorded ticks), **deep (history)** (fetches Deriv's
native candle history incl. **1d/4h** and runs the detector over it — display-only, excluded from
validation). Panels: chart (boxes + value lines + entry arrows), backtest verdict, **ATS funnel**
(contraction→breakout→pullback→entry and where gated), archive health.

## Multi-asset collection
`run_all.py` / `stop_all.py` launch one detached collector per symbol: synthetics **stpRNG**,
**1HZ50V** (RNG control) + real markets **frxUSDJPY, frxXAUUSD (Gold), OTC_NDX (NAS100)** (edge-possible
treatment). Real markets go STALE when their session closes (and now survive it).

## Tech stack
Python 3.13 · raw `websockets` · pandas + pyarrow (Parquet) · FastAPI + uvicorn · React + TypeScript
+ Vite + lightweight-charts · git/GitHub. No execution code by design.

## Status & honest findings
- Data spine: done, soak-verified gap-free. ATS detector + validation + dashboard: built, collecting 5
  symbols.
- So far: on synthetics every test returns **no edge / loses to random** (as theory demands for an RNG);
  one "winning" symbol was a 5-trade outlier the harness flags as noise. Real markets have too few
  trades yet (single digits) to judge; need weeks of session time. 1m trades are cost-dominated.
- Trading gate (locked, unmet): >500 reviewed entries AND edge surviving permutation p<0.05
  (family-wise) + OOS + low PBO + deflated-Sharpe hurdle on a real market. Likely outcome: nothing
  survives — which is success (the harness did its job).

## Bottom line
A disciplined, reproducible **edge-detection machine** that encodes the ATS method faithfully and then
tries hard to *falsify* it. Its value is honest measurement + risk discipline, not promised returns.


---

## Source code

### Engine core

#### `config.py`

```python
"""Configuration for the Deriv data-spine bot, loaded from environment / .env.

Phase 1 is data-only and demo-only. No trading parameters live here yet.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Repo root (this file's directory). data/ lives alongside the code.
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    # --- credentials / connection -------------------------------------------------
    # Phase 1 is data-only. On the legacy v3 API, market data (ticks/candles/active_symbols)
    # is PUBLIC — no authorize needed. We only authenticate for account calls (balance, buy,
    # portfolio) in later phases. So `authenticate` defaults to False and NO token is required
    # to run the spine.
    #
    # NOTE: Deriv migrated new accounts to "pat_"-prefixed Personal Access Tokens for their
    # NEW Options API (REST OTP -> authenticated WS URL). Those tokens do NOT work with this
    # legacy v3 {"authorize": token} call (they return InvalidToken). When we add trading
    # (Phase 3) we'll implement the OTP flow for the pat_ token; until then the token is unused.
    authenticate: bool = os.getenv("DERIV_AUTHENTICATE", "false").lower() == "true"
    api_token: str = os.getenv("DERIV_API_TOKEN", "")
    app_id: str = os.getenv("DERIV_APP_ID", "1089")
    endpoint: str = "wss://ws.derivws.com/websockets/v3"

    # --- market -------------------------------------------------------------------
    symbol: str = os.getenv("DERIV_SYMBOL", "stpRNG")  # Step Index

    # --- timeframes ---------------------------------------------------------------
    # We subscribe to ONE base candle stream (base_granularity, in seconds) plus the
    # raw tick stream. Higher timeframes are resampled in-process from the base frame.
    base_granularity: int = 60  # 1-minute base candles
    # Display/derived timeframes -> pandas resample rule. The base ("1m") resamples
    # to itself, which keeps the snapshot uniform.
    timeframes: dict[str, str] = field(
        default_factory=lambda: {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h"}
    )

    history_count: int = 5000          # base 1m candles to seed on connect (~83h) — enough depth for
                                       # the 1h rung of the ladder to warm + form pivots on cold start
    max_base_rows: int = 6000          # cap base frame to bound memory (> history_count, no instant trim)

    # --- ATS Master Pattern detector (ats_signals.py) — detect + LOG only, NO trading -------------
    # The ONLY methodology: TradeATS value-line + HTF→LTF pullback. Detector runs on candle CLOSE,
    # consumes MarketSnapshot.views only. See ats_signals.py.
    atr_period: int = 14               # Wilder ATR period (used by the contraction box / breakout buffer)
    signal_flush_every: int = 1        # write-through JSONL (tiny volume; survive ungraceful kills)
    # review_signals.py (offline outcome measurement)
    outcome_horizon_bars: int = 10     # forward window per signal, in bars of its timeframe
    outcome_move_points: float = 0.0   # first-touch barrier; 0 => use 0.5 * atr_at_signal
    # backtest_signals.py (contract-economics simulation)
    bt_payout_ratio: float = 0.95      # Rise/Fall WIN profit per unit stake. ASSUMPTION — real
                                       # payouts vary by symbol/duration (fetch via proposal API later).
    bt_stake: float = 1.0              # stake per simulated trade (unit stake)
    # bracket_backtest.py (structural ATS bracket: SL=expansion extreme, TP1=value line, TP2=box far
    # side; partial at TP1 then runner to TP2 with stop at break-even). P&L in R-multiples.
    bt_partial_frac: float = 0.5       # fraction banked at TP1 (the rest runs to TP2 / break-even)
    bt_stop_buffer_atr: float = 0.0    # stop placed this*ATR BEYOND the structural extreme (0 = at it)
    bt_cost_atr: float = 0.0           # round-trip cost (spread+slippage) per fill, in ATR units. 0 is
                                       # OPTIMISTIC — real markets have spread; raise to stress-test.
    bt_bracket_max_bars: int = 60      # time-stop: close at market after this many entry-TF bars
    # validate_signals.py (honest statistical validation — treat any edge as a null to disprove)
    n_permutations: int = 2000         # Monte-Carlo null draws for the permutation test
    walk_forward_oos_frac: float = 0.30  # fraction of (time-ordered) trades held out as out-of-sample
    cscv_blocks: int = 10              # S: time blocks for CSCV / Probability of Backtest Overfitting

    ats_signal_version: str = "ats_v3"  # v3 = value line is MEAN of contraction closes (transcript:
                                        # "average price established during the contraction"), not the
                                        # box midpoint (v2); + value_fade entry default. v1 = inside-bar.
    ats_htf: str = os.getenv("ATS_HTF", "5m")    # higher timeframe — sets directional bias. 5m (not
                                                 # 15m) for research THROUGHPUT: 15m so rarely forms a
                                                 # pivot contraction that entries almost never accumulate
                                                 # (1HZ50V: 26h -> 0 entries). 5m defines bias far more
                                                 # often so we reach a testable n. Set ATS_HTF=15m to
                                                 # restore the wider pairing. Does NOT manufacture edge —
                                                 # validate_signals still judges honestly.
    ats_ltf: str = os.getenv("ATS_LTF", "1m")    # lower timeframe — gives the entry
    # CONTRACTION = swing/pivot compression (LuxAlgo-audited): a confirmed pivot-high lower than the
    # prior pivot-high AND a pivot-low higher than the prior pivot-low, over a pivot lookback. This is
    # the canonical Forex Master Pattern definition — selective by nature (few, meaningful boxes).
    ats_pivot_lookback: int = 5                  # bars each side for ta.pivothigh/low (the "Contraction
                                                 # Detection Lookback"); larger = fewer, bigger coils
    # VALUE LINE = the "average price established during the contraction" (ATS training, 24:26).
    #   "contraction_mean" = mean of CLOSES over the contraction-box bars (faithful; default).
    #   "midpoint"         = (box_high+box_low)/2 (the v2 behaviour; range middle).
    ats_value_line_mode: str = os.getenv("ATS_VALUE_LINE_MODE", "contraction_mean")
    ats_breakout_buffer_atr: float = 0.0         # EXPANSION = plain box-boundary break (audited: no ATR
                                                 # buffer). >0 only to experiment with a tolerance.
    ats_pullback_tol_atr: float = 0.5            # ENTRY when LTF close returns within this*ATR of value
                                                 # (our own heuristic — no public ATS entry spec exists)
    # Entry variant (both are UNVERIFIED heuristics — no public ATS entry spec; tested via the harness):
    #   "continuation" = enter the WITH-HTF breakout's pullback to value (our original).
    #   "value_fade"   = enter the COUNTER-HTF expansion spike, in the HTF-bias direction (the
    #                    Google/TradeATS "buy the discount / sell the premium, snap back to value" rule).
    ats_entry_mode: str = os.getenv("ATS_ENTRY_MODE", "value_fade")
    ats_max_contraction_bars: int = 60           # abandon a contraction with no breakout within this
    ats_max_entry_bars: int = 20                 # abandon an expansion with no pullback within this
    validate_ats_pivot_lookbacks: tuple = (3, 5, 8, 13)  # PBO sweep — highest-leverage ATS param

    # --- storage ------------------------------------------------------------------
    data_dir: Path = ROOT / "data"
    tick_flush_every: int = 100        # flush the tick buffer to Parquet every N ticks

    # --- reconnection -------------------------------------------------------------
    reconnect_base_delay: float = 1.0  # seconds; exponential backoff start
    reconnect_max_delay: float = 30.0  # backoff ceiling
    market_closed_delay: float = 60.0  # wait between retries while a real market is closed (transient)
    # WS keepalive. Worst-case dead-socket detection ≈ ping_interval + ping_timeout, which is
    # also the upper bound on the live-tick gap before reconnect kicks in. Lower = smaller gaps.
    ping_interval: float = 10.0
    ping_timeout: float = 10.0
    # On reconnect, refetch the ticks missed during the outage so the archive stays gap-free.
    # Deriv caps ticks_history at ~5000 rows/request → covers gaps up to ~83 min at 1 tick/s.
    backfill_count: int = 5000

    @property
    def ws_url(self) -> str:
        return f"{self.endpoint}?app_id={self.app_id}"

    @property
    def tick_dir(self) -> Path:
        return self.data_dir / "ticks"

    @property
    def ats_signal_dir(self) -> Path:
        return self.data_dir / "signals_ats"

    @staticmethod
    def _tf_label_seconds(label: str) -> int:
        """Seconds for a timeframe LABEL like '1m'/'5m'/'15m'/'1h' (number + s/m/h/d unit)."""
        units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        num = int("".join(c for c in label if c.isdigit()))
        unit = "".join(c for c in label if c.isalpha()) or "m"
        return num * units[unit]

    @property
    def ats_ladder(self) -> tuple[str, ...]:
        """ATS timeframe ladder, ordered HIGH -> LOW (e.g. 15m, 5m, 1m). Each TF's entries are gated
        by the TF directly above it; the top TF is bias/context only. = all configured timeframes."""
        return tuple(sorted(self.timeframes, key=self._tf_label_seconds, reverse=True))

    @property
    def all_signal_timeframes(self) -> tuple[str, ...]:
        """The timeframes the store must build a TFView for — the full ATS ladder (all timeframes)."""
        return self.ats_ladder

    def view_params(self) -> dict:
        """Params the STORE needs to build TFViews (ATR + the swing-pivot contraction box)."""
        return {"atr_period": self.atr_period, "ats_pivot_lookback": self.ats_pivot_lookback}

    def ats_signal_params(self) -> dict:
        """Canonical ATS detector params — single source for AtsEngine and ats_params_hash."""
        return {
            "atr_period": self.atr_period,
            "ats_pivot_lookback": self.ats_pivot_lookback,
            "ats_breakout_buffer_atr": self.ats_breakout_buffer_atr,
            "ats_pullback_tol_atr": self.ats_pullback_tol_atr,
            "ats_max_contraction_bars": self.ats_max_contraction_bars,
            "ats_max_entry_bars": self.ats_max_entry_bars,
            "ats_entry_mode": self.ats_entry_mode,
            "ats_value_line_mode": self.ats_value_line_mode,
        }

    def ats_params_hash(self) -> str:
        import hashlib
        import json
        blob = json.dumps(self.ats_signal_params(), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]


CONFIG = Config()

```

#### `signals.py`

```python
"""Signal record schema — the one logged unit shared by the ATS detector and the review tools.

The ATS Master Pattern detector lives in ats_signals.py; this module only owns the SignalRecord
dataclass + schema version (kept separate so ats_signals.py stays free of any storage concern).

A record carries everything needed to recompute forward outcomes WITHOUT re-running the detector.
`bar_close_epoch` is the look-ahead firewall: the earliest instant a forward-outcome measurement
may inspect.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

SCHEMA_VERSION = 2  # v2 = ATS-only schema (dropped the legacy Bollinger band-width fields)


@dataclass(frozen=True)
class SignalRecord:
    schema_version: int
    signal_version: str
    detected_at_epoch: int
    symbol: str
    timeframe: str
    phase: str                      # "contraction" | "breakout" | "entry" | "entry_blocked"
    direction: str | None           # "up" | "down" | None
    bar_epoch: int                  # trigger bar open-time (dedup key with timeframe+phase)
    bar_close_epoch: int            # bar_epoch + tf_seconds — look-ahead firewall
    price_at_signal: float          # trigger bar close (outcome baseline)
    atr: float | None
    atr_at_contraction: float | None
    contraction_high: float | None  # the ATS contraction box high
    contraction_low: float | None   # the ATS contraction box low
    contraction_bars: int | None    # bars elapsed in the phase that led here
    episode_id: str                 # ties contraction → breakout → entry of one cycle
    params_hash: str
    # ATS value-line + entry context. value_line = midpoint of the frozen contraction box;
    # htf_bias = HTF side of its own value line at the LTF entry. The *_metadata fields are for
    # RESEARCH analysis only — any FILTER derived from them is a new config (must enter the PBO
    # sweep, never a post-hoc cherry-pick).
    value_line: float | None = None
    htf_bias: str | None = None
    dist_from_value_line: float | None = None
    bars_since_expansion: int | None = None
    htf_dist_from_value_line: float | None = None
    # Structural STOP reference for the ATS bracket = the Expansion-Phase swing extreme at entry
    # (the spike low for a long / spike high for a short). The bracket backtester places the stop
    # just beyond this. None on pre-bracket records (regenerate via backfill_signals to populate).
    stop_ref: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

```

#### `candles.py`

```python
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
    """Per-timeframe values for the ATS detector to consume — plain floats only, so the detector
    never touches pandas. Built from CLOSED bars (no forming bar). Indicator fields are None until
    enough closed bars exist (the warm-up gate, atr_period+1). See _compute_view."""
    tf: str
    closed_bar_epoch: int       # open-time epoch of the last CLOSED bar (the detection bar)
    close: float
    high: float
    low: float
    n_bars: int                 # closed bars available (warm-up gate)
    atr: float | None = None
    # Swing-pivot contraction (Forex Master Pattern, LuxAlgo-audited): contraction_now is True on the
    # bar where a new pivot confirms a lower-high + higher-low compression; box = the bounding pivots.
    contraction_now: bool = False
    box_high: float | None = None       # bounding swing pivot high of the contraction
    box_low: float | None = None        # bounding swing pivot low of the contraction
    box_close_mean: float | None = None  # mean CLOSE over the contraction-box bars ("average price
                                         # established during the contraction"); value-line candidate

    @property
    def ats_warm(self) -> bool:
        return self.atr is not None


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
    """Build a TFView (plain floats) from a CLOSED-bar OHLC frame for the ATS detector. Returns
    None if empty. Indicator fields stay None until atr_period+1 closed bars exist (warm-up gate).
    All pandas lives here so the detector downstream only ever sees floats."""
    if df is None or df.empty:
        return None
    close, high, low = df["close"], df["high"], df["low"]
    n = len(df)
    epoch = int(df.index[-1].timestamp())
    view = dict(tf=tf, closed_bar_epoch=epoch, close=float(close.iloc[-1]),
                high=float(high.iloc[-1]), low=float(low.iloc[-1]), n_bars=n)

    atr_period = int(p.get("atr_period", 14))
    length = int(p.get("ats_pivot_lookback", 5))

    # ATR (Wilder ~ EMA with alpha=1/period).
    if n >= atr_period + 1:
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
                       axis=1).max(axis=1)
        view["atr"] = float(tr.ewm(alpha=1 / atr_period, adjust=False).mean().iloc[-1])

    # Swing-pivot CONTRACTION (canonical FMP, LuxAlgo-audited): find pivot highs/lows with `length`
    # bars on each side (ta.pivothigh/ta.pivotlow). A contraction confirms on the bar where a NEW pivot
    # just completed AND the latest pivot-high is a LOWER high while the latest pivot-low is a HIGHER
    # low (range compressing from both sides). The box = those two bounding pivots; value line = its
    # midpoint (set downstream). A pivot at index i is only confirmed `length` bars later, at i+length.
    if n >= 2 * length + 1:
        H, L = high.to_numpy(), low.to_numpy()
        rmax = high.rolling(2 * length + 1, center=True).max().to_numpy()
        rmin = low.rolling(2 * length + 1, center=True).min().to_numpy()
        ph_idx = [i for i in range(length, n - length) if H[i] == rmax[i]]
        pl_idx = [i for i in range(length, n - length) if L[i] == rmin[i]]
        piv = n - 1 - length  # the bar whose pivot (if any) is confirmed at the current bar
        new_pivot = (piv in ph_idx) or (piv in pl_idx)
        if new_pivot and len(ph_idx) >= 2 and len(pl_idx) >= 2:
            lower_high = H[ph_idx[-1]] < H[ph_idx[-2]]
            higher_low = L[pl_idx[-1]] > L[pl_idx[-2]]
            if lower_high and higher_low:
                # "Average price established during the contraction" = mean of closes from the
                # earliest bounding pivot through the confirmation bar (the coil region).
                start = min(ph_idx[-1], pl_idx[-1])
                view.update(contraction_now=True,
                            box_high=float(H[ph_idx[-1]]), box_low=float(L[pl_idx[-1]]),
                            box_close_mean=float(close.iloc[start:].mean()))
    return TFView(**view)

```

#### `storage.py`

```python
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

```

#### `deriv_client.py`

```python
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

# Deriv error codes that are genuinely fatal (no point reconnecting). Everything else — notably
# MarketIsClosed (real markets close on weekends/sessions), RateLimit, etc. — is transient and retried.
FATAL_DERIV_CODES = {"InvalidToken", "AuthorizationRequired", "InvalidAppID", "InvalidAppMarkupPercentage"}


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
            except DerivError as e:
                if e.code in FATAL_DERIV_CODES:
                    raise  # auth/app errors are genuinely fatal — don't spin reconnecting
                # Transient (e.g. MarketIsClosed — real markets close on weekends/sessions and reopen).
                # Back off and retry instead of dying; wait longer when the market is simply closed.
                log.warning("Deriv error %s — retrying (transient)", e.code)
                if e.code == "MarketIsClosed":
                    delay = max(delay, self.cfg.market_closed_delay)
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

```

#### `ats_signals.py`

```python
"""ATS Master Pattern detector — value-line + HTF→LTF entry (signal_version from config, ats_v3).

A FAITHFUL, pure, side-effect-free encoding of the TradeATS / Forex-Master-Pattern method, distinct
from the Phase-2 breakout detector in signals.py. NO trading. NO I/O. NO pandas. Consumes only
`TFView` floats from MarketSnapshot (candles.py) and emits SignalRecords to a SEPARATE store.

The method, per the research:
  1. CONTRACTION  — a swing-pivot compression (lower-high AND higher-low). A *value line* is frozen
                    at the "average price during the contraction" (mean of contraction closes;
                    config ats_value_line_mode, "midpoint" optional) and projected forward.
  2. EXPANSION    — price breaks out of the box (close beyond box ± buffer*ATR).
  3. ENTRY        — on the LTF, price PULLS BACK to its value line, and we enter in the breakout
                    direction — but ONLY if it agrees with the HTF bias (HTF price vs HTF value line).

The HTF gate is the heart of ATS: you don't buy because price crossed a line, you buy because the
HTF is bullish AND the LTF pulled back to value. The engine wires HTF→LTF; each detector is per-TF.

Honest caveat: on a CSPRNG synthetic these structures are noise and the value line "remembers"
nothing — an edge is impossible by construction. The deliverable is the LOGGED entry + the
validation harness that adjudicates (permutation p / OOS / PBO) whether it had any forward edge.
"""
from __future__ import annotations

from dataclasses import replace
from enum import Enum

from signals import SCHEMA_VERSION, SignalRecord


class AtsPhase(str, Enum):
    NO_SIGNAL = "no_signal"
    CONTRACTION = "contraction"   # inside-bar coil detected; value line frozen
    EXPANSION = "expansion"       # broke out of the box; waiting for a pullback to value
    # terminal events emitted as records use phase strings: "contraction" | "breakout" | "entry"


# Phase strings written to records (the harness trades phase == "entry").
P_CONTRACTION = "contraction"
P_BREAKOUT = "breakout"
P_ENTRY = "entry"
P_ENTRY_BLOCKED = "entry_blocked"   # a pullback fired but the HTF gate rejected it — logged for the
                                    # funnel (NOT tradeable; backtest/validate filter phase == "entry")


class AtsTimeframeDetector:
    """ATS state machine for ONE timeframe. HTF-agnostic: it emits entry candidates; the engine
    applies the HTF-bias gate. Tracks a PERSISTENT value line (the current reference level, survives
    episode end until a new contraction overwrites it) for cross-timeframe bias."""

    def __init__(self, symbol: str, tf: str, tf_seconds: int, params: dict,
                 signal_version: str, params_hash: str):
        self.symbol = symbol
        self.tf = tf
        self.tf_seconds = tf_seconds
        self.p = params
        self.signal_version = signal_version
        self.params_hash = params_hash
        self.entry_mode = params.get("ats_entry_mode", "value_fade")
        self.value_line_mode = params.get("ats_value_line_mode", "contraction_mean")

        self.phase = AtsPhase.NO_SIGNAL
        self._last_epoch: int | None = None
        self.last_close: float | None = None
        # persistent reference (NOT cleared at episode end — value lines persist as S/R in ATS):
        self.value_line: float | None = None
        # frozen per episode:
        self.box_high: float | None = None
        self.box_low: float | None = None
        self.c_atr: float | None = None
        self.episode_id: str | None = None
        self.bars_in_contraction = 0
        # frozen at breakout:
        self.exp_dir: str | None = None
        self.bars_since_exp = 0
        # Expansion-phase swing extremes since the breakout (the structural STOP reference): a long's
        # stop sits below exp_low, a short's above exp_high.
        self.exp_high: float | None = None
        self.exp_low: float | None = None

    @property
    def bias(self) -> str | None:
        """HTF bias = which side of the (persistent) value line the latest close sits on."""
        if self.value_line is None or self.last_close is None:
            return None
        if self.last_close > self.value_line:
            return "up"
        if self.last_close < self.value_line:
            return "down"
        return None

    def on_closed_bar(self, view) -> list[SignalRecord]:
        if not view.ats_warm:
            return []
        if self._last_epoch is not None and view.closed_bar_epoch <= self._last_epoch:
            return []  # only act on a genuinely new closed bar (reconnects can re-feed)
        self._last_epoch = view.closed_bar_epoch
        self.last_close = view.close

        if self.phase is AtsPhase.NO_SIGNAL:
            # canonical FMP contraction: a swing-pivot lower-high + higher-low just confirmed
            if view.contraction_now and view.box_high is not None:
                return [self._enter_contraction(view)]
            return []

        if self.phase is AtsPhase.CONTRACTION:
            self.bars_in_contraction += 1
            buf = self.p["ats_breakout_buffer_atr"] * self.c_atr   # buf=0 (default) => plain break
            up = view.close > self.box_high + buf
            down = view.close < self.box_low - buf
            if up or down:
                break_dir = "up" if up else "down"
                breakout = self._enter_expansion(view, break_dir)
                if self.entry_mode == "value_fade":
                    # Fade the spike: enter AGAINST the break (= toward HTF value), at the spike price.
                    # Candidate dir = opposite of the break; the engine keeps it iff it matches HTF bias
                    # (i.e. the break was counter-trend), else logs entry_blocked. Entry now, no wait.
                    fade_dir = "down" if break_dir == "up" else "up"
                    return [breakout, self._emit_entry(view, fade_dir, 0)]
                return [breakout]
            if self.bars_in_contraction >= int(self.p["ats_max_contraction_bars"]):
                self._end_episode()
            return []

        # AtsPhase.EXPANSION (continuation mode) — wait for a pullback to value in the breakout dir.
        self.bars_since_exp += 1
        self.exp_high = max(self.exp_high, view.high)   # extend the expansion swing extremes
        self.exp_low = min(self.exp_low, view.low)
        tol = self.p["ats_pullback_tol_atr"] * self.c_atr
        pulled_back = (view.close <= self.value_line + tol) if self.exp_dir == "up" \
            else (view.close >= self.value_line - tol)
        if pulled_back:
            return [self._emit_entry(view, self.exp_dir, self.bars_since_exp)]
        if self.bars_since_exp >= int(self.p["ats_max_entry_bars"]):
            self._end_episode()
        return []

    # -- transitions -----------------------------------------------------------------
    def _enter_contraction(self, view) -> SignalRecord:
        self.phase = AtsPhase.CONTRACTION
        self.box_high = view.box_high
        self.box_low = view.box_low
        # Value line = "average price during the contraction" (mean of contraction closes) by
        # default; "midpoint" is the v2 fallback. Falls back to midpoint if the mean is unavailable.
        if self.value_line_mode == "contraction_mean" and view.box_close_mean is not None:
            self.value_line = view.box_close_mean
        else:
            self.value_line = (view.box_high + view.box_low) / 2.0
        # persistent reference line (survives episode end until a new contraction overwrites it)
        self.c_atr = view.atr
        self.episode_id = f"ats:{self.symbol}:{self.tf}:{view.closed_bar_epoch}"
        self.bars_in_contraction = 0
        return self._make(view, P_CONTRACTION, None, 0)

    def _enter_expansion(self, view, direction: str) -> SignalRecord:
        rec = self._make(view, P_BREAKOUT, direction, self.bars_in_contraction)
        self.phase = AtsPhase.EXPANSION
        self.exp_dir = direction
        self.bars_since_exp = 0
        self.exp_high = view.high   # seed the expansion swing extremes with the breakout bar
        self.exp_low = view.low
        return rec  # value_line + box + episode_id stay frozen through expansion

    def _emit_entry(self, view, direction: str, bars_since: int) -> SignalRecord:
        """Emit the (HTF-ungated) entry candidate in `direction`, then end the episode (one per
        episode). The engine applies the HTF-bias gate and stamps htf_bias/htf_dist."""
        # Structural stop reference = the expansion swing extreme on the side the trade risks:
        # a long (up) risks a deeper LOW; a short (down) risks a higher HIGH.
        stop_ref = self.exp_low if direction == "up" else self.exp_high
        rec = self._make(view, P_ENTRY, direction, bars_since,
                         dist=abs(view.close - self.value_line),
                         bars_since_expansion=bars_since, stop_ref=stop_ref)
        self._end_episode()
        return rec

    def _end_episode(self) -> None:
        # Keep value_line + last_close (persistent bias); reset only the episode machinery.
        self.phase = AtsPhase.NO_SIGNAL
        self.box_high = self.box_low = self.c_atr = self.episode_id = None
        self.exp_dir = None
        self.exp_high = self.exp_low = None
        self.bars_in_contraction = self.bars_since_exp = 0

    def _make(self, view, phase: str, direction: str | None, bars: int,
              dist: float | None = None, bars_since_expansion: int | None = None,
              stop_ref: float | None = None) -> SignalRecord:
        bar_close = view.closed_bar_epoch + self.tf_seconds
        return SignalRecord(
            schema_version=SCHEMA_VERSION,
            signal_version=self.signal_version,
            detected_at_epoch=bar_close,
            symbol=self.symbol,
            timeframe=self.tf,
            phase=phase,
            direction=direction,
            bar_epoch=view.closed_bar_epoch,
            bar_close_epoch=bar_close,
            price_at_signal=view.close,
            atr=view.atr,
            atr_at_contraction=self.c_atr,
            contraction_high=self.box_high,   # the ATS box = contraction range
            contraction_low=self.box_low,
            contraction_bars=bars,
            episode_id=self.episode_id or f"ats:{self.symbol}:{self.tf}:{view.closed_bar_epoch}",
            params_hash=self.params_hash,
            value_line=self.value_line,
            htf_bias=None,                    # engine fills for entry records
            dist_from_value_line=dist,
            bars_since_expansion=bars_since_expansion,
            htf_dist_from_value_line=None,    # engine fills for entry records
            stop_ref=stop_ref,                # structural stop reference (entry records only)
        )


class AtsEngine:
    """A timeframe LADDER: one detector per timeframe, ordered HIGH -> LOW (e.g. 15m, 5m, 1m). Each
    timeframe's entries are gated by the NEXT-HIGHER timeframe's bias (HTF close vs its value line),
    so every TF is analysed: the top TF is bias+context only (no higher TF to confirm it, so its
    entries are dropped — exactly the old HTF behaviour); every lower TF emits HTF-gated entries.
    A 2-element ladder [htf, ltf] reproduces the original single-pair engine. Context records
    (contraction/breakout) from ALL timeframes pass through. Dedups on (timeframe, bar_epoch, phase).
    """

    def __init__(self, symbol: str, params: dict, tf_ladder, tf_seconds: dict,
                 signal_version: str, params_hash: str):
        self.tfs = list(tf_ladder)                 # HIGH -> LOW
        self.dets = {tf: AtsTimeframeDetector(symbol, tf, int(tf_seconds[tf]), params,
                                              signal_version, params_hash)
                     for tf in self.tfs}
        # Convenience aliases (top = highest TF, bottom = lowest) for callers/tests/dashboard.
        self.htf = self.tfs[0] if self.tfs else None
        self.ltf = self.tfs[-1] if self.tfs else None
        self.htf_det = self.dets.get(self.htf) if self.tfs else None
        self.ltf_det = self.dets.get(self.ltf) if self.tfs else None
        self._seen: set[tuple[str, int, str]] = set()

    def on_snapshot(self, snap) -> list[SignalRecord]:
        out: list[SignalRecord] = []
        # Walk HIGH -> LOW so each higher TF's bias reflects the current bar before it gates the
        # timeframe below it. The TF directly above `tf` in the ladder is its bias/gate source.
        for i, tf in enumerate(self.tfs):
            view = snap.views.get(tf)
            if view is None:
                continue
            higher_det = self.dets[self.tfs[i - 1]] if i > 0 else None  # None for the top TF
            bias = higher_det.bias if higher_det is not None else None
            for rec in self.dets[tf].on_closed_bar(view):
                if rec.phase == P_ENTRY:
                    if higher_det is None:
                        continue  # top of ladder: no higher TF to confirm -> context only, drop entry
                    hv, hl = higher_det.value_line, higher_det.last_close
                    hdist = abs(hl - hv) if hv is not None and hl is not None else None
                    if bias is not None and rec.direction == bias:
                        rec = replace(rec, htf_bias=bias, htf_dist_from_value_line=hdist)
                    else:
                        # Higher-TF gate rejected it (no bias or counter-bias) — keep as a
                        # NON-tradeable blocked candidate so the funnel shows where entries drop out.
                        rec = replace(rec, phase=P_ENTRY_BLOCKED, htf_bias=(bias or "none"),
                                      htf_dist_from_value_line=hdist)
                self._add(rec, out)
        return out

    def _add(self, rec: SignalRecord, out: list[SignalRecord]) -> None:
        key = (rec.timeframe, rec.bar_epoch, rec.phase)
        if key in self._seen:
            return
        self._seen.add(key)
        out.append(rec)

```

#### `main.py`

```python
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
            cfg.symbol, cfg.ats_signal_params(), cfg.ats_ladder, tf_seconds,
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

```

### Multi-asset collection

#### `run_all.py`

```python
"""Launch each deriv-bot symbol as a DETACHED background process (Windows), logging to
logs/<symbol>.log. The processes survive closing the terminal. Stop with: python stop_all.py

Robust to the PowerShell `Start-Process` "Item has already been added: COMSPEC" bug — Python's
os.environ is case-insensitive, so it collapses the duplicate COMSPEC/ComSpec key.

IMPORTANT: run with NO foreground `python main.py` active for these symbols — two processes writing
the same data/<symbol>/ folder corrupt the archive. (This guard only catches prior run_all.py runs.)

Edit SYMBOLS to change which assets run.
"""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYMBOLS = [
    "stpRNG", "1HZ50V",                   # synthetics (CSPRNG, 24/7) — edge impossible by construction
    "frxUSDJPY", "frxXAUUSD", "OTC_NDX",  # REAL markets (USD/JPY, Gold/USD, US Tech 100) — edge POSSIBLE
]                                         # real markets have closing hours: weekend/overnight gaps are
                                          # EXPECTED, not failures (see CLAUDE.md). The control vs treatment
                                          # for "does the pattern thrive where order flow exists?"
LOGDIR = ROOT / "logs"
PIDFILE = LOGDIR / "pids.txt"
PY = ROOT / ".venv" / "Scripts" / "pythonw.exe"   # windowless: no console window pops up


def _alive(pid: int) -> bool:
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                       capture_output=True, text=True)
    return str(pid) in r.stdout


def main() -> None:
    LOGDIR.mkdir(exist_ok=True)
    if not PY.exists():
        raise SystemExit(f"venv python not found at {PY} - create/activate the venv first.")

    # Guard: refuse to start if a previous launch is still alive (prevents duplicate writers).
    if PIDFILE.exists():
        still = [ln for ln in PIDFILE.read_text().splitlines()
                 if "=" in ln and _alive(int(ln.split("=")[1]))]
        if still:
            raise SystemExit(f"Already running: {', '.join(still)}. Run: python stop_all.py")

    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    lines = []
    for sym in SYMBOLS:
        env = dict(os.environ)            # case-insensitive on Windows -> no COMSPEC dup
        env["DERIV_SYMBOL"] = sym
        logf = open(LOGDIR / f"{sym}.log", "a", encoding="utf-8")  # child inherits this handle
        p = subprocess.Popen([str(PY), "main.py"], cwd=str(ROOT), env=env,
                             stdout=logf, stderr=subprocess.STDOUT,
                             creationflags=flags, close_fds=True)
        lines.append(f"{sym}={p.pid}")
        print(f"started {sym} (pid {p.pid}) -> logs/{sym}.log")
    PIDFILE.write_text("\n".join(lines) + "\n")
    print("\nWatch a log (PowerShell):  Get-Content logs\\stpRNG.log -Wait -Tail 20")
    print("Stop all:                  python stop_all.py")


if __name__ == "__main__":
    main()

```

#### `stop_all.py`

```python
"""Stop all background deriv-bot instances started by run_all.py.

Hard-stop (Windows TerminateProcess), so up to ~99 buffered ticks and any buffered signals are
lost. But on the next run_all.py each instance BACKFILLS the gap from the tick archive, and
`python backfill_signals.py <symbol>` recovers any missed signals.
"""
import subprocess
from pathlib import Path

PIDFILE = Path(__file__).resolve().parent / "logs" / "pids.txt"


def main() -> None:
    if not PIDFILE.exists():
        print("No logs/pids.txt - nothing to stop.")
        return
    for ln in PIDFILE.read_text().splitlines():
        if "=" not in ln:
            continue
        sym, pid = ln.split("=", 1)
        # taskkill /T kills the whole tree (in case the venv launcher spawned a worker child).
        r = subprocess.run(["taskkill", "/PID", pid.strip(), "/T", "/F"],
                           capture_output=True, text=True)
        print(f"stopped {sym} (pid {pid})" if r.returncode == 0 else f"{sym} (pid {pid}): {r.stderr.strip() or r.stdout.strip()}")
    PIDFILE.unlink()
    print("Done. On next start, each instance backfills the downtime gap automatically.")


if __name__ == "__main__":
    main()

```

### Recovery & analysis

#### `backfill_signals.py`

```python
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

```

#### `check_archive.py`

```python
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

```

#### `review_signals.py`

```python
"""Phase 2 review tool — the actual research deliverable.

Measures whether logged signals had any FORWARD predictive value, by joining each signal against
the archived ticks (the record of record) and computing forward outcomes — then comparing the
result to a NULL MODEL (random entries on the same archive). Without the null comparison every
strategy looks smart; with it, "57% win" only matters if random isn't also ~57%.

Honest expectation: on a CSPRNG synthetic, expansion signals should be statistically
indistinguishable from random entries. That negative result is the deliverable — it proves the
patterns are noise BEFORE any money is risked.

Look-ahead firewall: outcomes use ONLY ticks with epoch STRICTLY GREATER than bar_close_epoch
(never the signal bar or the still-forming bar). This is the single guard against look-ahead bias.

Run:  python review_signals.py
      python review_signals.py --symbol stpRNG --tf 5m --phase expansion --horizon 10 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import CONFIG

GAP_S = 5            # tick spacing above this inside a window => incomplete (don't trust the outcome)
TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900}


def _load_signals(symbol: str, signal_dir=None) -> list[dict]:
    base = signal_dir if signal_dir is not None else CONFIG.ats_signal_dir
    out = []
    for f in sorted(glob.glob(str(base / symbol / "*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return out


def _load_ticks(symbol: str):
    files = sorted(glob.glob(str(CONFIG.tick_dir / symbol / "*.parquet")))
    if not files:
        return None, None
    df = (pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
          .drop_duplicates("epoch").sort_values("epoch").reset_index(drop=True))
    return df["epoch"].to_numpy(), df["quote"].to_numpy()


def _outcome(epochs, prices, baseline: float, start_excl: int, horizon_s: int,
             direction: str | None, barrier: float) -> dict | None:
    """Forward outcome over (start_excl, start_excl + horizon_s]. Returns None if the window is
    incomplete (not enough archived ticks / internal gap) so it can be excluded from aggregates."""
    hi = start_excl + horizon_s
    if epochs.size == 0 or hi > epochs[-1]:
        return None  # window hasn't fully elapsed in the archive yet
    i0 = int(np.searchsorted(epochs, start_excl, side="right"))  # strictly AFTER bar_close (firewall)
    i1 = int(np.searchsorted(epochs, hi, side="right"))
    w_ep, w_px = epochs[i0:i1], prices[i0:i1]
    if w_px.size == 0:
        return None
    # Continuity: reject windows with a gap at the edges or inside (outcome would be untrustworthy).
    if (w_ep[0] - start_excl) > GAP_S or (hi - w_ep[-1]) > GAP_S:
        return None
    if w_ep.size > 1 and int(np.max(np.diff(w_ep))) > GAP_S:
        return None

    last = float(w_px[-1])
    fwd = last - baseline                          # signed point move over the window
    mfe = float(np.max(w_px)) - baseline           # max favourable excursion (up)
    mae = float(np.min(w_px)) - baseline           # max adverse excursion (down)
    # Directional framing: for an "up" signal a positive fwd is a win; for "down", negative is.
    sign = 1.0 if direction == "up" else (-1.0 if direction == "down" else 0.0)
    dir_return = sign * fwd if sign else fwd
    win = dir_return > 0 if sign else None
    # First-touch barrier (±barrier from baseline), scanning ticks in order.
    hit = None
    if sign and barrier > 0:
        up_lvl, dn_lvl = baseline + barrier, baseline - barrier
        for px in w_px:
            if px >= up_lvl:
                hit = "up"; break
            if px <= dn_lvl:
                hit = "down"; break
        target = "up" if sign > 0 else "down"
        hit_target_first = (hit == target) if hit else None
    else:
        hit_target_first = None
    return dict(fwd_return=fwd, dir_return=dir_return, win=win, mfe=mfe, mae=mae,
                hit_target_first=hit_target_first, n_ticks=int(w_px.size))


def _agg(rows: list[dict], label: str) -> dict:
    n = len(rows)
    wins = [r["win"] for r in rows if r["win"] is not None]
    drets = [r["dir_return"] for r in rows]
    win_rate = (sum(wins) / len(wins)) if wins else float("nan")
    z = ((sum(wins) - 0.5 * len(wins)) / (0.5 * len(wins) ** 0.5)) if wins else float("nan")  # vs p=0.5
    return dict(label=label, n=n, win_rate=win_rate, z=z,
                mean_dir_return=float(np.mean(drets)) if drets else float("nan"),
                median_dir_return=float(np.median(drets)) if drets else float("nan"),
                mean_mfe=float(np.mean([r["mfe"] for r in rows])) if rows else float("nan"),
                mean_mae=float(np.mean([r["mae"] for r in rows])) if rows else float("nan"))


def _print_row(a: dict) -> None:
    print(f"  {a['label']:<26} n={a['n']:<5} win={a['win_rate']*100:5.1f}% (z={a['z']:+.2f})  "
          f"mean={a['mean_dir_return']:+.4f}  median={a['median_dir_return']:+.4f}  "
          f"MFE={a['mean_mfe']:+.4f} MAE={a['mean_mae']:+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=CONFIG.symbol)
    ap.add_argument("--tf", default=None, help="filter timeframe (e.g. 1m, 5m)")
    ap.add_argument("--phase", default=None, help="filter phase (contraction|breakout|entry)")
    ap.add_argument("--horizon", type=int, default=CONFIG.outcome_horizon_bars)
    ap.add_argument("--seed", type=int, default=42, help="null-model RNG seed (reproducible)")
    args = ap.parse_args()

    signals = _load_signals(args.symbol)
    if args.tf:
        signals = [s for s in signals if s.get("timeframe") == args.tf]
    if args.phase:
        signals = [s for s in signals if s.get("phase") == args.phase]
    if not signals:
        raise SystemExit(f"no signals found for {args.symbol} (tf={args.tf}, phase={args.phase}). "
                         f"Run main.py to collect some first.")
    epochs, prices = _load_ticks(args.symbol)
    if epochs is None:
        raise SystemExit(f"no tick archive under {CONFIG.tick_dir / args.symbol}")

    # Score every signal against the forward tick window.
    per_signal, groups, incomplete = [], {}, 0
    for s in signals:
        tf = s["timeframe"]
        hs = TF_SECONDS.get(tf, 60) * args.horizon
        atr = s.get("atr_at_contraction") or s.get("atr") or 0.0
        barrier = CONFIG.outcome_move_points if CONFIG.outcome_move_points > 0 else 0.5 * atr
        o = _outcome(epochs, prices, float(s["price_at_signal"]), int(s["bar_close_epoch"]),
                     hs, s.get("direction"), barrier)
        if o is None:
            incomplete += 1
            continue
        key = f"{tf}/{s['phase']}" + (f"/{s['direction']}" if s.get("direction") else "")
        groups.setdefault(key, []).append(o)
        per_signal.append({**{k: s.get(k) for k in
                              ("timeframe", "phase", "direction", "bar_epoch", "bar_close_epoch",
                               "price_at_signal", "value_line", "htf_bias")}, **o})

    # NULL MODEL: for each scored DIRECTIONAL signal, draw a random valid epoch with the same
    # direction + horizon, and compute the same outcome. If real ≈ null, there's no edge.
    rng = random.Random(args.seed)
    null_groups = {}
    valid_lo, valid_hi = epochs[0], epochs[-1]
    for s in signals:
        if not s.get("direction"):
            continue
        tf = s["timeframe"]; hs = TF_SECONDS.get(tf, 60) * args.horizon
        atr = s.get("atr_at_contraction") or s.get("atr") or 0.0
        barrier = CONFIG.outcome_move_points if CONFIG.outcome_move_points > 0 else 0.5 * atr
        o = None
        for _ in range(20):  # retry until we land a complete window
            t = rng.randint(int(valid_lo), int(valid_hi) - hs - 1)
            i = int(np.searchsorted(epochs, t, side="right"))
            if i >= epochs.size:
                continue
            base = float(prices[i])
            o = _outcome(epochs, prices, base, int(epochs[i]), hs, s["direction"], barrier)
            if o is not None:
                break
        if o is not None:
            key = f"{tf}/{s['phase']}/{s['direction']}"
            null_groups.setdefault(key, []).append(o)

    # Report.
    print(f"symbol: {args.symbol}   signals scored: {len(per_signal)}   "
          f"incomplete(excluded): {incomplete}   horizon: {args.horizon} bars")
    print(f"window firewall: ticks with epoch > bar_close_epoch only   gap reject: >{GAP_S}s")
    print("=" * 100)
    print("REAL SIGNALS")
    for key in sorted(groups):
        _print_row(_agg(groups[key], key))
    if null_groups:
        print("-" * 100)
        print("NULL MODEL (random entries, matched direction/horizon/count)")
        for key in sorted(null_groups):
            _print_row(_agg(null_groups[key], "random:" + key))
    print("=" * 100)

    # Verdict for directional groups: compare real win-rate to null, flag if within noise.
    for key in sorted(k for k in groups if k in null_groups):
        real, null = _agg(groups[key], key), _agg(null_groups[key], key)
        edge = (real["win_rate"] - null["win_rate"]) * 100
        verdict = ("NO EDGE (within ~noise of random)" if abs(real["z"]) < 2
                   else "investigate: real win-rate deviates from 50% — re-test on more data")
        print(f"  {key}: real {real['win_rate']*100:.1f}% vs random {null['win_rate']*100:.1f}% "
              f"(delta {edge:+.1f}pts) -> {verdict}")
    print("Reminder: on a CSPRNG synthetic, 'NO EDGE' is the expected, correct result.")

    # Per-signal CSV for spreadsheet review.
    if per_signal:
        date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        out = CONFIG.ats_signal_dir / args.symbol / f"_outcomes_{date}.csv"
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_signal[0].keys()))
            w.writeheader(); w.writerows(per_signal)
        print(f"per-signal outcomes -> {out}")


if __name__ == "__main__":
    main()

```

#### `backtest_signals.py`

```python
"""Contract-economics backtester (the honest centerpiece of Phase 2 review).

review_signals.py answers "did price move our way?". This answers the question that actually
matters: "would it have MADE MONEY?" — by replaying each directional (expansion) signal as a
simulated Deriv **Rise/Fall** contract against the archived ticks, applying the real payout
structure (a win returns +payout*stake, a loss returns -stake), and comparing the result to a
NULL MODEL of random entries.

Why this is decisive on a CSPRNG synthetic: a Rise/Fall contract paying 95% needs a win rate of
1/1.95 = 51.3% just to break even. On a fair random walk you get ~50%, so expectancy is NEGATIVE
by construction — the house edge. This tool quantifies that bleed in money terms, and shows the
real signals are statistically indistinguishable from random entries.

`run_backtest()` is the reusable core (used by the dashboard too); `main()` is the CLI wrapper.

Look-ahead firewall: a trade can only ENTER at the first tick STRICTLY AFTER bar_close_epoch.

Run:  python backtest_signals.py
      python backtest_signals.py --symbol 1HZ50V --payout 0.95 --duration-bars 10 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timezone

import numpy as np

from config import CONFIG
import review_signals as rv  # reuse _load_signals, _load_ticks, TF_SECONDS, GAP_S

MIN_TRADES_FOR_VERDICT = 200  # below this the win-rate is noise; verdict stays "insufficient data"


def simulate_trade(epochs, prices, signal_close_epoch: int, direction: str,
                   duration_s: int, payout: float, stake: float) -> dict | None:
    """Simulate one Rise/Fall contract. Enter at the first tick after signal_close_epoch, exit
    `duration_s` later. Returns the trade dict, or None if the window is incomplete (gap at
    entry/exit, or not enough archived ticks) so it can be excluded rather than scored wrong."""
    if epochs.size == 0:
        return None
    i_entry = int(np.searchsorted(epochs, signal_close_epoch, side="right"))  # strictly after = firewall
    if i_entry >= epochs.size:
        return None
    entry_epoch, entry_price = int(epochs[i_entry]), float(prices[i_entry])
    if entry_epoch - signal_close_epoch > rv.GAP_S:
        return None  # no prompt fill available (gap right after the signal)
    exit_target = entry_epoch + duration_s
    if exit_target > int(epochs[-1]):
        return None  # contract hasn't fully elapsed in the archive yet
    i_exit = int(np.searchsorted(epochs, exit_target, side="right")) - 1
    exit_epoch, exit_price = int(epochs[i_exit]), float(prices[i_exit])
    if exit_target - exit_epoch > rv.GAP_S:
        return None  # gap swallowing the expiry instant

    if direction == "up":
        win = exit_price > entry_price
    elif direction == "down":
        win = exit_price < entry_price
    else:
        return None
    tie = exit_price == entry_price
    pnl = 0.0 if tie else (payout * stake if win else -stake)
    return {"pnl": pnl, "win": (None if tie else win), "entry_price": entry_price,
            "exit_price": exit_price, "entry_epoch": entry_epoch, "exit_epoch": exit_epoch}


def aggregate(trades: list[dict], stake: float, label: str) -> dict:
    n = len(trades)
    decided = [t for t in trades if t["win"] is not None]
    wins = sum(1 for t in decided if t["win"])
    total_pnl = sum(t["pnl"] for t in trades)
    staked = n * stake
    return {
        "label": label, "n": n, "wins": wins, "decided": len(decided),
        "win_rate": (wins / len(decided)) if decided else float("nan"),
        "total_pnl": total_pnl,
        "pnl_per_trade": (total_pnl / n) if n else float("nan"),
        "roi_pct": (100 * total_pnl / staked) if staked else float("nan"),
    }


def run_backtest(symbol: str, tf: str | None = None, duration_bars: int | None = None,
                 payout: float | None = None, stake: float | None = None, seed: int = 42) -> dict:
    """Reusable core (also consumed by the dashboard). Replays each ATS pullback ENTRY as a
    simulated Rise/Fall contract vs the archived ticks and compares to a random null. `real`/`null`
    are aggregate dicts; `per_trade` is the per-signal detail (CLI writes it to CSV)."""
    duration_bars = duration_bars or CONFIG.outcome_horizon_bars
    payout = CONFIG.bt_payout_ratio if payout is None else payout
    stake = CONFIG.bt_stake if stake is None else stake
    breakeven = 1.0 / (1.0 + payout)

    all_sigs = rv._load_signals(symbol, signal_dir=CONFIG.ats_signal_dir)
    if tf:
        all_sigs = [s for s in all_sigs if s.get("timeframe") == tf]
    from collections import Counter
    pc = Counter(s.get("phase") for s in all_sigs)

    signals = [s for s in all_sigs
               if s.get("phase") == "entry" and s.get("direction") in ("up", "down")]
    epochs, prices = rv._load_ticks(symbol)
    if epochs is None or not signals:
        return {"symbol": symbol, "error": f"no tradeable ATS entry signals or tick archive for {symbol}",
                "real": None, "null": None, "breakeven": breakeven, "payout": payout,
                "duration_bars": duration_bars, "n_signals": len(signals), "trades": 0,
                "incomplete": 0, "per_trade": [], "phase_counts": dict(pc),
                "caveat": "Collect signals (main.py / backfill_signals.py) and ticks first."}

    rng = random.Random(seed)
    lo, hi = int(epochs[0]), int(epochs[-1])
    real, null, per_trade, incomplete = [], [], [], 0
    for s in signals:
        dur = rv.TF_SECONDS.get(s["timeframe"], 60) * duration_bars
        t = simulate_trade(epochs, prices, int(s["bar_close_epoch"]), s["direction"], dur, payout, stake)
        if t is None:
            incomplete += 1
            continue
        real.append(t)
        per_trade.append({**{k: s.get(k) for k in
                             ("timeframe", "direction", "bar_epoch", "bar_close_epoch")},
                          "duration_s": dur, **t})
        nt = None
        for _ in range(20):
            rt = rng.randint(lo, hi - dur - 1)
            nt = simulate_trade(epochs, prices, rt, s["direction"], dur, payout, stake)
            if nt is not None:
                break
        if nt is not None:
            null.append(nt)

    ra = aggregate(real, stake, "real")
    na = aggregate(null, stake, "null") if null else None
    # Honest verdict: null-aware AND power-aware. "win > break-even" alone is a fooled-by-randomness
    # trap — it ignores whether random entries did just as well, and ignores sample size.
    ntr, rw, rroi = ra["n"], ra["win_rate"], ra["roi_pct"]
    if ntr < MIN_TRADES_FOR_VERDICT:
        verdict = f"INSUFFICIENT DATA (n={ntr}, low power) - cannot tell edge from noise yet"
        verdict_class = "weak"
    elif na and (rw <= na["win_rate"] or rroi <= na["roi_pct"]):
        verdict = "NO EDGE - random entries match or beat the signals"
        verdict_class = "bad"
    elif rw > breakeven and (na is None or (rw > na["win_rate"] and rroi > na["roi_pct"])):
        verdict = "possible edge - confirm with validate_signals.py (permutation p, PBO)"
        verdict_class = "watch"
    else:
        verdict = "NO EDGE - loses to the house edge (win rate below break-even)"
        verdict_class = "bad"
    return {"symbol": symbol, "n_signals": len(signals), "trades": len(real), "incomplete": incomplete,
            "breakeven": breakeven, "payout": payout, "stake": stake, "duration_bars": duration_bars,
            "real": ra, "null": na, "verdict": verdict, "verdict_class": verdict_class, "per_trade": per_trade,
            "phase_counts": dict(pc),
            "caveat": "Payout is an ASSUMPTION (default 0.95). Even a fair 50% win rate loses to the "
                      "payout haircut - that is the house edge, and why no real-money trading is justified."}


def _print(a: dict) -> None:
    print(f"  {a['label']:<22} n={a['n']:<5} win={a['win_rate']*100:5.1f}%  "
          f"P&L={a['total_pnl']:+8.2f}  per-trade={a['pnl_per_trade']:+.4f}  ROI={a['roi_pct']:+.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=CONFIG.symbol)
    ap.add_argument("--tf", default=None, help="filter timeframe (e.g. 1m, 15m)")
    ap.add_argument("--duration-bars", type=int, default=CONFIG.outcome_horizon_bars)
    ap.add_argument("--payout", type=float, default=CONFIG.bt_payout_ratio)
    ap.add_argument("--stake", type=float, default=CONFIG.bt_stake)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    r = run_backtest(args.symbol, tf=args.tf, duration_bars=args.duration_bars,
                     payout=args.payout, stake=args.stake, seed=args.seed)
    if r.get("error"):
        raise SystemExit(r["error"])

    print(f"symbol: {r['symbol']}   tradeable signals: {r['n_signals']}   "
          f"trades simulated: {r['trades']}   incomplete(excluded): {r['incomplete']}")
    print(f"contract: Rise/Fall   duration: {r['duration_bars']} bars   payout: {r['payout']:.2f}   "
          f"stake: {r['stake']}   break-even win rate: {r['breakeven']*100:.1f}%")
    print("=" * 92)
    _print(r["real"])
    if r["null"]:
        _print(r["null"])
    print("=" * 92)
    print(f"VERDICT: real win rate {r['real']['win_rate']*100:.1f}% vs break-even "
          f"{r['breakeven']*100:.1f}% -> {r['verdict']}")
    print(f"Reminder: {r['caveat']}")

    if r["per_trade"]:
        date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        out = CONFIG.ats_signal_dir / args.symbol / f"_backtest_{date}.csv"
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(r["per_trade"][0].keys()))
            w.writeheader(); w.writerows(r["per_trade"])
        print(f"per-trade detail -> {out}")


if __name__ == "__main__":
    main()

```

#### `bracket_backtest.py`

```python
"""Structural ATS bracket backtester (CFD/Multiplier-style — NOT the Rise/Fall binary).

Encodes the TradeATS Master Pattern trade management literally:
  - ENTRY  : the logged ATS entry (value_fade = the overextended spike).
  - STOP   : just beyond the Expansion-Phase swing extreme (`stop_ref`) — structural invalidation.
  - TP1    : the value line (centre of the contraction box = Fair Market Value). Bank a partial,
             move the runner's stop to break-even.
  - TP2    : the opposite side of the contraction box (the structural trend target). Runner exits
             there, or at break-even if price snaps back first.
P&L is measured in R-multiples (R = entry-to-stop distance), walked tick-by-tick on the real tick
archive for honest first-touch fills. A NULL model re-runs the SAME bracket geometry at random entry
locations: if the ATS locations don't beat random, there is no location edge.

Honest notes: default cost = 0 (frictionless = optimistic; real spread/slippage only hurts a tight
structural stop). On a CSPRNG synthetic no edge is possible. Display/research only — NO trading.

Run:  python bracket_backtest.py --symbol stpRNG
      python bracket_backtest.py --symbol frxXAUUSD --cost-atr 0.05   # stress-test with friction
"""
from __future__ import annotations

import argparse
import random
import statistics

import numpy as np

from config import CONFIG
import review_signals as rv


def _bracket(sig: dict) -> dict | None:
    """Build the SL/TP1/TP2 price levels for one entry, or None if the geometry is degenerate."""
    d = sig.get("direction")
    E = sig.get("price_at_signal")
    sref = sig.get("stop_ref")
    tp1 = sig.get("value_line")
    atr = sig.get("atr") or 0.0
    box_hi, box_lo = sig.get("contraction_high"), sig.get("contraction_low")
    if d not in ("up", "down") or None in (E, sref, tp1, box_hi, box_lo):
        return None
    buf = CONFIG.bt_stop_buffer_atr * atr
    if d == "up":
        sl, tp2 = sref - buf, box_hi
        ok = sl < E < tp1 <= tp2
    else:
        sl, tp2 = sref + buf, box_lo
        ok = tp2 <= tp1 < E < sl
    if not ok:
        return None
    return {"dir": d, "E": float(E), "SL": float(sl), "TP1": float(tp1), "TP2": float(tp2),
            "R": abs(float(E) - float(sl))}


def walk(ep: np.ndarray, px: np.ndarray, start_epoch: int, b: dict, max_secs: int,
         cost: float, gap: int = 5) -> float | None:
    """Walk ticks after start_epoch; return realised R (partial@TP1 + runner@TP2/BE/time-stop).
    None if a >gap-second hole falls inside the trade window (fills can't be trusted)."""
    i0 = int(np.searchsorted(ep, start_epoch, side="right"))
    if i0 >= len(ep) or b["R"] <= 0:
        return None
    long = b["dir"] == "up"
    E, SL, TP1, TP2, R = b["E"], b["SL"], b["TP1"], b["TP2"], b["R"]
    end = start_epoch + max_secs
    realized, pos, stop, took = 0.0, 1.0, SL, False
    c = cost  # per-fill cost in price units (already ATR-scaled by caller); charge entry once
    realized -= c / R                                  # entry-side cost
    prev = ep[i0]
    for i in range(i0, len(ep)):
        e, p = int(ep[i]), float(px[i])
        if e - prev > gap:
            return None
        prev = e
        if e > end:                                    # time-stop: close runner at market
            realized += pos * ((p - E if long else E - p) - c) / R
            return realized
        if (p <= stop) if long else (p >= stop):       # stop (incl. break-even) checked first
            realized += pos * ((stop - E if long else E - stop) - c) / R
            return realized
        if not took:
            if (p >= TP1) if long else (p <= TP1):     # TP1 -> bank partial, runner stop to BE
                realized += CONFIG.bt_partial_frac * ((TP1 - E if long else E - TP1) - c) / R
                pos -= CONFIG.bt_partial_frac
                took, stop = True, E
        elif (p >= TP2) if long else (p <= TP2):       # TP2 -> runner target
            realized += pos * ((TP2 - E if long else E - TP2) - c) / R
            return realized
    realized += pos * ((float(px[-1]) - E if long else E - float(px[-1])) - c) / R
    return realized


def _stats(rs: list[float]) -> dict:
    if not rs:
        return {"n": 0, "win": float("nan"), "avg_r": float("nan"), "total_r": 0.0, "exp": float("nan")}
    wins = sum(1 for r in rs if r > 0)
    return {"n": len(rs), "win": wins / len(rs), "avg_r": statistics.fmean(rs),
            "total_r": sum(rs), "exp": statistics.fmean(rs)}


def run(symbol: str, tf: str | None, cost_atr: float, seed: int) -> dict:
    sigs = [s for s in rv._load_signals(symbol, signal_dir=CONFIG.ats_signal_dir)
            if s.get("phase") == "entry" and s.get("direction") in ("up", "down")
            and (tf is None or s.get("timeframe") == tf)]
    ep, px = rv._load_ticks(symbol)
    if ep is None:
        raise SystemExit(f"no tick archive for {symbol}")
    max_secs = CONFIG.bt_bracket_max_bars * 60  # entry TF granularity ~ minutes; 60s base unit
    rng = random.Random(seed)
    lo, hi = int(ep[0]), int(ep[-1])

    real, nul, skipped, incomplete = [], [], 0, 0
    for s in sigs:
        b = _bracket(s)
        if b is None:
            skipped += 1
            continue
        cost = cost_atr * (s.get("atr") or 0.0)
        r = walk(ep, px, int(s["bar_close_epoch"]), b, max_secs, cost)
        if r is None:
            incomplete += 1
            continue
        real.append(r)
        # NULL: same direction + same bracket distances, random entry location.
        d_sl, d1, d2 = b["E"] - b["SL"], b["TP1"] - b["E"], b["TP2"] - b["E"]
        rr = None
        for _ in range(10):
            e_r = rng.randint(lo, hi - max_secs - 1)
            j = int(np.searchsorted(ep, e_r, side="right"))
            if j >= len(px):
                continue
            E_r = float(px[j])
            nb = {"dir": b["dir"], "E": E_r, "SL": E_r - d_sl, "TP1": E_r + d1,
                  "TP2": E_r + d2, "R": b["R"]}
            rr = walk(ep, px, e_r, nb, max_secs, cost)
            if rr is not None:
                break
        if rr is not None:
            nul.append(rr)
    return {"symbol": symbol, "tf": tf or "all", "real": _stats(real), "null": _stats(nul),
            "skipped": skipped, "incomplete": incomplete, "cost_atr": cost_atr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=CONFIG.symbol)
    ap.add_argument("--tf", default=None, help="filter entry timeframe (e.g. 1m)")
    ap.add_argument("--cost-atr", type=float, default=CONFIG.bt_cost_atr,
                    help="round-trip cost per fill in ATR units (0 = frictionless, optimistic)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    r = run(args.symbol, args.tf, args.cost_atr, args.seed)
    re, nu = r["real"], r["null"]
    print(f"symbol: {r['symbol']}   tf: {r['tf']}   structural ATS bracket (R-multiples)")
    print(f"  SL=expansion extreme  TP1=value line (partial {CONFIG.bt_partial_frac:.0%}, then BE)  "
          f"TP2=box far side   cost={args.cost_atr} ATR/fill   max {CONFIG.bt_bracket_max_bars} bars")
    print(f"  skipped (degenerate geometry): {r['skipped']}   incomplete (tick gap): {r['incomplete']}")
    print("=" * 92)
    def row(name, s):
        if not s["n"]:
            print(f"  {name:8s} n=0  (none)"); return
        print(f"  {name:8s} n={s['n']:<4d} win={s['win']*100:5.1f}%  avg={s['avg_r']:+.3f}R  "
              f"total={s['total_r']:+.2f}R  expectancy={s['exp']:+.3f}R")
    row("real", re); row("null", nu)
    print("=" * 92)
    if re["n"] < 30:
        print(f"  !! LOW POWER (n={re['n']}): expectancy is noise until ~hundreds of trades.")
    if not (np.isnan(re["exp"]) or np.isnan(nu["exp"])):
        edge = re["exp"] - nu["exp"]
        print(f"  real vs null expectancy: {edge:+.3f}R  -> "
              f"{'beats random (verify with validate_signals + more n)' if edge > 0.05 else 'NO structural edge (within noise of random)'}")
    print("  Reminder: cost=0 is optimistic; a tight structural stop is spread-sensitive. On a CSPRNG "
          "synthetic, no edge is possible by construction. Research only — NO trading.")


if __name__ == "__main__":
    main()

```

#### `validate_signals.py`

```python
"""Honest statistical validation harness (Phase 2).

The deep-research verdict (Bailey/Borwein/Lopez de Prado & Zhu): the fix isn't a better pattern —
it's rigorous testing, because once you try more than one configuration, backtest overfitting is
*always* present and false positives become almost certain. So this tool treats any signal "edge"
as a NULL HYPOTHESIS to disprove, with four methods:

  1. Monte-Carlo permutation test  -> p-value that the real edge beats random entries
  2. Walk-forward / out-of-sample   -> does an in-sample edge survive on held-out later data?
  3. PBO via CSCV                    -> Probability of Backtest Overfitting across a param sweep
  4. Deflated (expected-max) Sharpe  -> is the best config's Sharpe beyond what N trials yield by luck

On a CSPRNG synthetic the honest expected reading is: p ~ uniform (~0.5), PBO ~ 0.5, no OOS survival.
NO trading. Pure measurement. The pure-math helpers (perm_pvalue, cscv_pbo, expected_max_sharpe) are
unit-tested in verify_validation.py.

Run:  python validate_signals.py --symbol stpRNG
"""
from __future__ import annotations

import argparse
import itertools
import math
import random
import statistics
from collections import Counter

import numpy as np
import pandas as pd

from config import CONFIG
import review_signals as rv
import backtest_signals as bt

EULER_GAMMA = 0.5772156649


# ----------------------------------------------------------------------------- pure stat helpers
def perm_pvalue(observed: float, null_samples: list[float]) -> float:
    """One-sided p-value: P(null >= observed), with +1 smoothing (never reports 0)."""
    n = len(null_samples)
    if n == 0:
        return float("nan")
    ge = sum(1 for x in null_samples if x >= observed)
    return (1 + ge) / (1 + n)


def cscv_pbo(M: np.ndarray) -> float:
    """Probability of Backtest Overfitting via Combinatorially-Symmetric Cross-Validation.
    M is a (blocks x configs) performance matrix. For every split of the blocks into equal IS/OOS
    halves, take the IS-best config and find its OOS rank; PBO = fraction of splits where the
    IS-best lands at/below the OOS median (logit lambda <= 0). ~0.5 => indistinguishable from luck."""
    S, N = M.shape
    if S < 2 or N < 2 or S % 2 != 0:
        return float("nan")
    blocks = range(S)
    half = S // 2
    lambdas = []
    for IS in itertools.combinations(blocks, half):
        ISset = set(IS)
        OOS = [b for b in blocks if b not in ISset]
        is_perf = M[list(IS)].sum(axis=0)
        oos_perf = M[OOS].sum(axis=0)
        cstar = int(np.argmax(is_perf))                      # best in-sample config
        order = np.argsort(oos_perf, kind="stable")          # ascending: worst..best
        rank = int(np.where(order == cstar)[0][0]) + 1       # 1=worst .. N=best (OOS)
        omega = min(max(rank / (N + 1), 1e-9), 1 - 1e-9)
        lambdas.append(math.log(omega / (1 - omega)))
    return sum(1 for x in lambdas if x <= 0) / len(lambdas)


def expected_max_sharpe(sharpes: list[float]) -> float:
    """Expected MAX Sharpe under the null (no skill), given N independent trials with the observed
    cross-trial Sharpe variance (Bailey/Lopez de Prado). A real best-Sharpe must clear this hurdle."""
    n = len(sharpes)
    if n < 2:
        return float("nan")
    v = statistics.pvariance(sharpes)
    if v <= 0:
        return 0.0
    nd = statistics.NormalDist()
    z1 = nd.inv_cdf(1 - 1.0 / n)
    z2 = nd.inv_cdf(1 - 1.0 / (n * math.e))
    return math.sqrt(v) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return float("nan")
    sd = statistics.pstdev(returns)
    return statistics.fmean(returns) / sd if sd > 0 else 0.0


# ----------------------------------------------------------------------------- data helpers
def _dir_return(t: dict, direction: str) -> float:
    return (t["exit_price"] - t["entry_price"]) if direction == "up" else (t["entry_price"] - t["exit_price"])


def _real_trades(symbol, duration_bars, payout, stake, tf=None):
    """Outcomes of the ACTUALLY-LOGGED ATS pullback entries (entry/exit via simulate_trade)."""
    sigs = [s for s in rv._load_signals(symbol, signal_dir=CONFIG.ats_signal_dir)
            if s.get("phase") == "entry" and s.get("direction") in ("up", "down")]
    if tf:
        sigs = [s for s in sigs if s.get("timeframe") == tf]
    ep, px = rv._load_ticks(symbol)
    if ep is None:
        return [], None, None
    out = []
    for s in sigs:
        dur = rv.TF_SECONDS.get(s["timeframe"], 60) * duration_bars
        t = bt.simulate_trade(ep, px, int(s["bar_close_epoch"]), s["direction"], dur, payout, stake)
        if t:
            out.append({"epoch": t["entry_epoch"], "dir": s["direction"], "dur": dur,
                        "dir_ret": _dir_return(t, s["direction"]), "pnl": t["pnl"], "win": t["win"]})
    return out, ep, px


def _ats_pnl_series(symbol, ep, px, candles, pivot_lookback, buffer, duration_bars, payout, stake):
    """Per-config replay for the ATS PBO sweep: rebuild the store + HTF-gated AtsEngine with a given
    ats_pivot_lookback × breakout buffer, replay the 1m candles, and backtest each pullback ENTRY.
    Not vectorized (ATS entries are sparse and the engine is cheap), but it IS the real detector."""
    from ats_signals import AtsEngine
    from candles import MultiTimeframeStore
    p = dict(CONFIG.ats_signal_params())
    p["ats_pivot_lookback"] = pivot_lookback
    p["ats_breakout_buffer_atr"] = buffer
    vp = dict(CONFIG.view_params()); vp["ats_pivot_lookback"] = pivot_lookback
    store = MultiTimeframeStore(symbol, CONFIG.timeframes, base_granularity=CONFIG.base_granularity,
                                signal_timeframes=CONFIG.all_signal_timeframes, signal_params=vp)
    tf_seconds = {tf: int(pd.Timedelta(CONFIG.timeframes[tf]).total_seconds())
                  for tf in CONFIG.all_signal_timeframes}
    engine = AtsEngine(symbol, p, CONFIG.ats_ladder, tf_seconds, "validate", "sweep")
    dur = duration_bars * 60  # ATS entries are on the 1m LTF
    series, prev = [], None
    for c in candles:
        _, is_new = store.upsert(c)
        if is_new and prev is not None:
            for rec in engine.on_snapshot(store.snapshot(c["close"], c["open_time"])):
                if rec.phase == "entry" and rec.direction in ("up", "down"):
                    t = bt.simulate_trade(ep, px, rec.bar_close_epoch, rec.direction, dur, payout, stake)
                    if t:
                        series.append((t["entry_epoch"], t["pnl"]))
        prev = c
    return series


def _winrate(rows):
    d = [r for r in rows if r["win"] is not None]
    return (sum(1 for r in d if r["win"]) / len(d)) if d else float("nan")


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=CONFIG.symbol)
    ap.add_argument("--tf", default=None)
    ap.add_argument("--duration-bars", type=int, default=CONFIG.outcome_horizon_bars)
    ap.add_argument("--payout", type=float, default=CONFIG.bt_payout_ratio)
    ap.add_argument("--stake", type=float, default=CONFIG.bt_stake)
    ap.add_argument("--n-perm", type=int, default=CONFIG.n_permutations)
    ap.add_argument("--family-size", type=int, default=5,
                    help="number of markets tested as a family (for Bonferroni-adjusted significance)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true",
                    help="skip the PBO/CSCV + deflated-Sharpe param sweep (parts 3-4). That sweep "
                         "replays the whole tick archive per config (O(candles^2)) and is only "
                         "meaningful with hundreds of trades; --quick runs permutation + walk-forward.")
    args = ap.parse_args()

    real, ep, px = _real_trades(args.symbol, args.duration_bars, args.payout, args.stake, args.tf)
    if ep is None:
        raise SystemExit(f"no tick archive for {args.symbol}")
    n = len(real)
    label = "ATS pullback-entry"
    print(f"symbol: {args.symbol}   method: {label}   trades: {n}   "
          f"duration: {args.duration_bars} bars   payout: {args.payout}")
    if n < 200:
        print(f"  !! LOW STATISTICAL POWER: n={n} trades. Permutation/CSCV/Sharpe tests need a few "
              f"hundred to be stable — treat every number below as INDICATIVE ONLY. The METHOD is what's "
              f"validated now; verdicts firm up as signals accumulate toward the 500-signal gate.")
    if n == 0:
        raise SystemExit(f"no completed {label} trades to validate yet — accumulate more data "
                         f"(ATS is selective; entries require an aligned HTF bias + a pullback to value).")
    print("=" * 90)

    # 1) Permutation test on mean directional price-return (payout-independent edge test)
    obs = statistics.fmean(r["dir_ret"] for r in real)
    rng = random.Random(args.seed)
    lo, hi = int(ep[0]), int(ep[-1])
    specs = [(r["dir"], r["dur"]) for r in real]
    null = []
    for _ in range(args.n_perm):
        vals = []
        for d, dur in specs:
            t = None
            for _ in range(10):
                t = bt.simulate_trade(ep, px, rng.randint(lo, hi - dur - 1), d, dur, args.payout, args.stake)
                if t:
                    break
            if t:
                vals.append(_dir_return(t, d))
        if vals:
            null.append(statistics.fmean(vals))
    p = perm_pvalue(obs, null)
    print(f"1) PERMUTATION TEST  mean dir-return obs={obs:+.5f}  null mean={statistics.fmean(null):+.5f}  "
          f"p-value={p:.3f}")
    print(f"   {'edge beyond random (p<0.05)' if p < 0.05 else 'NOT distinguishable from random entries'}")
    # Family-wise honesty: testing several markets makes one spurious p<0.05 ~ a coin-flip. A lone
    # uncorrected hit is NOISE; it only counts if it clears the Bonferroni-adjusted threshold.
    if args.family_size > 1:
        alpha_fw = 0.05 / args.family_size
        print(f"   family-wise (Bonferroni, {args.family_size} markets): need p<{alpha_fw:.3f}  -> "
              f"{'clears it' if p < alpha_fw else 'does NOT clear — treat as noise across the family'}")

    # 2) Walk-forward / out-of-sample
    rows = sorted(real, key=lambda r: r["epoch"])
    cut = int(len(rows) * (1 - CONFIG.walk_forward_oos_frac))
    IS, OOS = rows[:cut], rows[cut:]
    print(f"2) WALK-FORWARD  IS n={len(IS)} win={_winrate(IS)*100:.1f}% pnl={sum(r['pnl'] for r in IS):+.2f} | "
          f"OOS n={len(OOS)} win={_winrate(OOS)*100:.1f}% pnl={sum(r['pnl'] for r in OOS):+.2f}")
    print(f"   {'OOS edge survived' if _winrate(OOS) > 0.5 and sum(r['pnl'] for r in OOS) > 0 else 'no OOS edge (in-sample result did not carry over)'}")

    # 3) PBO via CSCV over the ATS param sweep  +  4) deflated (expected-max) Sharpe.
    # Sweeps the highest-leverage ATS param, ats_pivot_lookback, × the breakout buffer, replaying the
    # real HTF-gated engine per config.
    if args.quick:
        print("3-4) PBO/CSCV + deflated Sharpe: SKIPPED (--quick). The sweep replays the full archive "
              "per config (O(candles^2) x configs x ladder TFs) and is only meaningful at a few hundred "
              "trades — run the full validation as the sample approaches the 500-signal gate.")
        print("=" * 90)
        print("VERDICT (ATS, quick): permutation + walk-forward only. An edge still counts ONLY if it "
              "clears the family-wise p-threshold AND survives OOS AND (full run) has low PBO AND real n.")
        return
    from backfill_signals import build_candles
    candles = build_candles(ep, px)
    S = CONFIG.cscv_blocks
    span = max(1, hi - lo)
    sharpes = []
    grid = [(pl, buf) for pl in CONFIG.validate_ats_pivot_lookbacks for buf in (0.0, 0.25)]
    M = np.zeros((S, len(grid)))
    for ci, (pl, buf) in enumerate(grid):
        series = _ats_pnl_series(args.symbol, ep, px, candles, pl, buf,
                                 args.duration_bars, args.payout, args.stake)
        sharpes.append(sharpe([pnl for _, pnl in series]))
        for e, pnl in series:
            M[min(S - 1, int((e - lo) / span * S)), ci] += pnl
    nonempty_blocks = int((M.sum(axis=1) != 0).sum())
    print(f"3) PBO / CSCV  configs={len(grid)}  blocks={S} (non-empty {nonempty_blocks})")
    if nonempty_blocks < S:
        print(f"   insufficient data: only {nonempty_blocks}/{S} time blocks contain trades — PBO needs the "
              f"archive to span all blocks. Re-run once more data has accumulated.")
    else:
        pbo = cscv_pbo(M)
        print(f"   PBO = {pbo:.2f}   ({'overfit / no real edge (PBO ~ 0.5+)' if pbo >= 0.4 else 'low PBO — best config may generalize; verify with more data'})")
    valid_sh = [s for s in sharpes if not math.isnan(s)]
    if len(valid_sh) >= 2:
        best, hurdle = max(valid_sh), expected_max_sharpe(valid_sh)
        print(f"4) DEFLATED SHARPE  best config Sharpe={best:+.3f}  expected-max under null={hurdle:+.3f}  "
              f"({'clears the luck hurdle' if best > hurdle else 'within what ' + str(len(grid)) + ' trials produce by chance'})")
    print("=" * 90)
    print("VERDICT (ATS): an edge counts ONLY if it clears the family-wise p-threshold AND survives "
          "OOS AND has low PBO AND real n. On synthetics expect no edge (CSPRNG control). A lone real "
          "market hit on small n is NOISE until it survives all four — that is the honest bar.")


if __name__ == "__main__":
    main()

```

### Tests (known-truth)

#### `verify_feed.py`

```python
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

```

#### `verify_ats.py`

```python
"""ATS Master Pattern detector regression — offline, deterministic, PASS/FAIL (like verify_feed.py).

Feeds hand-built TFViews through ats_signals.py to prove the value-line + HTF→LTF pullback logic:
contraction (swing-pivot LH/HL, signalled by view.contraction_now + box), value-line midpoint, plain
box breakout, pullback entry, the HTF-bias gate, warm-up gate, timeout, and engine dedup. The pivot
DETECTION itself (in candles._compute_view) is covered separately by t_pivot_*. No network.

Run:  python verify_ats.py
"""
from __future__ import annotations

import logging

import pandas as pd

from ats_signals import AtsEngine, AtsPhase, AtsTimeframeDetector
from candles import TFView, _compute_view

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("verify_ats")

# Base params pin the modes these legacy tests were written for (midpoint value line + continuation
# entry), so they stay valid regardless of the config DEFAULTS (now contraction_mean + value_fade).
# The new defaults get their own dedicated tests below.
P = {
    "atr_period": 14, "ats_pivot_lookback": 5, "ats_breakout_buffer_atr": 0.0,
    "ats_pullback_tol_atr": 0.5, "ats_max_contraction_bars": 3, "ats_max_entry_bars": 3,
    "ats_entry_mode": "continuation", "ats_value_line_mode": "midpoint",
}


def view(epoch, close, *, contraction=False, box_high=101.0, box_low=99.0, atr=1.0, warm=True,
         box_close_mean=None):
    """A warm (or non-warm) ATS TFView. The detector reads close, contraction_now, box_high/low,
    box_close_mean, atr."""
    return TFView(
        tf="1m", closed_bar_epoch=epoch, close=close, high=close + 1, low=close - 1, n_bars=200,
        atr=atr if warm else None,
        contraction_now=contraction,
        box_high=box_high if contraction else None,
        box_low=box_low if contraction else None,
        box_close_mean=box_close_mean if contraction else None,
    )


def det(tf="1m", secs=60) -> AtsTimeframeDetector:
    return AtsTimeframeDetector("TEST", tf, secs, P, "ats_test", "hash")


class Snap:
    def __init__(self, views):
        self.views = views


def engine() -> AtsEngine:
    return AtsEngine("TEST", P, ["15m", "1m"], {"15m": 900, "1m": 60}, "ats_test", "hash")


# -- single-timeframe state machine --------------------------------------------------
def t_warmup_gate() -> bool:
    return det().on_closed_bar(view(60, 100.0, contraction=True, warm=False)) == []


def t_contraction_and_value_line() -> bool:
    d = det()
    assert d.on_closed_bar(view(60, 100.0, contraction=False)) == []     # no pivot compression yet
    out = d.on_closed_bar(view(120, 100.0, contraction=True, box_high=101.0, box_low=99.0))
    return (len(out) == 1 and out[0].phase == "contraction" and out[0].direction is None
            and out[0].value_line == 100.0)


def t_value_line_offset_box() -> bool:
    out = det().on_closed_bar(view(120, 102.0, contraction=True, box_high=104.0, box_low=100.0))
    return len(out) == 1 and out[0].value_line == 102.0


def t_breakout_up_plain() -> bool:
    d = det()
    d.on_closed_bar(view(120, 100.0, contraction=True, box_high=101.0, box_low=99.0))
    out = d.on_closed_bar(view(180, 101.01, contraction=False))   # plain break: >101 (no ATR buffer)
    return len(out) == 1 and out[0].phase == "breakout" and out[0].direction == "up"


def t_no_break_inside_box() -> bool:
    d = det()
    d.on_closed_bar(view(120, 100.0, contraction=True, box_high=101.0, box_low=99.0))
    out = d.on_closed_bar(view(180, 101.0, contraction=False))    # == box_high -> not a break
    return out == [] and d.phase is AtsPhase.CONTRACTION


def t_breakout_down_plain() -> bool:
    d = det()
    d.on_closed_bar(view(120, 100.0, contraction=True, box_high=101.0, box_low=99.0))
    out = d.on_closed_bar(view(180, 98.99, contraction=False))    # plain break: <99
    return len(out) == 1 and out[0].phase == "breakout" and out[0].direction == "down"


def t_entry_pullback_up() -> bool:
    d = det()
    d.on_closed_bar(view(120, 100.0, contraction=True, box_high=101.0, box_low=99.0))  # value 100
    d.on_closed_bar(view(180, 102.0, contraction=False))          # breakout up
    out = d.on_closed_bar(view(240, 99.9, contraction=False))     # pull back to <= 100+0.5 -> entry up
    return (len(out) == 1 and out[0].phase == "entry" and out[0].direction == "up"
            and out[0].bars_since_expansion == 1
            and abs(out[0].dist_from_value_line - 0.1) < 1e-9
            and d.phase is AtsPhase.NO_SIGNAL)


def t_entry_timeout() -> bool:
    d = det()  # ats_max_entry_bars = 3
    d.on_closed_bar(view(120, 100.0, contraction=True, box_high=101.0, box_low=99.0))
    d.on_closed_bar(view(180, 102.0, contraction=False))          # breakout up
    outs = [d.on_closed_bar(view(180 + 60 * i, 103.0, contraction=False)) for i in range(1, 4)]
    return all(o == [] for o in outs) and d.phase is AtsPhase.NO_SIGNAL


def t_refeed_dedup() -> bool:
    d = det()
    r1 = d.on_closed_bar(view(120, 100.0, contraction=True))
    r2 = d.on_closed_bar(view(120, 100.0, contraction=True))      # same epoch re-fed
    return len(r1) == 1 and r2 == []


# -- value-line mode (contraction_mean, new default) ---------------------------------
def t_value_line_contraction_mean() -> bool:
    # contraction_mean uses the mean of contraction closes, NOT the box midpoint (102 here).
    p = {**P, "ats_value_line_mode": "contraction_mean"}
    d = AtsTimeframeDetector("TEST", "1m", 60, p, "ats_test", "hash")
    out = d.on_closed_bar(view(120, 100.0, contraction=True, box_high=104.0, box_low=100.0,
                               box_close_mean=101.3))
    return len(out) == 1 and abs(out[0].value_line - 101.3) < 1e-9


def t_value_line_mean_fallback_midpoint() -> bool:
    # contraction_mean mode but mean unavailable -> safe fallback to midpoint (102).
    p = {**P, "ats_value_line_mode": "contraction_mean"}
    d = AtsTimeframeDetector("TEST", "1m", 60, p, "ats_test", "hash")
    out = d.on_closed_bar(view(120, 100.0, contraction=True, box_high=104.0, box_low=100.0))
    return len(out) == 1 and out[0].value_line == 102.0


# -- entry mode (value_fade, new default) --------------------------------------------
def t_value_fade_enters_against_break() -> bool:
    # value_fade fires immediately on breakout, in the OPPOSITE direction (fade the spike), and
    # ends the episode (no EXPANSION wait). Up-break -> a "down" entry candidate.
    p = {**P, "ats_entry_mode": "value_fade"}
    d = AtsTimeframeDetector("TEST", "1m", 60, p, "ats_test", "hash")
    d.on_closed_bar(view(120, 100.0, contraction=True, box_high=101.0, box_low=99.0))
    out = d.on_closed_bar(view(180, 101.01, contraction=False))   # plain up-break
    phases = {(r.phase, r.direction) for r in out}
    return (("breakout", "up") in phases and ("entry", "down") in phases
            and d.phase is AtsPhase.NO_SIGNAL)


# -- engine: HTF-bias gate -----------------------------------------------------------
def _set_htf_bias_up(eng: AtsEngine) -> None:
    # HTF contraction sets value_line=99 (midpoint 98..100); bar close 99.5 > 99 -> bias "up".
    eng.on_snapshot(Snap({"15m": view(900, 99.5, contraction=True, box_high=100.0, box_low=98.0)}))


def _ltf_up_entry(eng: AtsEngine):
    eng.on_snapshot(Snap({"1m": view(60, 100.0, contraction=True, box_high=101.0, box_low=99.0)}))
    eng.on_snapshot(Snap({"1m": view(120, 102.0, contraction=False)}))      # breakout up
    return eng.on_snapshot(Snap({"1m": view(180, 99.9, contraction=False)}))  # pullback -> up entry


def _ltf_down_entry(eng: AtsEngine):
    eng.on_snapshot(Snap({"1m": view(60, 100.0, contraction=True, box_high=101.0, box_low=99.0)}))
    eng.on_snapshot(Snap({"1m": view(120, 98.0, contraction=False)}))       # breakout down
    return eng.on_snapshot(Snap({"1m": view(180, 100.1, contraction=False)}))  # pullback -> down entry


def t_gate_keeps_aligned() -> bool:
    eng = engine(); _set_htf_bias_up(eng)
    entries = [r for r in _ltf_up_entry(eng) if r.phase == "entry"]
    return (len(entries) == 1 and entries[0].direction == "up" and entries[0].htf_bias == "up"
            and entries[0].htf_dist_from_value_line is not None)


def t_gate_blocks_counter() -> bool:
    eng = engine(); _set_htf_bias_up(eng)
    out = _ltf_down_entry(eng)
    blocked = [r for r in out if r.phase == "entry_blocked"]
    return ([r for r in out if r.phase == "entry"] == []
            and len(blocked) == 1 and blocked[0].htf_bias == "up")


def t_gate_blocks_no_bias() -> bool:
    eng = engine()  # HTF never seen -> bias None
    out = _ltf_up_entry(eng)
    blocked = [r for r in out if r.phase == "entry_blocked"]
    return ([r for r in out if r.phase == "entry"] == []
            and len(blocked) == 1 and blocked[0].htf_bias == "none")


# -- timeframe ladder (3 TFs: 15m -> 5m -> 1m) ---------------------------------------
def t_ladder_mid_tf_entry_gated() -> bool:
    # 15m sets bias up; the 5m detector's pullback entry is kept and tagged with the 15m bias
    # (proves the MIDDLE timeframe now produces HTF-gated entries, not just 1m).
    eng = AtsEngine("TEST", P, ["15m", "5m", "1m"], {"15m": 900, "5m": 300, "1m": 60},
                    "ats_test", "hash")
    eng.on_snapshot(Snap({"15m": view(900, 99.5, contraction=True, box_high=100.0, box_low=98.0)}))
    eng.on_snapshot(Snap({"5m": view(300, 100.0, contraction=True, box_high=101.0, box_low=99.0)}))
    eng.on_snapshot(Snap({"5m": view(600, 102.0, contraction=False)}))            # 5m breakout up
    out = eng.on_snapshot(Snap({"5m": view(900, 99.9, contraction=False)}))       # pullback -> entry
    entries = [r for r in out if r.phase == "entry"]
    return (len(entries) == 1 and entries[0].timeframe == "5m"
            and entries[0].direction == "up" and entries[0].htf_bias == "up")


def t_ladder_top_tf_entries_dropped() -> bool:
    # The top TF has no higher TF to confirm it -> its entries are dropped (context only).
    eng = AtsEngine("TEST", P, ["15m", "1m"], {"15m": 900, "1m": 60}, "ats_test", "hash")
    eng.on_snapshot(Snap({"15m": view(900, 100.0, contraction=True, box_high=101.0, box_low=99.0)}))
    eng.on_snapshot(Snap({"15m": view(1800, 102.0, contraction=False)}))          # breakout up
    out = eng.on_snapshot(Snap({"15m": view(2700, 99.9, contraction=False)}))     # would-be entry
    return [r for r in out if r.phase == "entry"] == []


# -- pivot DETECTION (candles._compute_view) -----------------------------------------
def _frame(highs, lows, closes):
    idx = pd.to_datetime(range(60, 60 + 60 * len(highs), 60), unit="s", utc=True)
    return pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes}, index=idx)


def t_pivot_contraction_detected() -> bool:
    # Build bars whose pivot-highs step DOWN and pivot-lows step UP (a compression), length=2.
    # highs: a peak at i=2 (110), a lower peak at i=6 (108); lows: a trough at i=2 (90), higher at i=6 (92).
    p = {"atr_period": 3, "ats_pivot_lookback": 2}
    highs = [100, 101, 110, 101, 100, 101, 108, 101, 100]
    lows  = [99,  98,  90,  98,  99,  98,  92,  98,  99]
    closes = [100] * 9
    v = _compute_view(_frame(highs, lows, closes), "1m", p)
    # at the last bar (i=8), pivot at i=6 (=8-2) is confirmed: lower-high(108<110) + higher-low(92>90)
    return v is not None and v.contraction_now and v.box_high == 108.0 and v.box_low == 92.0


def t_pivot_box_close_mean() -> bool:
    # Same compression as t_pivot_contraction_detected, but closes vary over the box window
    # [min(ph,pl)=6 .. end]: closes[6:9] = 100,102,104 -> mean 102. (closes don't affect pivots.)
    p = {"atr_period": 3, "ats_pivot_lookback": 2}
    highs = [100, 101, 110, 101, 100, 101, 108, 101, 100]
    lows  = [99,  98,  90,  98,  99,  98,  92,  98,  99]
    closes = [100, 100, 100, 100, 100, 100, 100, 102, 104]
    v = _compute_view(_frame(highs, lows, closes), "1m", p)
    return (v is not None and v.contraction_now and v.box_close_mean is not None
            and abs(v.box_close_mean - 102.0) < 1e-9)


def t_pivot_no_contraction_when_expanding() -> bool:
    # Pivot-highs step UP (expansion, not compression) -> no contraction.
    p = {"atr_period": 3, "ats_pivot_lookback": 2}
    highs = [100, 101, 105, 101, 100, 101, 110, 101, 100]
    lows  = [99,  98,  95,  98,  99,  98,  90,  98,  99]
    closes = [100] * 9
    v = _compute_view(_frame(highs, lows, closes), "1m", p)
    return v is not None and not v.contraction_now


CHECKS = [
    ("warmup_gate", t_warmup_gate),
    ("contraction_and_value_line", t_contraction_and_value_line),
    ("value_line_offset_box", t_value_line_offset_box),
    ("breakout_up_plain", t_breakout_up_plain),
    ("no_break_inside_box", t_no_break_inside_box),
    ("breakout_down_plain", t_breakout_down_plain),
    ("entry_pullback_up", t_entry_pullback_up),
    ("entry_timeout", t_entry_timeout),
    ("refeed_dedup", t_refeed_dedup),
    ("value_line_contraction_mean", t_value_line_contraction_mean),
    ("value_line_mean_fallback_midpoint", t_value_line_mean_fallback_midpoint),
    ("value_fade_enters_against_break", t_value_fade_enters_against_break),
    ("gate_keeps_aligned", t_gate_keeps_aligned),
    ("gate_blocks_counter", t_gate_blocks_counter),
    ("gate_blocks_no_bias", t_gate_blocks_no_bias),
    ("ladder_mid_tf_entry_gated", t_ladder_mid_tf_entry_gated),
    ("ladder_top_tf_entries_dropped", t_ladder_top_tf_entries_dropped),
    ("pivot_contraction_detected", t_pivot_contraction_detected),
    ("pivot_box_close_mean", t_pivot_box_close_mean),
    ("pivot_no_contraction_when_expanding", t_pivot_no_contraction_when_expanding),
]


def main() -> None:
    results = {}
    for name, fn in CHECKS:
        try:
            results[name] = bool(fn())
        except Exception as e:
            results[name] = False
            log.info("  [ERROR] %s raised %s: %s", name, e.__class__.__name__, e)
    log.info("%s", "=" * 50)
    for name, ok in results.items():
        log.info("  %-36s %s", name, "PASS" if ok else "FAIL")
    passed = sum(results.values())
    log.info("%s\n%d/%d checks passed", "=" * 50, passed, len(results))
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

```

#### `verify_validation.py`

```python
"""Test the tester — verify the statistical core of validate_signals.py on inputs with KNOWN truth.

If these pass, the permutation test, CSCV/PBO, and deflated-Sharpe math are correct independent of
any live data — so when they say "no edge" on the bot's signals, that verdict is trustworthy.

Run:  python verify_validation.py
"""
from __future__ import annotations

import logging

import numpy as np

from validate_signals import perm_pvalue, cscv_pbo, expected_max_sharpe, sharpe

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("verify_validation")


def t_perm_significant() -> bool:
    # observed far above the null distribution -> small p
    return perm_pvalue(200.0, list(range(100))) < 0.05


def t_perm_not_significant() -> bool:
    # observed at the null median -> p ~ 0.5
    p = perm_pvalue(49.5, list(range(100)))
    return 0.4 < p < 0.65


def t_perm_worse_than_null() -> bool:
    # observed below everything -> p ~ 1.0
    return perm_pvalue(-1.0, list(range(100))) > 0.95


def t_pbo_genuine_edge_low() -> bool:
    # config 0 is best in EVERY block -> IS-best generalizes OOS -> PBO ~ 0
    M = np.ones((10, 8)); M[:, 0] = 5.0
    return cscv_pbo(M) < 0.1


def t_pbo_random_midrange() -> bool:
    # no config is genuinely better -> PBO ~ 0.5 (indistinguishable from luck)
    M = np.random.RandomState(0).randn(10, 8)
    pbo = cscv_pbo(M)
    return 0.2 < pbo < 0.8


def t_pbo_guards_bad_shape() -> bool:
    import math
    return math.isnan(cscv_pbo(np.zeros((3, 4))))  # odd #blocks -> NaN, not a crash


def t_expmax_increases_with_trials() -> bool:
    # same cross-trial Sharpe variance, more trials -> higher expected-max hurdle
    small = expected_max_sharpe([-1.0, 1.0])               # N=2
    large = expected_max_sharpe([-1.0, 1.0] * 50)          # N=100, same variance
    return large > small > 0


def t_expmax_zero_variance() -> bool:
    return expected_max_sharpe([0.5] * 6) == 0.0


def t_sharpe_basic() -> bool:
    s = sharpe([1.0, 1.0, 1.0])      # zero variance -> 0.0
    import math
    return s == 0.0 and math.isnan(sharpe([1.0]))


CHECKS = [
    ("perm_significant", t_perm_significant),
    ("perm_not_significant", t_perm_not_significant),
    ("perm_worse_than_null", t_perm_worse_than_null),
    ("pbo_genuine_edge_low", t_pbo_genuine_edge_low),
    ("pbo_random_midrange", t_pbo_random_midrange),
    ("pbo_guards_bad_shape", t_pbo_guards_bad_shape),
    ("expmax_increases_with_trials", t_expmax_increases_with_trials),
    ("expmax_zero_variance", t_expmax_zero_variance),
    ("sharpe_basic", t_sharpe_basic),
]


def main() -> None:
    results = {}
    for name, fn in CHECKS:
        try:
            results[name] = bool(fn())
        except Exception as e:
            results[name] = False
            log.info("  [ERROR] %s: %s", name, e)
    log.info("%s", "=" * 50)
    for name, ok in results.items():
        log.info("  %-32s %s", name, "PASS" if ok else "FAIL")
    passed = sum(results.values())
    log.info("%s\n%d/%d checks passed", "=" * 50, passed, len(results))
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

```

### Dashboard backend

#### `dashboard/server.py`

```python
"""FastAPI dashboard backend (read-only). REST over file readers + WS over the live feed.

Run:  .venv\\Scripts\\python -m uvicorn dashboard.server:app --port 8000
Then run the Vite dev server in dashboard/web (npm run dev), which proxies /api and /ws here.
NO trading endpoints — this is a research viewer.
"""
from __future__ import annotations

import asyncio
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
GRANULARITY = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}


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


@app.get("/api/archive_candles")
def api_archive_candles(symbol: str, tf: str = "1m", count: int = 2000):
    """Historical candles resampled from the tick archive (for the chart's 'archive' view)."""
    g = GRANULARITY.get(tf)
    if g is None:
        return JSONResponse({"error": f"unsupported tf {tf}"}, status_code=400)
    return readers.archive_candles(symbol, g, count)


@app.get("/api/deep")
async def api_deep(symbol: str, tf: str = "15m", count: int = 2000):
    """Deep historical view: fetch `count` candles of `tf` straight from Deriv (far deeper than the
    tick archive), run the detector over them for boxes + value lines, and attach ladder entries
    from the signal log. Display-only. The CPU-heavy replay runs in a thread so live feeds keep
    flowing; the result is cached in readers."""
    f = feeds.get(symbol)
    if not f:
        return JSONResponse({"error": "unknown symbol"}, status_code=404)
    g = GRANULARITY.get(tf)
    if g is None:
        return JSONResponse({"error": f"unsupported tf {tf}"}, status_code=400)
    candles = await f.history_candles(g, count)
    overlay = await asyncio.get_event_loop().run_in_executor(
        None, readers.deep_overlay, symbol, tf, candles)
    return {"candles": candles, **overlay}


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

```

#### `dashboard/readers.py`

```python
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
    sigs = rv._load_signals(symbol)  # ATS stream (review._load_signals defaults to ats_signal_dir)
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
    value_lines = _build_value_lines(sigs, htf, ltf)
    entries = [{"bar_epoch": s["bar_epoch"], "direction": s.get("direction"),
                "price": s.get("price_at_signal"), "tf": s["timeframe"],
                "value_line": s.get("value_line"), "htf_bias": s.get("htf_bias")}
               for s in sigs if s.get("phase") == "entry"]
    return {"symbol": symbol, "htf": htf, "ltf": ltf,
            "value_lines": value_lines[-limit:], "entries": entries[-limit:],
            "funnel": _ats_funnel(sigs, htf, ltf)}


_TF_SECS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}


def _build_value_lines(sigs: list[dict], htf: str, ltf: str) -> list[dict]:
    """One entry per contraction: the box (start/end × high/low) AND the value line projected
    FORWARD from the box — drawn the way TradeATS draws it (a box + a 'point of origin' line that
    extends right), not a single line connecting contractions. Projection runs to the next same-tf
    contraction, capped, so lines don't overlap or run forever."""
    length = CONFIG.ats_pivot_lookback
    cons = [s for s in sigs if s.get("phase") == "contraction" and s.get("value_line") is not None
            and s.get("timeframe") in (htf, ltf)]
    by_tf: dict[str, list] = {}
    for s in cons:
        by_tf.setdefault(s["timeframe"], []).append(s)
    out = []
    for tfname, lst in by_tf.items():
        secs = _TF_SECS.get(tfname, 60)
        lst.sort(key=lambda s: s["bar_epoch"])
        for i, s in enumerate(lst):
            be = int(s["bar_epoch"])
            nxt = int(lst[i + 1]["bar_epoch"]) if i + 1 < len(lst) else be + 60 * secs
            out.append({
                "tf": tfname, "epoch": be, "value_line": s["value_line"],
                "box_start": be - 2 * length * secs, "box_end": be,
                "box_high": s.get("contraction_high"), "box_low": s.get("contraction_low"),
                "line_end": min(nxt, be + 120 * secs),   # project forward to next, capped
            })
    out.sort(key=lambda v: v["epoch"])
    return out


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


def deep_value_lines(tf: str, candles: list[dict]) -> list[dict]:
    """Run the REAL single-timeframe ATS detector over a DEEP candle history (fetched live from
    Deriv, far longer than the tick archive) to produce contraction boxes + value lines across
    weeks. DISPLAY-ONLY — never written to the signal log, so the validated stats stay clean.
    CPU-heavy (per-bar _compute_view, ~O(n^2)); the server calls this off the event loop."""
    from candles import _compute_view
    from ats_signals import AtsTimeframeDetector
    length = CONFIG.ats_pivot_lookback
    if len(candles) < 2 * length + 1:
        return []
    idx = pd.to_datetime([c["time"] for c in candles], unit="s", utc=True)
    frame = pd.DataFrame({k: [float(c[k]) for c in candles] for k in ("open", "high", "low", "close")},
                         index=idx)
    det = AtsTimeframeDetector("deep", tf, _TF_SECS.get(tf, 60), CONFIG.ats_signal_params(), "deep", "deep")
    cons = []
    for i in range(len(frame)):
        view = _compute_view(frame.iloc[: i + 1], tf, CONFIG.view_params())
        if view is None:
            continue
        for rec in det.on_closed_bar(view):
            if rec.phase == "contraction":
                cons.append({"timeframe": tf, "bar_epoch": rec.bar_epoch, "value_line": rec.value_line,
                             "contraction_high": rec.contraction_high,
                             "contraction_low": rec.contraction_low, "phase": "contraction"})
    return _build_value_lines(cons, tf, tf)  # htf==ltf==tf -> keeps this tf's boxes/lines


def deep_overlay(symbol: str, tf: str, candles: list[dict]) -> dict:
    """Deep historical overlay for the chart: structure (boxes + value lines) computed over the
    fetched deep candles for `tf`, plus the real ladder ENTRY arrows for `tf` read from the
    (backfilled) signal log. Shaped as an AtsOverlay with htf==ltf==tf so the existing chart draws
    it. Cached on (symbol, tf, depth, last-candle) so it recomputes only when the data moves."""
    key = ("deep", symbol, tf, len(candles), candles[-1]["time"] if candles else 0)
    vls = _memo(key, 300.0, lambda: deep_value_lines(tf, candles))
    sigs = rv._load_signals(symbol, signal_dir=CONFIG.ats_signal_dir)
    entries = [{"bar_epoch": s["bar_epoch"], "direction": s.get("direction"),
                "price": s.get("price_at_signal"), "tf": s["timeframe"],
                "value_line": s.get("value_line"), "htf_bias": s.get("htf_bias")}
               for s in sigs if s.get("phase") == "entry" and s.get("timeframe") == tf]
    return {"symbol": symbol, "htf": tf, "ltf": tf, "value_lines": vls, "entries": entries,
            "funnel": _ats_funnel(sigs, CONFIG.ats_htf, CONFIG.ats_ltf)}


def backtest_summary(symbol: str, payout: float | None = None, duration_bars: int | None = None) -> dict:
    key = ("bt", symbol, payout, duration_bars)
    return _memo(key, 10.0, lambda: bt.run_backtest(symbol, payout=payout, duration_bars=duration_bars))


def health(symbol: str) -> dict:
    return _memo(("health", symbol), 8.0, lambda: _health(symbol))


def _health(symbol: str) -> dict:
    files = sorted(glob.glob(str(CONFIG.tick_dir / symbol / "*.parquet")))
    sig_files = glob.glob(str(CONFIG.ats_signal_dir / symbol / "*.jsonl"))
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

```

#### `dashboard/live.py`

```python
"""Live market feed for the dashboard — one Deriv subscription per symbol, fanned out to WS clients.

Read-only viewer: reuses DerivClient + MultiTimeframeStore exactly like main.py's wire-up, but does
NOT persist (the collector bots already archive every tick). Each LiveFeed opens its own public
Deriv connection (no token) and broadcasts tick + forming-candle events to subscribed browsers.
"""
from __future__ import annotations

import asyncio
import logging
import time

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
        self._candle_cache: dict = {}  # (granularity, count) -> (monotonic_ts, data)

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

    async def history_candles(self, granularity: int, count: int = 500) -> list[dict]:
        """Fetch NATIVE candle history from Deriv at any granularity (for the chart's timeframe
        switcher — full history regardless of the in-process base). Cached ~10s so the frontend
        poll doesn't hammer the API. Falls back to the last cache (or []) if the socket is mid-drop."""
        key = (int(granularity), int(count))
        hit = self._candle_cache.get(key)
        if hit and time.monotonic() - hit[0] < 10:
            return hit[1]
        try:
            res = await self.client.send({"ticks_history": self.symbol, "style": "candles",
                                          "granularity": int(granularity), "count": int(count),
                                          "end": "latest"})
            data = [{"time": int(c["epoch"]), "open": float(c["open"]), "high": float(c["high"]),
                     "low": float(c["low"]), "close": float(c["close"])} for c in res.get("candles", [])]
            self._candle_cache[key] = (time.monotonic(), data)
            return data
        except Exception:
            return hit[1] if hit else []

    async def stop(self) -> None:
        await self.client.close()
        if self._task:
            self._task.cancel()

```

### Dashboard frontend (React/TS)

#### `dashboard/web/src/api.ts`

```ts
import { useEffect, useRef } from "react";

export type Candle = { time: number; open: number; high: number; low: number; close: number };
export type SignalRec = {
  timeframe: string; phase: string; direction: string | null; bar_epoch: number;
  price_at_signal: number; value_line?: number | null; htf_bias?: string | null; [k: string]: any;
};
export type Health = {
  symbol: string; ticks: number; signals: number; last_tick_age_s: number | null;
  coverage_pct: number | null; gaps: number | null; live: boolean;
};
export type Backtest = {
  error?: string; verdict?: string; verdict_class?: string; caveat?: string; breakeven?: number;
  real?: { win_rate: number; total_pnl: number; roi_pct: number; n: number };
  null?: { win_rate: number; total_pnl: number; roi_pct: number; n: number } | null;
};
export type AtsValueLine = {
  epoch: number; value_line: number; tf: string;
  box_start: number; box_end: number; box_high: number | null; box_low: number | null;
  line_end: number;
};
export type AtsEntry = {
  bar_epoch: number; direction: string | null; price: number | null; tf: string;
  value_line: number | null; htf_bias: string | null;
};
export type AtsFunnel = {
  htf_contractions: number; htf_breakouts: number;
  ltf_contractions: number; ltf_breakouts: number;
  pullback_candidates: number; entries: number;
  blocked_no_bias: number; blocked_counter: number;
};
export type AtsOverlay = {
  symbol: string; htf: string; ltf: string;
  value_lines: AtsValueLine[]; entries: AtsEntry[]; funnel: AtsFunnel;
};

const j = (url: string) => fetch(url).then((r) => r.json());

export const getSymbols = (): Promise<{ symbol: string; live: boolean }[]> => j("/api/symbols");
export const getCandles = (s: string, tf = "1m", count = 500): Promise<Candle[]> =>
  j(`/api/candles?symbol=${s}&tf=${tf}&count=${count}`);
export const getArchiveCandles = (s: string, tf = "1m", count = 2000): Promise<Candle[]> =>
  j(`/api/archive_candles?symbol=${s}&tf=${tf}&count=${count}`);
export const getSignals = (s: string, limit = 100): Promise<SignalRec[]> =>
  j(`/api/signals?symbol=${s}&limit=${limit}`);
export const getBacktest = (s: string): Promise<Backtest> => j(`/api/backtest?symbol=${s}`);
export const getHealth = (s: string): Promise<Health> => j(`/api/health?symbol=${s}`);
export const getAts = (s: string): Promise<AtsOverlay> => j(`/api/ats?symbol=${s}`);
/** Deep historical view: candles fetched from Deriv at `tf` + ATS overlay (htf==ltf==tf) computed over them. */
export type DeepView = AtsOverlay & { candles: Candle[] };
export const getDeep = (s: string, tf = "15m", count = 2000): Promise<DeepView> =>
  j(`/api/deep?symbol=${s}&tf=${tf}&count=${count}`);

/** Subscribe to the live WS feed for `symbol`; auto-reconnects on drop. */
export function useLiveFeed(
  symbol: string,
  onTick: (price: number, epoch: number) => void,
  onCandle: (bar: Candle) => void
) {
  const cb = useRef({ onTick, onCandle });
  cb.current = { onTick, onCandle };
  useEffect(() => {
    if (!symbol) return;
    let ws: WebSocket | null = null;
    let closed = false;
    let retry: ReturnType<typeof setTimeout>;
    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws?symbol=${symbol}`);
      ws.onmessage = (e) => {
        const m = JSON.parse(e.data);
        if (m.type === "tick") cb.current.onTick(m.price, m.epoch);
        else if (m.type === "candle") cb.current.onCandle(m.bar);
      };
      ws.onclose = () => { if (!closed) retry = setTimeout(connect, 1500); };
      ws.onerror = () => ws?.close();
    };
    connect();
    return () => { closed = true; clearTimeout(retry); ws?.close(); };
  }, [symbol]);
}

```

#### `dashboard/web/src/App.tsx`

```tsx
import { useCallback, useEffect, useState, type ReactNode } from "react";
import Chart from "./Chart";
import {
  type AtsOverlay, type Backtest, type Candle, type Health, type SignalRec,
  getArchiveCandles, getAts, getBacktest, getCandles, getDeep, getHealth, getSignals, getSymbols, useLiveFeed,
} from "./api";

const TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"];
type Mode = "live" | "archive" | "deep";

export default function App() {
  const [symbols, setSymbols] = useState<string[]>([]);
  const [symbol, setSymbol] = useState("");
  const [tf, setTf] = useState("1m");
  const [mode, setMode] = useState<Mode>("live");
  const [candles, setCandles] = useState<Candle[]>([]);
  const [liveBar, setLiveBar] = useState<Candle | null>(null);
  const [price, setPrice] = useState<number | null>(null);
  const [signals, setSignals] = useState<SignalRec[]>([]);
  const [ats, setAts] = useState<AtsOverlay | null>(null);
  const [bt, setBt] = useState<Backtest | null>(null);
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    getSymbols().then((s) => {
      setSymbols(s.map((x) => x.symbol));
      if (s[0]) setSymbol((cur) => cur || s[0].symbol);
    });
  }, []);

  useEffect(() => {
    if (!symbol) return;
    let stop = false;
    setCandles([]); setLiveBar(null);
    const loadCandles = () =>
      (mode === "archive" ? getArchiveCandles(symbol, tf) : getCandles(symbol, tf))
        .then((c) => { if (!stop) setCandles(c); });
    const poll = () => {
      if (mode === "deep") {
        // Deep history: candles + overlay (boxes/value lines + entries) for the viewed tf, in one call.
        getDeep(symbol, tf).then((d) => { if (!stop) { setCandles(d.candles); setAts(d); } });
      } else {
        loadCandles();                     // refetch candles so higher TFs stay current
        getAts(symbol).then((a) => { if (!stop) setAts(a); });
      }
      getSignals(symbol).then(setSignals);
      getBacktest(symbol).then(setBt);
      getHealth(symbol).then(setHealth);
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => { stop = true; clearInterval(id); };
  }, [symbol, tf, mode]);

  const onTick = useCallback((p: number) => setPrice(p), []);
  const onCandle = useCallback((bar: Candle) => setLiveBar({ ...bar }), []);
  useLiveFeed(symbol, onTick, onCandle);

  return (
    <div className="app">
      <div className="caveat">
        Research harness — Deriv synthetic indices are CSPRNG-generated: there is no predictive edge.
        Demo only · No trading · Read-only viewer.
      </div>
      <header>
        <h1>Deriv Research Dashboard</h1>
        <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
          {symbols.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <select value={tf} onChange={(e) => setTf(e.target.value)} title="chart timeframe">
          {TIMEFRAMES.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
        <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}
          title="live feed · archive (resampled from recorded ticks) · deep (Deriv candle history, display-only indicators)">
          <option value="live">live</option>
          <option value="archive">archive</option>
          <option value="deep">deep (history)</option>
        </select>
        <span className="price">{price !== null ? price.toFixed(5) : "—"}</span>
        {health && <span className={"badge " + (health.live ? "ok" : "bad")}>{health.live ? "LIVE" : "STALE"}</span>}
      </header>

      <Chart candles={candles} liveBar={liveBar} tf={tf} ats={ats} mode={mode} />

      <div className="grid">
        <BacktestPanel bt={bt} />
        <AtsFunnelPanel ats={ats} />
        <HealthPanel health={health} />
      </div>

      <SignalsTable signals={signals} />
    </div>
  );
}

function pct(x: number | undefined) { return x === undefined || isNaN(x) ? "—" : (x * 100).toFixed(1); }

function BacktestPanel({ bt }: { bt: Backtest | null }) {
  if (!bt) return <Panel title="Backtest — would it make money?"><p>loading…</p></Panel>;
  if (bt.error || !bt.real) return <Panel title="Backtest — would it make money?"><p className="small">{bt.error || "no tradeable signals yet"}</p></Panel>;
  const r = bt.real, n = bt.null;
  const vcls = { good: "ok", watch: "watch", weak: "weak", bad: "bad" }[bt.verdict_class || "bad"] || "bad";
  return (
    <Panel title="Backtest — would it make money?">
      <table>
        <thead><tr><th></th><th>win %</th><th>P&amp;L</th><th>ROI</th></tr></thead>
        <tbody>
          <tr><td>Real signals</td><td>{pct(r.win_rate)}</td><td>{r.total_pnl.toFixed(2)}</td><td>{r.roi_pct.toFixed(1)}%</td></tr>
          {n && <tr className="dim"><td>Random (null)</td><td>{pct(n.win_rate)}</td><td>{n.total_pnl.toFixed(2)}</td><td>{n.roi_pct.toFixed(1)}%</td></tr>}
        </tbody>
      </table>
      <p>break-even win rate <b>{pct(bt.breakeven)}%</b> · <span className={vcls}>{bt.verdict}</span></p>
      <p className="small">{bt.caveat}</p>
    </Panel>
  );
}

function AtsFunnelPanel({ ats }: { ats: AtsOverlay | null }) {
  if (!ats) return <Panel title="ATS funnel"><p>loading…</p></Panel>;
  const f = ats.funnel;
  const row = (label: string, n: number) => (
    <tr><td>{label}</td><td style={{ textAlign: "right" }}><b>{n}</b></td></tr>
  );
  return (
    <Panel title="ATS funnel — where the chain collapses">
      <table>
        <tbody>
          {row(`${ats.htf} contractions`, f.htf_contractions)}
          {row(`${ats.htf} breakouts`, f.htf_breakouts)}
          {row(`${ats.ltf} contractions`, f.ltf_contractions)}
          {row(`${ats.ltf} breakouts`, f.ltf_breakouts)}
          {row("pullback candidates", f.pullback_candidates)}
          {row("→ entries (HTF-aligned)", f.entries)}
          {row("blocked: no HTF bias", f.blocked_no_bias)}
          {row("blocked: counter-bias", f.blocked_counter)}
        </tbody>
      </table>
      <p className="small">
        Read top-down: if {ats.htf} contractions ≈ 0 the HTF rarely sets up; high "no HTF bias"
        means the {ats.htf} bias is undefined when {ats.ltf} pulls back. Diagnoses the bottleneck
        without changing any rule. Entries are selective by design — no edge implied.
      </p>
    </Panel>
  );
}

function HealthPanel({ health }: { health: Health | null }) {
  if (!health) return <Panel title="Archive health"><p>loading…</p></Panel>;
  return (
    <Panel title="Archive health">
      <p>ticks <b>{health.ticks.toLocaleString()}</b> · signals <b>{health.signals}</b></p>
      <p>coverage <b>{health.coverage_pct ?? "—"}%</b> · gaps <b>{health.gaps ?? "—"}</b></p>
      <p>last archived tick <b>{health.last_tick_age_s ?? "—"}s</b> ago
        <span className={"badge " + (health.live ? "ok" : "bad")}>{health.live ? "fresh" : "stale"}</span></p>
      <p className="small">Freshness is archive-based (bots flush every ~100 ticks); the live chart above is real-time.</p>
    </Panel>
  );
}

function SignalsTable({ signals }: { signals: SignalRec[] }) {
  return (
    <Panel title={`Recent ATS signals (${signals.length})`}>
      <div className="scroll">
        <table className="sig">
          <thead><tr><th>tf</th><th>phase</th><th>dir</th><th>price</th><th>value line</th><th>HTF bias</th><th>bar epoch</th></tr></thead>
          <tbody>
            {signals.slice(0, 60).map((s, i) => (
              <tr key={i}>
                <td>{s.timeframe}</td>
                <td className={s.phase === "entry" ? "ok" : ""}>{s.phase}</td>
                <td>{s.direction ?? "—"}</td>
                <td>{s.price_at_signal?.toFixed?.(5)}</td>
                <td>{s.value_line != null ? s.value_line.toFixed(5) : "—"}</td>
                <td>{s.htf_bias ?? "—"}</td>
                <td>{s.bar_epoch}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return <div className="panel"><h3>{title}</h3>{children}</div>;
}

```

#### `dashboard/web/src/Chart.tsx`

```tsx
import { useEffect, useRef, useState } from "react";
import { createChart, type IChartApi, type ISeriesApi, type Time } from "lightweight-charts";
import type { AtsEntry, AtsOverlay, Candle } from "./api";

type Tip = { x: number; y: number; e: AtsEntry } | null;

export default function Chart({
  candles, liveBar, tf, ats, mode,
}: {
  candles: Candle[]; liveBar: Candle | null;
  tf: string; ats: AtsOverlay | null; mode: "live" | "archive" | "deep";
}) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const series = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const overlay = useRef<ISeriesApi<"Line">[]>([]);   // ATS boxes + forward value lines (one each)
  const entriesRef = useRef<AtsEntry[]>([]);
  const [tip, setTip] = useState<Tip>(null);

  useEffect(() => {
    if (!el.current) return;
    const c = createChart(el.current, {
      height: 440,
      layout: { background: { color: "#0e1117" }, textColor: "#c9d1d9" },
      grid: { vertLines: { color: "#1b2230" }, horzLines: { color: "#1b2230" } },
      timeScale: { timeVisible: true, secondsVisible: false },
      rightPriceScale: { borderColor: "#1b2230" },
    });
    chart.current = c;
    series.current = c.addCandlestickSeries({
      upColor: "#26a69a", downColor: "#ef5350", borderVisible: false,
      wickUpColor: "#26a69a", wickDownColor: "#ef5350",
    });
    // Hover tooltip: when the crosshair is over an ATS entry bar, show its details.
    c.subscribeCrosshairMove((param) => {
      if (!param.time || !param.point) { setTip(null); return; }
      const t = param.time as number;
      const e = entriesRef.current.find((x) => x.bar_epoch === t);
      setTip(e ? { x: param.point.x, y: param.point.y, e } : null);
    });
    const ro = new ResizeObserver(() => c.applyOptions({ width: el.current!.clientWidth }));
    ro.observe(el.current);
    return () => { ro.disconnect(); c.remove(); chart.current = null; series.current = null; overlay.current = []; };
  }, []);

  useEffect(() => {
    if (series.current && candles.length) series.current.setData(candles as any);
  }, [candles]);

  useEffect(() => {
    // The WS feed only streams the forming 1m bar — apply it on the live 1m chart only.
    if (series.current && liveBar && tf === "1m" && mode === "live") series.current.update(liveBar as any);
  }, [liveBar, tf, mode]);

  // ATS overlay (TradeATS "global view"): each swing-pivot contraction = a box (top/bottom over its
  // bars) + a VALUE LINE projected forward. Drawn for the timeframe being viewed (the pivot detector
  // is selective, so these are meaningful swing compressions, not noise). One short line series per
  // element (no left-edge clamp); only elements intersecting the visible window are drawn.
  useEffect(() => {
    const c = chart.current;
    if (!c) return;
    for (const s of overlay.current) c.removeSeries(s);
    overlay.current = [];
    if (!candles.length || !ats) return;
    if (tf !== ats.htf && tf !== ats.ltf) return;
    const lo = candles[0].time as number, hi = candles[candles.length - 1].time as number;
    const seg = (color: string, width: 1 | 2, t0: number, t1: number, v: number | null) => {
      if (v == null || t1 < lo || t0 > hi) return;
      const s = c.addLineSeries({
        color, lineWidth: width, priceLineVisible: false, lastValueVisible: false,
        crosshairMarkerVisible: false,
      });
      s.setData([{ time: t0 as Time, value: v }, { time: t1 as Time, value: v }]);
      overlay.current.push(s);
    };
    for (const v of ats.value_lines) {
      if (v.tf !== tf) continue;                                 // value lines for the viewed timeframe
      seg("#58a6ff", 2, v.box_start, v.line_end, v.value_line);  // value line (point of origin)
      seg("#3b6ea5", 1, v.box_start, v.box_end, v.box_high);     // box top
      seg("#3b6ea5", 1, v.box_start, v.box_end, v.box_low);      // box bottom
    }
  }, [ats, tf, candles]);

  // ATS pullback entries (purple arrows) — on the LTF chart, clipped to the visible window.
  useEffect(() => {
    if (!series.current) return;
    const lo = candles.length ? (candles[0].time as number) : -Infinity;
    const hi = candles.length ? (candles[candles.length - 1].time as number) : Infinity;
    const entries = (ats && tf === ats.ltf) ? ats.entries : [];
    entriesRef.current = entries;
    const markers = entries
      .filter((e) => e.bar_epoch >= lo && e.bar_epoch <= hi)
      .map((e) => {
        const up = e.direction === "up";
        return {
          time: e.bar_epoch as Time, position: (up ? "belowBar" : "aboveBar") as any,
          color: "#d2a8ff", shape: (up ? "arrowUp" : "arrowDown") as any, text: "ATS",
        };
      })
      .sort((a, b) => (a.time as number) - (b.time as number));
    series.current.setMarkers(markers as any);
  }, [ats, tf, candles]);

  const e = tip?.e;
  return (
    <div className="chartwrap">
      <div ref={el} style={{ width: "100%" }} />
      <div className="legend">
        <span><i className="vline" />value line (point of origin)</span>
        <span><i className="box" />swing contraction box</span>
        <span><i className="arr up" style={{ borderBottomColor: "#d2a8ff" }} />ATS pullback entry ({ats?.ltf ?? "1m"})</span>
      </div>
      {e && (
        <div className="tip" style={{ left: tip!.x + 14, top: tip!.y + 8 }}>
          <b>ATS entry {e.direction}</b> · {e.tf}<br />
          price {e.price?.toFixed?.(5)}<br />
          value {e.value_line?.toFixed?.(5)} · HTF bias {e.htf_bias ?? "—"}
        </div>
      )}
    </div>
  );
}

```

#### `dashboard/web/src/main.tsx`

```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

```
