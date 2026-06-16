"""Phase 2 signal detection — a pure, side-effect-free volatility contraction/expansion detector.

NO trading. NO I/O. NO pandas. This module consumes ONLY `TFView` floats from `MarketSnapshot`
(built in candles.py) and emits `SignalRecord`s to be LOGGED for later review. It is fully
unit-testable in isolation (see verify_signals.py).

Per timeframe, a small state machine:

    NO_SIGNAL --(band-width in low tail)--> CONTRACTION --(range breakout)--> EXPANSION -> NO_SIGNAL
                                                  |__(no breakout in time)__> NO_SIGNAL (timeout)

Honest caveat: on a CSPRNG synthetic these transitions are reading noise; the deliverable is the
LOGGED record + the review tool that measures whether they had any forward edge (they won't).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

SCHEMA_VERSION = 1


class Phase(str, Enum):
    NO_SIGNAL = "no_signal"
    CONTRACTION = "contraction"
    EXPANSION = "expansion"
    TREND = "trend"          # after expansion, the move CONTINUED in the breakout direction
    REVERSAL = "reversal"    # after expansion, price retraced back through the breakout level


@dataclass(frozen=True)
class SignalRecord:
    """One logged signal. Carries everything needed to recompute forward outcomes later WITHOUT
    re-running the detector. `bar_close_epoch` is the look-ahead firewall: the earliest instant a
    forward-outcome measurement may inspect."""
    schema_version: int
    signal_version: str
    detected_at_epoch: int
    symbol: str
    timeframe: str
    phase: str                      # "contraction" | "expansion" | "trend" | "reversal"
    direction: str | None           # "up" | "down" | None (trend=continue dir; reversal=new dir)
    bar_epoch: int                  # trigger bar open-time (dedup key with timeframe+phase)
    bar_close_epoch: int            # bar_epoch + tf_seconds — look-ahead firewall
    price_at_signal: float          # trigger bar close (outcome baseline)
    band_width: float
    bw_threshold: float
    bw_percentile: float
    bbw_zscore: float
    atr: float
    atr_at_contraction: float | None
    contraction_high: float | None
    contraction_low: float | None
    contraction_bars: int | None    # bars elapsed in the phase that led here (0 for contraction)
    episode_id: str                 # ties contraction → expansion → trend/reversal of one episode
    params_hash: str
    # ATS Master Pattern fields (None for the Phase-2 detector). value_line = midpoint of the frozen
    # contraction box; htf_bias = HTF side of its own value line at the LTF entry. The three *_metadata
    # fields are for RESEARCH analysis only — any FILTER derived from them is a new config (must enter
    # the PBO sweep, never a post-hoc cherry-pick).
    value_line: float | None = None
    htf_bias: str | None = None
    dist_from_value_line: float | None = None      # |entry close - LTF value line| at entry
    bars_since_expansion: int | None = None         # LTF bars from breakout to the entry pullback
    htf_dist_from_value_line: float | None = None   # |HTF close - HTF value line| at entry

    def to_dict(self) -> dict:
        return asdict(self)


class TimeframeDetector:
    """State machine for ONE timeframe. Fed closed-bar TFViews via on_closed_bar()."""

    def __init__(self, symbol: str, tf: str, tf_seconds: int, params: dict,
                 signal_version: str, params_hash: str):
        self.symbol = symbol
        self.tf = tf
        self.tf_seconds = tf_seconds
        self.p = params
        self.signal_version = signal_version
        self.params_hash = params_hash

        self.phase = Phase.NO_SIGNAL
        self._armed = True            # hysteresis: must re-arm (vol rises) before a new contraction
        self._last_epoch: int | None = None
        # frozen at contraction entry:
        self.c_high: float | None = None
        self.c_low: float | None = None
        self.c_atr: float | None = None
        self.c_entry_epoch: int | None = None
        self.episode_id: str | None = None
        self.bars_in_contraction = 0
        # frozen at expansion entry (for the TREND phase):
        self.exp_dir: str | None = None
        self.exp_close: float | None = None
        self.bars_in_trend = 0

    def on_closed_bar(self, view) -> SignalRecord | None:
        # Warm-up gate + only ever act on a genuinely NEW closed bar (reconnects can re-feed bars).
        if not view.warm:
            return None
        if self._last_epoch is not None and view.closed_bar_epoch <= self._last_epoch:
            return None
        self._last_epoch = view.closed_bar_epoch

        contraction_pct = self.p["contraction_pct"]
        exit_pct = self.p["contraction_exit_pct"]

        if self.phase is Phase.NO_SIGNAL:
            if not self._armed:
                # Wait for volatility to climb back out of the low tail before re-arming, so one
                # quiet stretch can't emit a burst of contraction signals.
                if view.bw_percentile > exit_pct:
                    self._armed = True
                return None
            if view.bw_percentile <= contraction_pct:
                return self._enter_contraction(view)
            return None

        if self.phase is Phase.CONTRACTION:
            self.bars_in_contraction += 1
            n = self.p["breakout_atr_mult"]
            up = view.close > self.c_high + n * self.c_atr
            down = view.close < self.c_low - n * self.c_atr
            if up or down:
                return self._enter_expansion(view, "up" if up else "down")
            if self.bars_in_contraction >= self.p["max_contraction_bars"]:
                self._end_episode()  # timed out — emit nothing
            return None

        # Phase.TREND — tracking the post-expansion move: continuation vs retrace through breakout.
        self.bars_in_trend += 1
        cont = self.p["trend_continue_atr"] * self.c_atr
        if self.exp_dir == "up":
            if view.close >= self.exp_close + cont:                       # extended further up
                return self._close_episode(view, Phase.TREND, "up")
            if view.close < self.c_high:                                  # fell back through breakout
                return self._close_episode(view, Phase.REVERSAL, "down")
        else:  # down breakout
            if view.close <= self.exp_close - cont:                       # extended further down
                return self._close_episode(view, Phase.TREND, "down")
            if view.close > self.c_low:                                   # popped back through breakout
                return self._close_episode(view, Phase.REVERSAL, "up")
        if self.bars_in_trend >= self.p["trend_max_bars"]:
            self._end_episode()  # inconclusive — emit nothing
        return None

    # -- transitions -----------------------------------------------------------------
    def _enter_contraction(self, view) -> SignalRecord:
        self.phase = Phase.CONTRACTION
        self.c_high = view.range_high
        self.c_low = view.range_low
        self.c_atr = view.atr
        self.c_entry_epoch = view.closed_bar_epoch
        self.bars_in_contraction = 0
        self.episode_id = f"{self.symbol}:{self.tf}:{view.closed_bar_epoch}"
        return self._make(view, Phase.CONTRACTION, None, 0)

    def _enter_expansion(self, view, direction: str) -> SignalRecord:
        """Emit the expansion breakout, then enter TREND to track what happens next."""
        rec = self._make(view, Phase.EXPANSION, direction, self.bars_in_contraction)
        self.phase = Phase.TREND
        self.exp_dir = direction
        self.exp_close = view.close
        self.bars_in_trend = 0
        return rec  # episode_id + c_* stay frozen through the trend phase

    def _close_episode(self, view, phase: Phase, direction: str) -> SignalRecord:
        """Emit a trend/reversal record AND end the episode (back to NO_SIGNAL, disarmed)."""
        rec = self._make(view, phase, direction, self.bars_in_trend)
        self._end_episode()
        return rec

    def _end_episode(self) -> None:
        self.phase = Phase.NO_SIGNAL
        self._armed = False  # require hysteresis re-arm
        self.c_high = self.c_low = self.c_atr = self.c_entry_epoch = self.episode_id = None
        self.exp_dir = self.exp_close = None
        self.bars_in_contraction = self.bars_in_trend = 0

    def _make(self, view, phase: Phase, direction: str | None, contraction_bars: int) -> SignalRecord:
        bar_close = view.closed_bar_epoch + self.tf_seconds
        return SignalRecord(
            schema_version=SCHEMA_VERSION,
            signal_version=self.signal_version,
            detected_at_epoch=bar_close,
            symbol=self.symbol,
            timeframe=self.tf,
            phase=phase.value,
            direction=direction,
            bar_epoch=view.closed_bar_epoch,
            bar_close_epoch=bar_close,
            price_at_signal=view.close,
            band_width=view.band_width,
            bw_threshold=view.bw_threshold,
            bw_percentile=view.bw_percentile,
            bbw_zscore=view.bbw_zscore,
            atr=view.atr,
            atr_at_contraction=self.c_atr,
            contraction_high=self.c_high,
            contraction_low=self.c_low,
            contraction_bars=contraction_bars,
            episode_id=self.episode_id,
            params_hash=self.params_hash,
        )


class SignalEngine:
    """Owns one detector per signal timeframe; fans a snapshot out to each and dedups output on
    (timeframe, bar_epoch, phase) within the session (the SignalStore dedups on disk too)."""

    def __init__(self, symbol: str, params: dict, signal_timeframes, tf_seconds: dict,
                 signal_version: str, params_hash: str):
        self.timeframes = tuple(signal_timeframes)
        self.detectors = {
            tf: TimeframeDetector(symbol, tf, int(tf_seconds[tf]), params, signal_version, params_hash)
            for tf in self.timeframes
        }
        self._seen: set[tuple[str, int, str]] = set()

    def on_snapshot(self, snap) -> list[SignalRecord]:
        out: list[SignalRecord] = []
        for tf in self.timeframes:
            view = snap.views.get(tf)
            if view is None:
                continue
            rec = self.detectors[tf].on_closed_bar(view)
            if rec is None:
                continue
            key = (rec.timeframe, rec.bar_epoch, rec.phase)
            if key in self._seen:
                continue
            self._seen.add(key)
            out.append(rec)
        return out
