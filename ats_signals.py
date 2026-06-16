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
        return rec  # value_line + box + episode_id stay frozen through expansion

    def _emit_entry(self, view, direction: str, bars_since: int) -> SignalRecord:
        """Emit the (HTF-ungated) entry candidate in `direction`, then end the episode (one per
        episode). The engine applies the HTF-bias gate and stamps htf_bias/htf_dist."""
        rec = self._make(view, P_ENTRY, direction, bars_since,
                         dist=abs(view.close - self.value_line),
                         bars_since_expansion=bars_since)
        self._end_episode()
        return rec

    def _end_episode(self) -> None:
        # Keep value_line + last_close (persistent bias); reset only the episode machinery.
        self.phase = AtsPhase.NO_SIGNAL
        self.box_high = self.box_low = self.c_atr = self.episode_id = None
        self.exp_dir = None
        self.bars_in_contraction = self.bars_since_exp = 0

    def _make(self, view, phase: str, direction: str | None, bars: int,
              dist: float | None = None, bars_since_expansion: int | None = None) -> SignalRecord:
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
        )


class AtsEngine:
    """Owns an HTF detector + an LTF detector. The HTF sets bias; LTF pullback entries are emitted
    ONLY when their direction agrees with the HTF bias. Context records (contraction/breakout) from
    BOTH timeframes pass through (for the dashboard value line + funnel stats). Dedups output on
    (timeframe, bar_epoch, phase) within the session (SignalStore also dedups on disk)."""

    def __init__(self, symbol: str, params: dict, htf: str, ltf: str, tf_seconds: dict,
                 signal_version: str, params_hash: str):
        self.htf = htf
        self.ltf = ltf
        self.htf_det = AtsTimeframeDetector(symbol, htf, int(tf_seconds[htf]), params,
                                            signal_version, params_hash)
        self.ltf_det = AtsTimeframeDetector(symbol, ltf, int(tf_seconds[ltf]), params,
                                            signal_version, params_hash)
        self._seen: set[tuple[str, int, str]] = set()

    def on_snapshot(self, snap) -> list[SignalRecord]:
        out: list[SignalRecord] = []
        # HTF first so its value_line/bias reflect the current bar before gating LTF entries.
        htf_view = snap.views.get(self.htf)
        if htf_view is not None:
            for rec in self.htf_det.on_closed_bar(htf_view):
                if rec.phase != P_ENTRY:        # HTF is for bias + context only; drop HTF entries
                    self._add(rec, out)

        ltf_view = snap.views.get(self.ltf)
        if ltf_view is not None:
            bias = self.htf_det.bias
            for rec in self.ltf_det.on_closed_bar(ltf_view):
                if rec.phase == P_ENTRY:
                    hv, hl = self.htf_det.value_line, self.htf_det.last_close
                    hdist = abs(hl - hv) if hv is not None and hl is not None else None
                    if bias is not None and rec.direction == bias:
                        rec = replace(rec, htf_bias=bias, htf_dist_from_value_line=hdist)
                    else:
                        # HTF gate rejected it (no bias or counter-bias) — keep as a NON-tradeable
                        # blocked candidate so the funnel shows where entries are filtered out.
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
