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
