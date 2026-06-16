"""ATS Master Pattern detector regression — offline, deterministic, PASS/FAIL (like verify_signals.py).

Feeds hand-built TFViews through ats_signals.py to prove the value-line + HTF→LTF pullback logic:
contraction (inside bars), value-line midpoint, box breakout, pullback entry, the HTF-bias gate,
warm-up gate, timeout, and engine dedup. No pandas, no network.

Run:  python verify_ats.py
"""
from __future__ import annotations

import logging

from ats_signals import AtsEngine, AtsPhase, AtsTimeframeDetector
from candles import TFView

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("verify_ats")

# Small, explicit params. max_contraction/entry kept short so timeouts fire quickly.
P = {
    "atr_period": 14, "ats_contraction_bars": 2, "ats_breakout_buffer_atr": 0.25,
    "ats_pullback_tol_atr": 0.0, "ats_max_contraction_bars": 3, "ats_max_entry_bars": 3,
}


def view(epoch, close, *, inside_run, box_high=101.0, box_low=99.0, atr=1.0, warm=True):
    """A warm (or non-warm) ATS TFView. high/low set wide so they don't interfere; the detector
    reads close, inside_run, box_high/low, atr."""
    return TFView(
        tf="1m", closed_bar_epoch=epoch, close=close, high=close + 1, low=close - 1, n_bars=200,
        atr=atr if warm else None,
        inside_run=inside_run if warm else None,
        box_high=box_high, box_low=box_low,
    )


def det(tf="1m", secs=60) -> AtsTimeframeDetector:
    return AtsTimeframeDetector("TEST", tf, secs, P, "ats_test", "hash")


class Snap:
    """Minimal MarketSnapshot stand-in: just a .views dict (tf -> TFView)."""
    def __init__(self, views):
        self.views = views


def engine() -> AtsEngine:
    return AtsEngine("TEST", P, "15m", "1m", {"15m": 900, "1m": 60}, "ats_test", "hash")


# -- single-timeframe state machine --------------------------------------------------
def t_warmup_gate() -> bool:
    d = det()
    return d.on_closed_bar(view(60, 100.0, inside_run=2, warm=False)) == []


def t_contraction_and_value_line() -> bool:
    d = det()
    assert d.on_closed_bar(view(60, 100.0, inside_run=1)) == []     # only 1 inside bar -> nothing
    out = d.on_closed_bar(view(120, 100.0, inside_run=2,
                                box_high=101.0, box_low=99.0))       # 2 inside bars -> contraction
    return (len(out) == 1 and out[0].phase == "contraction" and out[0].direction is None
            and out[0].value_line == 100.0)                          # midpoint of 99..101


def t_value_line_offset_box() -> bool:
    d = det()
    out = d.on_closed_bar(view(120, 102.0, inside_run=2, box_high=104.0, box_low=100.0))
    return len(out) == 1 and out[0].value_line == 102.0              # midpoint of 100..104


def t_breakout_up() -> bool:
    d = det()
    d.on_closed_bar(view(120, 100.0, inside_run=2, box_high=101.0, box_low=99.0, atr=1.0))
    out = d.on_closed_bar(view(180, 102.0, inside_run=0))            # >101 + 0.25*1 = 101.25 -> up
    return len(out) == 1 and out[0].phase == "breakout" and out[0].direction == "up"


def t_breakout_buffer_blocks() -> bool:
    d = det()
    d.on_closed_bar(view(120, 100.0, inside_run=2, box_high=101.0, box_low=99.0, atr=1.0))
    out = d.on_closed_bar(view(180, 101.1, inside_run=0))            # 101.1 < 101.25 -> no breakout yet
    return out == [] and d.phase is AtsPhase.CONTRACTION


def t_breakout_down() -> bool:
    d = det()
    d.on_closed_bar(view(120, 100.0, inside_run=2, box_high=101.0, box_low=99.0, atr=1.0))
    out = d.on_closed_bar(view(180, 98.0, inside_run=0))             # <99 - 0.25 = 98.75 -> down
    return len(out) == 1 and out[0].phase == "breakout" and out[0].direction == "down"


def t_entry_pullback_up() -> bool:
    d = det()
    d.on_closed_bar(view(120, 100.0, inside_run=2, box_high=101.0, box_low=99.0, atr=1.0))  # value 100
    d.on_closed_bar(view(180, 102.0, inside_run=0))                  # breakout up
    out = d.on_closed_bar(view(240, 99.9, inside_run=0))             # pull back to <= 100 -> entry up
    return (len(out) == 1 and out[0].phase == "entry" and out[0].direction == "up"
            and out[0].bars_since_expansion == 1
            and abs(out[0].dist_from_value_line - 0.1) < 1e-9
            and d.phase is AtsPhase.NO_SIGNAL)                       # one entry per episode -> reset


def t_entry_timeout() -> bool:
    d = det()  # ats_max_entry_bars = 3
    d.on_closed_bar(view(120, 100.0, inside_run=2, box_high=101.0, box_low=99.0, atr=1.0))
    d.on_closed_bar(view(180, 102.0, inside_run=0))                  # breakout up
    outs = [d.on_closed_bar(view(180 + 60 * i, 103.0, inside_run=0)) for i in range(1, 4)]  # never pulls back
    return all(o == [] for o in outs) and d.phase is AtsPhase.NO_SIGNAL


def t_refeed_dedup() -> bool:
    d = det()
    r1 = d.on_closed_bar(view(120, 100.0, inside_run=2))
    r2 = d.on_closed_bar(view(120, 100.0, inside_run=2))            # same epoch re-fed
    return len(r1) == 1 and r2 == []


# -- engine: HTF-bias gate -----------------------------------------------------------
def _set_htf_bias_up(eng: AtsEngine) -> None:
    # HTF contraction sets value_line=99 (midpoint 98..100); the bar close 99.5 > 99 -> bias "up".
    eng.on_snapshot(Snap({"15m": view(900, 99.5, inside_run=2, box_high=100.0, box_low=98.0)}))


def _ltf_up_entry(eng: AtsEngine):
    eng.on_snapshot(Snap({"1m": view(60, 100.0, inside_run=2, box_high=101.0, box_low=99.0, atr=1.0)}))
    eng.on_snapshot(Snap({"1m": view(120, 102.0, inside_run=0)}))   # breakout up
    return eng.on_snapshot(Snap({"1m": view(180, 99.9, inside_run=0)}))  # pullback -> up entry


def _ltf_down_entry(eng: AtsEngine):
    eng.on_snapshot(Snap({"1m": view(60, 100.0, inside_run=2, box_high=101.0, box_low=99.0, atr=1.0)}))
    eng.on_snapshot(Snap({"1m": view(120, 98.0, inside_run=0)}))    # breakout down
    return eng.on_snapshot(Snap({"1m": view(180, 100.1, inside_run=0)}))  # pullback -> down entry


def t_gate_keeps_aligned() -> bool:
    eng = engine()
    _set_htf_bias_up(eng)
    out = _ltf_up_entry(eng)
    entries = [r for r in out if r.phase == "entry"]
    return (len(entries) == 1 and entries[0].direction == "up"
            and entries[0].htf_bias == "up"
            and entries[0].htf_dist_from_value_line is not None)


def t_gate_blocks_counter() -> bool:
    eng = engine()
    _set_htf_bias_up(eng)                          # HTF bias up
    out = _ltf_down_entry(eng)                     # LTF wants a DOWN entry -> must be dropped
    return [r for r in out if r.phase == "entry"] == []


def t_gate_blocks_no_bias() -> bool:
    eng = engine()                                 # HTF never seen -> bias None
    out = _ltf_up_entry(eng)
    return [r for r in out if r.phase == "entry"] == []


def t_engine_context_passthrough() -> bool:
    # Even with no HTF bias, the LTF contraction/breakout CONTEXT records still pass through.
    eng = engine()
    c = eng.on_snapshot(Snap({"1m": view(60, 100.0, inside_run=2, box_high=101.0, box_low=99.0)}))
    return len(c) == 1 and c[0].phase == "contraction"


CHECKS = [
    ("warmup_gate", t_warmup_gate),
    ("contraction_and_value_line", t_contraction_and_value_line),
    ("value_line_offset_box", t_value_line_offset_box),
    ("breakout_up", t_breakout_up),
    ("breakout_buffer_blocks", t_breakout_buffer_blocks),
    ("breakout_down", t_breakout_down),
    ("entry_pullback_up", t_entry_pullback_up),
    ("entry_timeout", t_entry_timeout),
    ("refeed_dedup", t_refeed_dedup),
    ("gate_keeps_aligned", t_gate_keeps_aligned),
    ("gate_blocks_counter", t_gate_blocks_counter),
    ("gate_blocks_no_bias", t_gate_blocks_no_bias),
    ("engine_context_passthrough", t_engine_context_passthrough),
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
        log.info("  %-34s %s", name, "PASS" if ok else "FAIL")
    passed = sum(results.values())
    log.info("%s\n%d/%d checks passed", "=" * 50, passed, len(results))
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
