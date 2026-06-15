"""Phase 2 detector regression — offline, deterministic, PASS/FAIL (like verify_feed.py).

Two groups:
  A. State machine (signals.py) fed hand-built TFViews — transitions, warm-up gate, timeout,
     hysteresis, dedup. No pandas, no network.
  B. Store indicator layer (candles.py) on a synthesised base frame — warm-up threshold and the
     closed-bar-only / partial-5m-bar exclusion.

Run:  python verify_signals.py
"""
from __future__ import annotations

import logging

from candles import MultiTimeframeStore, TFView, _compute_view
from config import CONFIG
from signals import SignalEngine, TimeframeDetector

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("verify_signals")

# Small, explicit params for the state-machine tests.
P = {
    "atr_period": 14, "bb_window": 20, "bb_std": 2.0, "vol_lookback": 100,
    "contraction_pct": 0.20, "contraction_exit_pct": 0.40,
    "contraction_range_bars": 20, "breakout_atr_mult": 1.0, "max_contraction_bars": 3,
    "trend_continue_atr": 1.0, "trend_max_bars": 5,
}


def view(epoch, close, *, bw_pct, rhigh=101.0, rlow=99.0, atr=1.0, warm=True):
    """Build a warm (or non-warm) TFView with the fields the detector reads."""
    return TFView(
        tf="1m", closed_bar_epoch=epoch, close=close, high=close, low=close, n_bars=200,
        atr=atr if warm else None,
        band_width=0.01 if warm else None,
        bw_threshold=0.01 if warm else None,
        bw_percentile=bw_pct if warm else None,
        bbw_zscore=0.0 if warm else None,
        range_high=rhigh if warm else None,
        range_low=rlow if warm else None,
    )


def det() -> TimeframeDetector:
    return TimeframeDetector("TEST", "1m", 60, P, "test_v", "hash")


# -- Group A: state machine ----------------------------------------------------------
def t_warmup_gate() -> bool:
    d = det()
    r = d.on_closed_bar(view(60, 100.0, bw_pct=0.1, warm=False))
    return r is None  # not warm -> nothing


def t_contraction_entry() -> bool:
    d = det()
    assert d.on_closed_bar(view(60, 100.0, bw_pct=0.50)) is None   # high vol -> no signal
    r = d.on_closed_bar(view(120, 100.0, bw_pct=0.10))             # low tail -> contraction
    return r is not None and r.phase == "contraction" and r.direction is None and r.contraction_bars == 0


def t_expansion_up() -> bool:
    d = det()
    d.on_closed_bar(view(120, 100.0, bw_pct=0.10, rhigh=101.0, rlow=99.0, atr=1.0))  # freeze 99..101, atr1
    r = d.on_closed_bar(view(180, 102.5, bw_pct=0.10))            # >101 + 1*1 = 102 -> breakout up
    return r is not None and r.phase == "expansion" and r.direction == "up" and r.contraction_bars == 1


def t_expansion_down() -> bool:
    d = det()
    d.on_closed_bar(view(120, 100.0, bw_pct=0.10, rhigh=101.0, rlow=99.0, atr=1.0))
    r = d.on_closed_bar(view(180, 97.5, bw_pct=0.10))             # <99 - 1*1 = 98 -> breakout down
    return r is not None and r.phase == "expansion" and r.direction == "down"


def t_timeout_no_expansion() -> bool:
    d = det()  # max_contraction_bars = 3
    d.on_closed_bar(view(120, 100.0, bw_pct=0.10, rhigh=101.0, rlow=99.0))
    outs = [d.on_closed_bar(view(120 + 60 * i, 100.0, bw_pct=0.10)) for i in range(1, 4)]  # no breakout
    # No expansion emitted, and detector returned to NO_SIGNAL (disarmed).
    from signals import Phase
    return all(o is None for o in outs) and d.phase is Phase.NO_SIGNAL


def t_trend_continuation() -> bool:
    d = det()
    d.on_closed_bar(view(120, 100.0, bw_pct=0.10, rhigh=101.0, rlow=99.0, atr=1.0))  # contraction
    d.on_closed_bar(view(180, 102.5, bw_pct=0.10))               # expansion up (exp_close=102.5) -> TREND
    r = d.on_closed_bar(view(240, 104.0, bw_pct=0.10))           # >= 102.5 + 1*1 = 103.5 -> trend up
    return r is not None and r.phase == "trend" and r.direction == "up"


def t_reversal() -> bool:
    d = det()
    d.on_closed_bar(view(120, 100.0, bw_pct=0.10, rhigh=101.0, rlow=99.0, atr=1.0))  # contraction
    d.on_closed_bar(view(180, 102.5, bw_pct=0.10))               # expansion up -> TREND
    r = d.on_closed_bar(view(240, 100.5, bw_pct=0.10))           # < contraction_high 101 -> reversal down
    return r is not None and r.phase == "reversal" and r.direction == "down"


def t_hysteresis() -> bool:
    d = det()
    d.on_closed_bar(view(120, 100.0, bw_pct=0.10, rhigh=101.0, rlow=99.0, atr=1.0))  # contraction
    d.on_closed_bar(view(180, 102.5, bw_pct=0.10))               # expansion up -> TREND
    d.on_closed_bar(view(240, 100.0, bw_pct=0.10))               # retrace -> reversal, episode ends (disarmed)
    # A still-low bar must NOT immediately re-fire a contraction (hysteresis).
    r_low = d.on_closed_bar(view(300, 100.0, bw_pct=0.10))
    d.on_closed_bar(view(360, 100.0, bw_pct=0.50))               # vol climbs above exit_pct -> re-arm
    r_re = d.on_closed_bar(view(420, 100.0, bw_pct=0.10, rhigh=101.0, rlow=99.0))  # now contraction allowed
    return r_low is None and r_re is not None and r_re.phase == "contraction"


def t_refeed_dedup() -> bool:
    d = det()
    r1 = d.on_closed_bar(view(120, 100.0, bw_pct=0.10))
    r2 = d.on_closed_bar(view(120, 100.0, bw_pct=0.10))          # same bar epoch re-fed
    return r1 is not None and r2 is None


def t_engine_dedup() -> bool:
    eng = SignalEngine("TEST", P, ("1m",), {"1m": 60}, "test_v", "hash")

    class Snap:  # minimal stand-in with a .views dict
        def __init__(self, v):
            self.views = {"1m": v}

    eng.on_snapshot(Snap(view(60, 100.0, bw_pct=0.50)))
    first = eng.on_snapshot(Snap(view(120, 100.0, bw_pct=0.10)))   # contraction
    dup = eng.on_snapshot(Snap(view(120, 100.0, bw_pct=0.10)))     # identical -> deduped
    return len(first) == 1 and len(dup) == 0


# -- Group B: store indicator layer --------------------------------------------------
def _synth(n, start=0, step=60, base=100.0, amp=0.3, seed=1):
    import random
    rng = random.Random(seed)
    out, price = [], base
    for i in range(n):
        o = price
        c = price + rng.uniform(-amp, amp)
        out.append({"epoch": start + i * step, "open": o, "high": max(o, c) + amp,
                    "low": min(o, c) - amp, "close": c})
        price = c
    return out


def _store(candles):
    s = MultiTimeframeStore(CONFIG.symbol, CONFIG.timeframes, base_granularity=60)
    s.load_history(candles)
    return s


def t_warmup_threshold() -> bool:
    min_bars = max(CONFIG.atr_period + 1, CONFIG.bb_window + CONFIG.vol_lookback)  # 120
    cold = _compute_view(_store(_synth(50)).closed_frame("1m"), "1m", CONFIG.signal_params())
    warm = _compute_view(_store(_synth(min_bars + 10)).closed_frame("1m"), "1m", CONFIG.signal_params())
    return cold is not None and not cold.warm and warm is not None and warm.warm


def t_closed_frame_drops_forming() -> bool:
    s = _store(_synth(5))                       # 5 base bars; last is "forming"
    return len(s.closed_frame("1m")) == 4


def t_closed_frame_excludes_partial_5m() -> bool:
    # 1003 bars from epoch 0 (a 5min boundary): drop forming -> 1002 -> 200 full 5m groups + a
    # 2-bar partial that must be excluded.
    s = _store(_synth(1003, start=0))
    return len(s.closed_frame("5m")) == 200


CHECKS = [
    ("warmup_gate", t_warmup_gate),
    ("contraction_entry", t_contraction_entry),
    ("expansion_up", t_expansion_up),
    ("expansion_down", t_expansion_down),
    ("timeout_no_expansion", t_timeout_no_expansion),
    ("trend_continuation", t_trend_continuation),
    ("reversal", t_reversal),
    ("hysteresis", t_hysteresis),
    ("refeed_dedup", t_refeed_dedup),
    ("engine_dedup", t_engine_dedup),
    ("warmup_threshold", t_warmup_threshold),
    ("closed_frame_drops_forming", t_closed_frame_drops_forming),
    ("closed_frame_excludes_partial_5m", t_closed_frame_excludes_partial_5m),
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
