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
