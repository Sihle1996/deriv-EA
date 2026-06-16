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
