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
