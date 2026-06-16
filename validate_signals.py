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
