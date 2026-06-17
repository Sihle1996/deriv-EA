# Pre-registration — OTC_NDX (NAS100) falsification test

> Written and committed **2026-06-17, BEFORE running**, to prevent p-hacking. The deep-archive scan
> (1h→15m, 4 markets) produced one almost-significant result: **OTC_NDX n=15, win 73.3%, ROI +43%,
> permutation p=0.058** — which fails even the uncorrected 0.05 bar and badly fails family-wise. This
> single, frozen test exists to TRY TO FALSIFY it on a larger, lower-timeframe sample. One shot.

## Hypothesis
ATS `value_fade` (the current default detector) has a real forward edge on OTC_NDX.
**Null:** it does not — the deep-archive p=0.058 was small-n / multiple-comparison noise.

## Frozen design (no changes after seeing the result)
- **Market:** OTC_NDX only.
- **Base timeframe:** 5m. Fetch the maximum 5m candle history Deriv serves (chain `end`-cursor
  requests; expected ceiling ~6k bars ≈ the symbol's available history).
- **Ladder:** 1h → 15m → 5m (15m and 1h resampled in-process from the 5m base). Entries on 5m,
  gated by the next-higher TF bias, exactly as in production.
- **Detector:** `ats_value_line_mode=contraction_mean`, `ats_entry_mode=value_fade`. **Params FROZEN**
  at `params_hash = 4b0f8ba6670d`: `atr_period=14`, `ats_pivot_lookback=5`,
  `ats_breakout_buffer_atr=0.0`, `ats_pullback_tol_atr=0.5`, `ats_max_contraction_bars=60`,
  `ats_max_entry_bars=20`.
- **Outcome:** close-to-close (Rise/Fall), horizon = 10 bars (= 50 min on 5m), payout 0.95. Honest
  without ticks; the structural bracket is NOT part of this test.
- **Null model:** random entry bars, matched direction + horizon, same candle series.
- **OOS split:** time-ordered, last 30% held out (`walk_forward_oos_frac=0.30`).

## Success criteria (ALL must hold to call it "interesting")
1. **Permutation p < 0.0125** (family-wise / Bonferroni for the 4 markets NAS100 was selected from).
2. **OOS survival:** in the held-out last 30%, win rate > break-even (51.3%) AND ROI > 0, i.e. the
   in-sample result carries over rather than being front-loaded.

(PBO/CSCV is deliberately **out of scope**: it requires a parameter sweep, and this test forbids
sweeps. OOS survival + family-wise permutation are the operative overfit guards for a single frozen
config.)

## Commitment
- Evaluated **once**. 
- If it **fails** either criterion → conclusion is **NOISE**; we do **NOT** then try 3m/2m, a
  different ladder, a different timeout, or any other config to "rescue" it. That would be p-hacking.
- If it **passes both** → it graduates to a genuine candidate worth a deeper, tick-based study
  (still demo, still no trading).

**Prior (honestly stated up front):** low probability it survives. Most likely outcome = the
framework correctly dissolves an attractive false positive.
