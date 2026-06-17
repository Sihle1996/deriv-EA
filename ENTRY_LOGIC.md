# ENTRY_LOGIC.md — how ATS entries are triggered (versions + the two distinct hypotheses)

> Durable record of the entry-trigger rules, their version history, and — critically — the
> **two philosophically different hypotheses** the engine can run. Written so that six months from
> now we remember not just the *code* but the *hypothesis behind each mode*. Detect-and-log only —
> **no execution code exists.** On a CSPRNG synthetic no edge is possible by construction; the point
> of being faithful is so the validation harness can *measure honestly*, not to imply profit.

## ⚠️ Honest labelling (read first)
No public ATS source specifies entry/stop/target rules (those are paid-course/marketing material),
so **everything below the contraction/value-line/expansion layer is OUR heuristic, not "the" ATS
rule.** More importantly, the current **default has drifted from a trend-following reading of ATS
into a mean-reversion model.** Keep the two distinguishable in code, docs, and validation reports:

| Name | Family | One-line | `ats_entry_mode` |
|------|--------|----------|------------------|
| **ATS-VALUE-FADE** (current default) | **mean reversion** | fade the breakout spike back to value | `value_fade` |
| **ATS-CONTINUATION** | **trend following** | enter the pullback to value, with the breakout | `continuation` |

Both run under schema `ats_v3` and are distinguished at runtime by `ats_entry_mode`, which is part of
`ats_params_hash()` — so their logged records never get conflated. **Do not refer to either as just
"ATS"** once results are reported; they are different research questions.

## Version history
- **v1 — inside-bar (removed).** Contraction = consecutive inside bars (lower-high AND higher-low);
  → breakout → pullback → continuation. Far too many boxes (~100/archive); not selective. Discarded.
- **v2 — swing-pivot.** Contraction = a swing-pivot compression (LuxAlgo-audited `ta.pivothigh/low`):
  a confirmed pivot-high lower than the prior AND pivot-low higher than the prior. Value line = box
  **midpoint** `(high+low)/2`. Selective (~39/archive). Entry default = `continuation`.
- **v3 — current.** Two faithfulness changes from the training transcript: (a) value line = **mean of
  closes inside the contraction** ("average price established during the contraction"), not the
  midpoint; (b) default entry flipped to **`value_fade`**. Plus a **timeframe LADDER `1h→15m→5m→1m`**
  (each TF's entries gated by the one above) and a structural **`stop_ref`** captured per entry.

## The shared pipeline (both modes)
Runs **on candle close**, per timeframe, on plain floats from a `TFView`.

1. **Contraction (arms the setup).** `candles._compute_view` confirms a swing-pivot compression
   (`ats_pivot_lookback=5` bars each side; latest pivot-high < prior pivot-high AND latest pivot-low >
   prior pivot-low). Freezes the **box** (`box_high/box_low`), the **value line** (mean of box closes;
   `ats_value_line_mode="contraction_mean"`, "midpoint" optional), and ATR (Wilder, `atr_period=14`).
   `NO_SIGNAL → CONTRACTION`.
2. **Expansion (breakout).** `close > box_high + buffer` (up) or `close < box_low - buffer` (down);
   `ats_breakout_buffer_atr=0` → a plain boundary break. Abandoned after `ats_max_contraction_bars=60`.
3. **Entry candidate** — mode-specific (see below). Emits direction + entry price + `stop_ref`
   (expansion swing extreme) + value line (target).
4. **HTF-bias gate (what makes it tradeable).** The candidate is kept **only if its direction equals
   the next-higher TF's bias** = which side of the *higher* TF's value line the *higher* TF's last
   close sits on. Agree → logged as **`entry`** (tradeable, stamped `htf_bias`); no-bias/counter →
   **`entry_blocked`** (funnel only, not traded). The top rung (1h) has no higher TF → its entries are
   dropped. **One entry per contraction episode**, then it waits for the next contraction.

## Mode A — ATS-VALUE-FADE (default, MEAN REVERSION)
**Mechanical rule:** the instant the box breaks, emit an entry candidate in the **OPPOSITE** direction
of the break, at the breakout bar's close (no wait). Up break → **short** candidate; down break →
**long** candidate. Target = the value line; stop = just beyond the expansion swing extreme.
Because of the HTF gate, a kept entry means **the break was *counter* to the higher-TF bias** (a
counter-trend liquidity spike) and you enter *with* the HTF bias by fading that spike.

**Hypothesis (why it should work):** breakouts away from equilibrium often *overshoot* (a two-sided
liquidity sweep before the real move); price reverts toward fair value; the higher-timeframe bias
filters *which* reversions to take (only fade spikes that go against the bigger trend).

**Honest caveat:** this is a **mean-reversion** model and is **not** the public trend-following ATS
picture. It is a defensible quant hypothesis, but label it as ATS-VALUE-FADE, never plain "ATS."

## Mode B — ATS-CONTINUATION (TREND FOLLOWING)
**Mechanical rule:** on breakout, enter EXPANSION and **wait** for price to pull back to the value
line (`close` within `ats_pullback_tol_atr=0.5 × ATR` of it) in the breakout direction, then emit a
candidate **in the breakout direction**. Abandoned if no pullback within `ats_max_entry_bars=20`.
Kept only if the break direction agrees with the HTF bias (so: with-trend breakout + with-trend HTF).

**Hypothesis (why it should work):** an expansion reveals directional intent; the pull-back to value
offers a lower-risk entry to *continue* in the expansion direction, aligned with the higher timeframe.

**Honest caveat:** closer to the public ATS reading, but the specific pullback trigger/tolerance is
still our heuristic, not a published rule.

## Worked example (real logged trade — `value_fade`, frxUSDJPY 1m)
Contraction box `[160.430, 160.443]`, value line `160.4337`. Price broke **up** to `160.444` →
`value_fade` emitted a **short** at 160.444 (stop_ref `160.451`, target = value `160.4337`). The **5m
bias was down**, so the short agreed with HTF bias and was **kept** as a tradeable `entry`. It then
lost (price didn't revert) — irrelevant at n=1; the value is that the **path is fully auditable**:
box → break → fade → gate → logged entry.

## Parameters (config.py)
`ats_ladder = 1h→15m→5m→1m` · `ats_pivot_lookback=5` · `atr_period=14` · `ats_breakout_buffer_atr=0`
· `ats_value_line_mode="contraction_mean"` · `ats_entry_mode="value_fade"` (default) ·
`ats_pullback_tol_atr=0.5` · `ats_max_contraction_bars=60` · `ats_max_entry_bars=20`. Bracket
(structural test): SL=`stop_ref`, TP1=value line (partial + break-even runner), TP2=box far side.

## Status / discipline
Both modes are **research configurations**, judged only by `validate_signals.py` (permutation /
walk-forward / PBO / deflated Sharpe) once a real sample exists. We are in **Phase A (collect)**; the
default stays `value_fade`. The continuation-vs-value-fade comparison is a **Phase C** question
(n>100 real entries) — not to be tuned or pre-judged now. See CLAUDE.md "Research roadmap".
