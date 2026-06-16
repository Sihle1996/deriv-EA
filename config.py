"""Configuration for the Deriv data-spine bot, loaded from environment / .env.

Phase 1 is data-only and demo-only. No trading parameters live here yet.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Repo root (this file's directory). data/ lives alongside the code.
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    # --- credentials / connection -------------------------------------------------
    # Phase 1 is data-only. On the legacy v3 API, market data (ticks/candles/active_symbols)
    # is PUBLIC — no authorize needed. We only authenticate for account calls (balance, buy,
    # portfolio) in later phases. So `authenticate` defaults to False and NO token is required
    # to run the spine.
    #
    # NOTE: Deriv migrated new accounts to "pat_"-prefixed Personal Access Tokens for their
    # NEW Options API (REST OTP -> authenticated WS URL). Those tokens do NOT work with this
    # legacy v3 {"authorize": token} call (they return InvalidToken). When we add trading
    # (Phase 3) we'll implement the OTP flow for the pat_ token; until then the token is unused.
    authenticate: bool = os.getenv("DERIV_AUTHENTICATE", "false").lower() == "true"
    api_token: str = os.getenv("DERIV_API_TOKEN", "")
    app_id: str = os.getenv("DERIV_APP_ID", "1089")
    endpoint: str = "wss://ws.derivws.com/websockets/v3"

    # --- market -------------------------------------------------------------------
    symbol: str = os.getenv("DERIV_SYMBOL", "stpRNG")  # Step Index

    # --- timeframes ---------------------------------------------------------------
    # We subscribe to ONE base candle stream (base_granularity, in seconds) plus the
    # raw tick stream. Higher timeframes are resampled in-process from the base frame.
    base_granularity: int = 60  # 1-minute base candles
    # Display/derived timeframes -> pandas resample rule. The base ("1m") resamples
    # to itself, which keeps the snapshot uniform.
    timeframes: dict[str, str] = field(
        default_factory=lambda: {"1m": "1min", "5m": "5min", "15m": "15min"}
    )

    history_count: int = 1000          # base candles to seed on connect (~16h of 1m)
    max_base_rows: int = 5000          # cap base frame to bound memory during the soak

    # --- ATS Master Pattern detector (ats_signals.py) — detect + LOG only, NO trading -------------
    # The ONLY methodology: TradeATS value-line + HTF→LTF pullback. Detector runs on candle CLOSE,
    # consumes MarketSnapshot.views only. See ats_signals.py.
    atr_period: int = 14               # Wilder ATR period (used by the contraction box / breakout buffer)
    signal_flush_every: int = 1        # write-through JSONL (tiny volume; survive ungraceful kills)
    # review_signals.py (offline outcome measurement)
    outcome_horizon_bars: int = 10     # forward window per signal, in bars of its timeframe
    outcome_move_points: float = 0.0   # first-touch barrier; 0 => use 0.5 * atr_at_signal
    # backtest_signals.py (contract-economics simulation)
    bt_payout_ratio: float = 0.95      # Rise/Fall WIN profit per unit stake. ASSUMPTION — real
                                       # payouts vary by symbol/duration (fetch via proposal API later).
    bt_stake: float = 1.0              # stake per simulated trade (unit stake)
    # validate_signals.py (honest statistical validation — treat any edge as a null to disprove)
    n_permutations: int = 2000         # Monte-Carlo null draws for the permutation test
    walk_forward_oos_frac: float = 0.30  # fraction of (time-ordered) trades held out as out-of-sample
    cscv_blocks: int = 10              # S: time blocks for CSCV / Probability of Backtest Overfitting

    ats_signal_version: str = "ats_v2"  # v2 = canonical swing-PIVOT contraction (was inside-bar v1)
    ats_htf: str = os.getenv("ATS_HTF", "5m")    # higher timeframe — sets directional bias. 5m (not
                                                 # 15m) for research THROUGHPUT: 15m so rarely forms a
                                                 # pivot contraction that entries almost never accumulate
                                                 # (1HZ50V: 26h -> 0 entries). 5m defines bias far more
                                                 # often so we reach a testable n. Set ATS_HTF=15m to
                                                 # restore the wider pairing. Does NOT manufacture edge —
                                                 # validate_signals still judges honestly.
    ats_ltf: str = os.getenv("ATS_LTF", "1m")    # lower timeframe — gives the entry
    # CONTRACTION = swing/pivot compression (LuxAlgo-audited): a confirmed pivot-high lower than the
    # prior pivot-high AND a pivot-low higher than the prior pivot-low, over a pivot lookback. This is
    # the canonical Forex Master Pattern definition — selective by nature (few, meaningful boxes).
    ats_pivot_lookback: int = 5                  # bars each side for ta.pivothigh/low (the "Contraction
                                                 # Detection Lookback"); larger = fewer, bigger coils
    ats_breakout_buffer_atr: float = 0.0         # EXPANSION = plain box-boundary break (audited: no ATR
                                                 # buffer). >0 only to experiment with a tolerance.
    ats_pullback_tol_atr: float = 0.5            # ENTRY when LTF close returns within this*ATR of value
                                                 # (our own heuristic — no public ATS entry spec exists)
    # Entry variant (both are UNVERIFIED heuristics — no public ATS entry spec; tested via the harness):
    #   "continuation" = enter the WITH-HTF breakout's pullback to value (our original).
    #   "value_fade"   = enter the COUNTER-HTF expansion spike, in the HTF-bias direction (the
    #                    Google/TradeATS "buy the discount / sell the premium, snap back to value" rule).
    ats_entry_mode: str = os.getenv("ATS_ENTRY_MODE", "continuation")
    ats_max_contraction_bars: int = 60           # abandon a contraction with no breakout within this
    ats_max_entry_bars: int = 20                 # abandon an expansion with no pullback within this
    validate_ats_pivot_lookbacks: tuple = (3, 5, 8, 13)  # PBO sweep — highest-leverage ATS param

    # --- storage ------------------------------------------------------------------
    data_dir: Path = ROOT / "data"
    tick_flush_every: int = 100        # flush the tick buffer to Parquet every N ticks

    # --- reconnection -------------------------------------------------------------
    reconnect_base_delay: float = 1.0  # seconds; exponential backoff start
    reconnect_max_delay: float = 30.0  # backoff ceiling
    # WS keepalive. Worst-case dead-socket detection ≈ ping_interval + ping_timeout, which is
    # also the upper bound on the live-tick gap before reconnect kicks in. Lower = smaller gaps.
    ping_interval: float = 10.0
    ping_timeout: float = 10.0
    # On reconnect, refetch the ticks missed during the outage so the archive stays gap-free.
    # Deriv caps ticks_history at ~5000 rows/request → covers gaps up to ~83 min at 1 tick/s.
    backfill_count: int = 5000

    @property
    def ws_url(self) -> str:
        return f"{self.endpoint}?app_id={self.app_id}"

    @property
    def tick_dir(self) -> Path:
        return self.data_dir / "ticks"

    @property
    def ats_signal_dir(self) -> Path:
        return self.data_dir / "signals_ats"

    @property
    def all_signal_timeframes(self) -> tuple[str, ...]:
        """The timeframes the store must build a TFView for — the ATS HTF + LTF."""
        return tuple(sorted({self.ats_htf, self.ats_ltf}))

    def view_params(self) -> dict:
        """Params the STORE needs to build TFViews (ATR + the swing-pivot contraction box)."""
        return {"atr_period": self.atr_period, "ats_pivot_lookback": self.ats_pivot_lookback}

    def ats_signal_params(self) -> dict:
        """Canonical ATS detector params — single source for AtsEngine and ats_params_hash."""
        return {
            "atr_period": self.atr_period,
            "ats_pivot_lookback": self.ats_pivot_lookback,
            "ats_breakout_buffer_atr": self.ats_breakout_buffer_atr,
            "ats_pullback_tol_atr": self.ats_pullback_tol_atr,
            "ats_max_contraction_bars": self.ats_max_contraction_bars,
            "ats_max_entry_bars": self.ats_max_entry_bars,
            "ats_entry_mode": self.ats_entry_mode,
        }

    def ats_params_hash(self) -> str:
        import hashlib
        import json
        blob = json.dumps(self.ats_signal_params(), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]


CONFIG = Config()
