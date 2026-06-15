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

    # --- Phase 2: signal research engine (detect + LOG only; NO trading) ----------
    # Detector runs on candle CLOSE, consumes MarketSnapshot.views only. See signals.py.
    signal_version: str = "phase2_v1"  # human-readable rule tag; bump when logic/defaults change
    signal_timeframes: tuple[str, ...] = ("1m", "5m")
    atr_period: int = 14               # Wilder ATR period
    bb_window: int = 20                # Bollinger window for band-width volatility measure
    bb_std: float = 2.0               # Bollinger std multiplier
    vol_lookback: int = 100            # window for the band-width percentile/z-score baseline
    contraction_pct: float = 0.20      # enter CONTRACTION when bw_percentile <= this
    contraction_exit_pct: float = 0.40 # hysteresis: re-arm only after bw_percentile rises above this
    contraction_range_bars: int = 20   # bars used to freeze the breakout range at contraction entry
    breakout_atr_mult: float = 1.0     # EXPANSION when close breaks range by N*atr_at_contraction
    max_contraction_bars: int = 60     # abandon a contraction (no signal) if no breakout within this
    signal_flush_every: int = 1        # write-through JSONL (tiny volume; survive ungraceful kills)
    # review_signals.py (offline outcome measurement)
    outcome_horizon_bars: int = 10     # forward window per signal, in bars of its timeframe
    outcome_move_points: float = 0.0   # first-touch barrier; 0 => use 0.5 * atr_at_signal
    # backtest_signals.py (contract-economics simulation)
    bt_payout_ratio: float = 0.95      # Rise/Fall WIN profit per unit stake. ASSUMPTION — real
                                       # payouts vary by symbol/duration (fetch via proposal API later).
    bt_stake: float = 1.0              # stake per simulated trade (unit stake)

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
    def signal_dir(self) -> Path:
        return self.data_dir / "signals"

    def signal_params(self) -> dict:
        """Canonical detector params — the single source for both SignalEngine and params_hash.
        Order matters for the hash; keep it stable."""
        return {
            "atr_period": self.atr_period,
            "bb_window": self.bb_window,
            "bb_std": self.bb_std,
            "vol_lookback": self.vol_lookback,
            "contraction_pct": self.contraction_pct,
            "contraction_exit_pct": self.contraction_exit_pct,
            "contraction_range_bars": self.contraction_range_bars,
            "breakout_atr_mult": self.breakout_atr_mult,
            "max_contraction_bars": self.max_contraction_bars,
        }

    def params_hash(self) -> str:
        """Short stable fingerprint of the detector params, recorded on every signal."""
        import hashlib
        import json
        blob = json.dumps(self.signal_params(), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]


CONFIG = Config()
