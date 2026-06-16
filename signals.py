"""Signal record schema — the one logged unit shared by the ATS detector and the review tools.

The ATS Master Pattern detector lives in ats_signals.py; this module only owns the SignalRecord
dataclass + schema version (kept separate so ats_signals.py stays free of any storage concern).

A record carries everything needed to recompute forward outcomes WITHOUT re-running the detector.
`bar_close_epoch` is the look-ahead firewall: the earliest instant a forward-outcome measurement
may inspect.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

SCHEMA_VERSION = 2  # v2 = ATS-only schema (dropped the legacy Bollinger band-width fields)


@dataclass(frozen=True)
class SignalRecord:
    schema_version: int
    signal_version: str
    detected_at_epoch: int
    symbol: str
    timeframe: str
    phase: str                      # "contraction" | "breakout" | "entry" | "entry_blocked"
    direction: str | None           # "up" | "down" | None
    bar_epoch: int                  # trigger bar open-time (dedup key with timeframe+phase)
    bar_close_epoch: int            # bar_epoch + tf_seconds — look-ahead firewall
    price_at_signal: float          # trigger bar close (outcome baseline)
    atr: float | None
    atr_at_contraction: float | None
    contraction_high: float | None  # the ATS contraction box high
    contraction_low: float | None   # the ATS contraction box low
    contraction_bars: int | None    # bars elapsed in the phase that led here
    episode_id: str                 # ties contraction → breakout → entry of one cycle
    params_hash: str
    # ATS value-line + entry context. value_line = midpoint of the frozen contraction box;
    # htf_bias = HTF side of its own value line at the LTF entry. The *_metadata fields are for
    # RESEARCH analysis only — any FILTER derived from them is a new config (must enter the PBO
    # sweep, never a post-hoc cherry-pick).
    value_line: float | None = None
    htf_bias: str | None = None
    dist_from_value_line: float | None = None
    bars_since_expansion: int | None = None
    htf_dist_from_value_line: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)
