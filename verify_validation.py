"""Test the tester — verify the statistical core of validate_signals.py on inputs with KNOWN truth.

If these pass, the permutation test, CSCV/PBO, and deflated-Sharpe math are correct independent of
any live data — so when they say "no edge" on the bot's signals, that verdict is trustworthy.

Run:  python verify_validation.py
"""
from __future__ import annotations

import logging

import numpy as np

from validate_signals import perm_pvalue, cscv_pbo, expected_max_sharpe, sharpe

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("verify_validation")


def t_perm_significant() -> bool:
    # observed far above the null distribution -> small p
    return perm_pvalue(200.0, list(range(100))) < 0.05


def t_perm_not_significant() -> bool:
    # observed at the null median -> p ~ 0.5
    p = perm_pvalue(49.5, list(range(100)))
    return 0.4 < p < 0.65


def t_perm_worse_than_null() -> bool:
    # observed below everything -> p ~ 1.0
    return perm_pvalue(-1.0, list(range(100))) > 0.95


def t_pbo_genuine_edge_low() -> bool:
    # config 0 is best in EVERY block -> IS-best generalizes OOS -> PBO ~ 0
    M = np.ones((10, 8)); M[:, 0] = 5.0
    return cscv_pbo(M) < 0.1


def t_pbo_random_midrange() -> bool:
    # no config is genuinely better -> PBO ~ 0.5 (indistinguishable from luck)
    M = np.random.RandomState(0).randn(10, 8)
    pbo = cscv_pbo(M)
    return 0.2 < pbo < 0.8


def t_pbo_guards_bad_shape() -> bool:
    import math
    return math.isnan(cscv_pbo(np.zeros((3, 4))))  # odd #blocks -> NaN, not a crash


def t_expmax_increases_with_trials() -> bool:
    # same cross-trial Sharpe variance, more trials -> higher expected-max hurdle
    small = expected_max_sharpe([-1.0, 1.0])               # N=2
    large = expected_max_sharpe([-1.0, 1.0] * 50)          # N=100, same variance
    return large > small > 0


def t_expmax_zero_variance() -> bool:
    return expected_max_sharpe([0.5] * 6) == 0.0


def t_sharpe_basic() -> bool:
    s = sharpe([1.0, 1.0, 1.0])      # zero variance -> 0.0
    import math
    return s == 0.0 and math.isnan(sharpe([1.0]))


CHECKS = [
    ("perm_significant", t_perm_significant),
    ("perm_not_significant", t_perm_not_significant),
    ("perm_worse_than_null", t_perm_worse_than_null),
    ("pbo_genuine_edge_low", t_pbo_genuine_edge_low),
    ("pbo_random_midrange", t_pbo_random_midrange),
    ("pbo_guards_bad_shape", t_pbo_guards_bad_shape),
    ("expmax_increases_with_trials", t_expmax_increases_with_trials),
    ("expmax_zero_variance", t_expmax_zero_variance),
    ("sharpe_basic", t_sharpe_basic),
]


def main() -> None:
    results = {}
    for name, fn in CHECKS:
        try:
            results[name] = bool(fn())
        except Exception as e:
            results[name] = False
            log.info("  [ERROR] %s: %s", name, e)
    log.info("%s", "=" * 50)
    for name, ok in results.items():
        log.info("  %-32s %s", name, "PASS" if ok else "FAIL")
    passed = sum(results.values())
    log.info("%s\n%d/%d checks passed", "=" * 50, passed, len(results))
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
