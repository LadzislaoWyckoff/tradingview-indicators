"""Value-at-Risk and Expected Shortfall from a conditional volatility model,
plus the coverage tests that say whether the numbers can be trusted.

This is where a volatility forecast earns its keep.  A VaR number is only
useful if its exceedances arrive at the right *rate* (Kupiec) and are not
*clustered* (Christoffersen) -- a model that produces the right count of
breaches but delivers them all in one week has told you nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

__all__ = ["value_at_risk", "expected_shortfall", "kupiec_pof",
           "christoffersen_independence", "conditional_coverage",
           "VaRBacktest", "backtest_var"]


def value_at_risk(sigma, alpha: float = 0.01, dist=None, dist_params=(), mean=0.0):
    """One-step VaR as a *negative* return threshold.

    ``sigma`` is the conditional volatility in return units (0.012 for 1.2%).
    The result is the level the return is expected to fall below with
    probability ``alpha``, so it is normally negative.
    """
    from .distributions import get_distribution

    dist = get_distribution("normal") if dist is None else dist
    sigma = np.asarray(sigma, dtype=float)
    q = float(dist.ppf(np.array([alpha]), dist_params)[0])
    return np.asarray(mean, dtype=float) + sigma * q


def expected_shortfall(sigma, alpha: float = 0.01, dist=None, dist_params=(), mean=0.0):
    """Average return conditional on breaching the VaR level."""
    from .distributions import get_distribution

    dist = get_distribution("normal") if dist is None else dist
    sigma = np.asarray(sigma, dtype=float)
    es_z = dist.expected_shortfall(alpha, dist_params)
    return np.asarray(mean, dtype=float) + sigma * es_z


def kupiec_pof(hits, alpha: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures test.  H0: breach rate equals ``alpha``."""
    hits = np.asarray(hits, dtype=float)
    n = hits.size
    x = float(np.sum(hits))
    if n == 0:
        return float("nan"), float("nan")
    pi_hat = x / n
    if x == 0 or x == n:
        # the log-likelihood at the boundary is still well defined
        ll_unrestricted = 0.0
    else:
        ll_unrestricted = x * np.log(pi_hat) + (n - x) * np.log(1.0 - pi_hat)
    ll_restricted = x * np.log(alpha) + (n - x) * np.log(1.0 - alpha)
    stat = 2.0 * (ll_unrestricted - ll_restricted)
    return float(stat), float(stats.chi2.sf(stat, 1))


def christoffersen_independence(hits) -> tuple[float, float]:
    """Christoffersen test for clustering.  H0: breaches are independent.

    Rejection is the dangerous case: it means breaches bunch together, which is
    exactly the pattern that turns a tolerable VaR into a blown account.
    """
    hits = np.asarray(hits, dtype=int)
    if hits.size < 2:
        return float("nan"), float("nan")
    prev, curr = hits[:-1], hits[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    if n01 + n11 == 0 or n00 + n01 == 0 or n10 + n11 == 0:
        return 0.0, 1.0
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def _ll(p, k, m):
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return k * np.log(p) + m * np.log(1.0 - p)

    ll_markov = _ll(pi01, n01, n00) + _ll(pi11, n11, n10)
    ll_indep = _ll(pi, n01 + n11, n00 + n10)
    stat = 2.0 * (ll_markov - ll_indep)
    return float(stat), float(stats.chi2.sf(stat, 1))


def conditional_coverage(hits, alpha: float) -> tuple[float, float]:
    """Joint test of correct rate *and* independence (Christoffersen CC)."""
    pof, _ = kupiec_pof(hits, alpha)
    ind, _ = christoffersen_independence(hits)
    stat = pof + ind
    if not np.isfinite(stat):
        return float("nan"), float("nan")
    return float(stat), float(stats.chi2.sf(stat, 2))


@dataclass
class VaRBacktest:
    alpha: float
    n: int
    n_breaches: int
    expected_breaches: float
    breach_rate: float
    kupiec: tuple[float, float]
    independence: tuple[float, float]
    conditional_coverage: tuple[float, float]
    es_backtest: dict = field(default_factory=dict)
    hits: np.ndarray = field(default_factory=lambda: np.empty(0))

    def __str__(self) -> str:
        lines = [
            f"VaR backtest at alpha = {self.alpha:.1%}  ({self.n} out-of-sample days)",
            f"  breaches: {self.n_breaches} observed vs {self.expected_breaches:.1f} expected"
            f"   (rate {self.breach_rate:.2%})",
            f"  Kupiec POF          stat={self.kupiec[0]:8.4f}  p={self.kupiec[1]:.4f}"
            f"   {'ok' if self.kupiec[1] > 0.05 else 'REJECT: wrong breach rate'}",
            f"  Independence        stat={self.independence[0]:8.4f}  p={self.independence[1]:.4f}"
            f"   {'ok' if self.independence[1] > 0.05 else 'REJECT: breaches cluster'}",
            f"  Conditional cover.  stat={self.conditional_coverage[0]:8.4f}"
            f"  p={self.conditional_coverage[1]:.4f}"
            f"   {'ok' if self.conditional_coverage[1] > 0.05 else 'REJECT'}",
        ]
        if self.es_backtest:
            lines.append(
                f"  ES on breach days: realized {self.es_backtest['realized']:.4%} "
                f"vs predicted {self.es_backtest['predicted']:.4%} "
                f"(ratio {self.es_backtest['ratio']:.2f}; >1 means ES is too optimistic)"
            )
        return "\n".join(lines)


def backtest_var(returns, var_levels, es_levels=None, alpha: float = 0.01) -> VaRBacktest:
    """Score a sequence of one-step VaR forecasts against what happened.

    ``returns`` and ``var_levels`` must be aligned: ``var_levels[i]`` is the
    forecast made at ``i-1`` for day ``i``.
    """
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var_levels, dtype=float)
    if r.shape != v.shape:
        raise ValueError(f"returns {r.shape} and var_levels {v.shape} must align")
    valid = np.isfinite(r) & np.isfinite(v)
    r, v = r[valid], v[valid]
    hits = (r < v).astype(int)
    n = hits.size

    es_report: dict = {}
    if es_levels is not None:
        e = np.asarray(es_levels, dtype=float)[valid]
        if hits.sum() > 0:
            realized = float(np.mean(r[hits == 1]))
            predicted = float(np.mean(e[hits == 1]))
            es_report = {
                "realized": realized,
                "predicted": predicted,
                "ratio": realized / predicted if predicted != 0 else float("nan"),
                "n": int(hits.sum()),
            }

    return VaRBacktest(
        alpha=alpha,
        n=n,
        n_breaches=int(hits.sum()),
        expected_breaches=alpha * n,
        breach_rate=float(hits.mean()) if n else float("nan"),
        kupiec=kupiec_pof(hits, alpha),
        independence=christoffersen_independence(hits),
        conditional_coverage=conditional_coverage(hits, alpha),
        es_backtest=es_report,
        hits=hits,
    )
