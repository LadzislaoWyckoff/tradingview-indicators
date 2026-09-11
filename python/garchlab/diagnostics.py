"""Specification tests for a fitted GARCH model.

A GARCH fit is only worth forecasting with if its standardized residuals look
like the i.i.d. draws the model claims they are.  The tests here check exactly
that, and they are the difference between a model you can size positions off
and a curve that merely tracks past volatility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "ljung_box",
    "arch_lm",
    "jarque_bera",
    "sign_bias_test",
    "news_impact_curve",
    "diagnose",
]


@dataclass
class TestResult:
    name: str
    statistic: float
    pvalue: float
    df: float
    note: str = ""

    def __str__(self) -> str:
        verdict = "ok" if self.pvalue > 0.05 else "REJECT at 5%"
        return (
            f"{self.name:<34} stat={self.statistic:10.4f}  df={self.df:5.0f}  "
            f"p={self.pvalue:7.4f}  {verdict}"
        )


def ljung_box(x, lags: int = 20, ddof: int = 0, label: str = "") -> TestResult:
    """Ljung-Box test for autocorrelation up to ``lags``.

    Applied to the standardized residuals it checks the mean equation; applied
    to their squares it checks whether the variance equation has soaked up the
    volatility clustering.  ``ddof`` deducts estimated parameters from the
    degrees of freedom.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    x = x - x.mean()
    denom = float(np.dot(x, x))
    stat = 0.0
    for k in range(1, lags + 1):
        rho = float(np.dot(x[k:], x[:-k])) / denom
        stat += rho * rho / (n - k)
    stat *= n * (n + 2)
    df = max(lags - ddof, 1)
    name = f"Ljung-Box({lags})" + (f" on {label}" if label else "")
    return TestResult(name, stat, float(stats.chi2.sf(stat, df)), df,
                      "H0: no autocorrelation")


def arch_lm(resids, lags: int = 12) -> TestResult:
    """Engle's LM test for remaining ARCH effects.

    Run it on the *standardized* residuals of a fitted model: a rejection means
    the variance equation has not captured the clustering and the model needs a
    longer lag structure or a different shape.
    """
    e = np.asarray(resids, dtype=float)
    e = e[np.isfinite(e)]
    e2 = e ** 2
    n = e2.size
    y = e2[lags:]
    x = np.column_stack([e2[lags - k - 1 : n - k - 1] for k in range(lags)])
    x = np.column_stack([np.ones(y.size), x])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if ss_tot == 0 else 1.0 - float(np.sum(resid ** 2)) / ss_tot
    stat = y.size * r2
    return TestResult(
        f"ARCH-LM({lags})", stat, float(stats.chi2.sf(stat, lags)), lags,
        "H0: no remaining ARCH effects",
    )


def jarque_bera(x) -> TestResult:
    """Normality of the standardized residuals.

    Expect this to reject under a normal innovation assumption -- that is the
    whole reason to fit a Student-t or skewed-t instead.  It is informative,
    not disqualifying.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    z = (x - x.mean()) / x.std(ddof=0)
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))
    stat = n / 6.0 * (skew ** 2 + 0.25 * (kurt - 3.0) ** 2)
    return TestResult(
        "Jarque-Bera", stat, float(stats.chi2.sf(stat, 2)), 2,
        f"skew={skew:.3f} kurtosis={kurt:.3f}",
    )


def sign_bias_test(std_resids, lag_resids=None) -> TestResult:
    """Engle-Ng joint sign-bias test.

    Regresses squared standardized residuals on indicators of the previous
    day's sign and on the signed previous shock.  A rejection says the model is
    mispricing the asymmetry -- the usual cure on an equity index is to move
    from GARCH to GJR or EGARCH.
    """
    z = np.asarray(std_resids, dtype=float)
    z = z[np.isfinite(z)]
    source = z if lag_resids is None else np.asarray(lag_resids, dtype=float)
    y = z[1:] ** 2
    prev = source[:-1]
    neg = (prev < 0).astype(float)
    x = np.column_stack([np.ones(y.size), neg, neg * prev, (1.0 - neg) * prev])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if ss_tot == 0 else 1.0 - float(np.sum(resid ** 2)) / ss_tot
    stat = y.size * r2
    return TestResult(
        "Engle-Ng sign bias (joint)", stat, float(stats.chi2.sf(stat, 3)), 3,
        "H0: asymmetry correctly specified",
    )


def news_impact_curve(result, grid=None) -> tuple[np.ndarray, np.ndarray]:
    """Conditional variance tomorrow as a function of today's shock.

    Holds the lagged variance at its unconditional level and sweeps the shock,
    which is the cleanest way to *see* the leverage effect: a symmetric GARCH
    gives a parabola centred on zero, GJR and EGARCH give a curve tilted to the
    left.  Returns ``(shock_in_percent, next_day_annualized_vol)``.
    """
    from .models import EGARCH, GARCH, GJRGARCH

    vm = result.vol_model
    params = np.asarray(result.vol_params, dtype=float)
    h_bar = result.unconditional_variance()
    if not np.isfinite(h_bar):
        h_bar = float(np.mean(result.conditional_variance))
    if grid is None:
        sd = float(np.sqrt(h_bar))
        grid = np.linspace(-5.0 * sd, 5.0 * sd, 401)
    grid = np.asarray(grid, dtype=float)

    p = vm.p
    if isinstance(vm, GJRGARCH):
        omega, alpha, gamma = params[0], params[1], params[1 + p]
        beta = params[1 + 2 * p]
        h_next = omega + (alpha + gamma * (grid < 0)) * grid ** 2 + beta * h_bar
    elif isinstance(vm, EGARCH):
        omega, alpha, gamma = params[0], params[1], params[1 + p]
        beta = params[1 + 2 * p]
        vm.set_dist_state(result.dist, result.dist_params)
        z = grid / np.sqrt(h_bar)
        log_h = omega + alpha * (np.abs(z) - vm._e_abs_z) + gamma * z + beta * np.log(h_bar)
        h_next = np.exp(np.clip(log_h, -30.0, 30.0))
    elif isinstance(vm, GARCH):
        omega, alpha, beta = params[0], params[1], params[1 + p]
        h_next = omega + alpha * grid ** 2 + beta * h_bar
    else:  # pragma: no cover - defensive
        raise TypeError(f"no news impact curve for {type(vm).__name__}")

    vol = np.sqrt(h_next) / result.scale * np.sqrt(result.periods_per_year)
    return grid / result.scale, vol


def diagnose(result, lags: int = 20, arch_lags: int = 12) -> dict[str, TestResult]:
    """Run the whole specification battery on a fitted model."""
    z = result.std_resids
    return {
        "ljung_box_z": ljung_box(z, lags, label="z"),
        "ljung_box_z2": ljung_box(z ** 2, lags, label="z^2"),
        "arch_lm": arch_lm(z, arch_lags),
        "jarque_bera": jarque_bera(z),
        "sign_bias": sign_bias_test(z),
    }
