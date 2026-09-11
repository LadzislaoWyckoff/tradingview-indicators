"""Standardized innovation distributions for GARCH-type models.

Every distribution here is *standardized*: E[z] = 0 and Var[z] = 1.  That is
what lets the variance recursion own the scale, so the shape parameters only
describe tail thickness and asymmetry.

Implemented
-----------
``normal``  no shape parameters.
``t``       Student-t standardized to unit variance, nu > 2.
``skewt``   Hansen (1994) skewed Student-t, nu > 2 and -1 < lambda < 1.

Each distribution exposes the same four pieces the estimator and the risk
layer need: ``loglikelihood``, ``cdf``, ``ppf`` and ``simulate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import special, stats

__all__ = ["Normal", "StudentT", "SkewT", "get_distribution"]

_LOG_2PI = float(np.log(2.0 * np.pi))


class Distribution:
    """Interface shared by the standardized innovation distributions."""

    name: str = "base"
    #: human readable names of the shape parameters, in order
    param_names: tuple[str, ...] = ()

    def starting_params(self) -> np.ndarray:
        return np.empty(0)

    def bounds(self) -> list[tuple[float, float]]:
        return []

    def loglikelihood(self, z: np.ndarray, params: Sequence[float]) -> np.ndarray:
        """Per-observation log density of the standardized innovations."""
        raise NotImplementedError

    def cdf(self, z: np.ndarray, params: Sequence[float]) -> np.ndarray:
        raise NotImplementedError

    def ppf(self, q: np.ndarray, params: Sequence[float]) -> np.ndarray:
        raise NotImplementedError

    def simulate(self, size, params: Sequence[float], rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    # -- shared helpers ---------------------------------------------------
    def prob_negative(self, params: Sequence[float]) -> float:
        """P(z < 0).  GJR needs it to turn the leverage term into a forecast."""
        return float(self.cdf(np.array([0.0]), params)[0])

    def mean_abs(self, params: Sequence[float], nodes: int = 512) -> float:
        """E|z| for the standardized distribution.  EGARCH centres on it."""
        x, w = np.polynomial.legendre.leggauss(nodes)
        u = 0.5 * (x + 1.0)
        u = np.clip(u, 1e-10, 1.0 - 1e-10)
        return float(np.sum(w * np.abs(self.ppf(u, params))) * 0.5)

    def expected_shortfall(self, q: float, params: Sequence[float], nodes: int = 512) -> float:
        """Left-tail ES at level ``q`` of the standardized distribution.

        ES(q) = (1/q) * integral_0^q F^{-1}(u) du, evaluated with Gauss-Legendre
        nodes so it works for every distribution here, skewed ones included.
        """
        if not 0.0 < q < 1.0:
            raise ValueError("q must lie in (0, 1)")
        x, w = np.polynomial.legendre.leggauss(nodes)
        # map the [-1, 1] Legendre nodes onto (0, q)
        u = 0.5 * q * (x + 1.0)
        return float(np.sum(w * self.ppf(u, params)) * 0.5)


@dataclass
class Normal(Distribution):
    name: str = "normal"
    param_names: tuple[str, ...] = ()

    def loglikelihood(self, z, params=()):
        z = np.asarray(z, dtype=float)
        return -0.5 * (_LOG_2PI + z * z)

    def mean_abs(self, params=(), nodes: int = 512):
        return float(np.sqrt(2.0 / np.pi))

    def cdf(self, z, params=()):
        return stats.norm.cdf(np.asarray(z, dtype=float))

    def ppf(self, q, params=()):
        return stats.norm.ppf(np.asarray(q, dtype=float))

    def simulate(self, size, params=(), rng=None):
        rng = np.random.default_rng() if rng is None else rng
        return rng.standard_normal(size)


@dataclass
class StudentT(Distribution):
    name: str = "t"
    param_names: tuple[str, ...] = ("nu",)

    def starting_params(self):
        return np.array([8.0])

    def bounds(self):
        # nu <= 2 has no variance and nu just above it has no finite kurtosis;
        # no equity return series lives there, and letting the optimizer visit
        # that corner invites a degenerate "all tail, no dynamics" solution
        return [(2.2, 300.0)]

    def loglikelihood(self, z, params):
        z = np.asarray(z, dtype=float)
        nu = float(params[0])
        const = (
            special.gammaln(0.5 * (nu + 1.0))
            - special.gammaln(0.5 * nu)
            - 0.5 * np.log(np.pi * (nu - 2.0))
        )
        return const - 0.5 * (nu + 1.0) * np.log1p(z * z / (nu - 2.0))

    def mean_abs(self, params, nodes: int = 512):
        # closed form; the base-class quadrature would otherwise sit inside
        # the EGARCH likelihood loop
        nu = float(params[0])
        return float(
            2.0
            * np.sqrt(nu - 2.0)
            * np.exp(special.gammaln(0.5 * (nu + 1.0)) - special.gammaln(0.5 * nu))
            / ((nu - 1.0) * np.sqrt(np.pi))
        )

    def cdf(self, z, params):
        nu = float(params[0])
        return stats.t.cdf(np.asarray(z, dtype=float) * np.sqrt(nu / (nu - 2.0)), df=nu)

    def ppf(self, q, params):
        nu = float(params[0])
        return stats.t.ppf(np.asarray(q, dtype=float), df=nu) * np.sqrt((nu - 2.0) / nu)

    def simulate(self, size, params, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        nu = float(params[0])
        return rng.standard_t(nu, size=size) * np.sqrt((nu - 2.0) / nu)


@dataclass
class SkewT(Distribution):
    """Hansen (1994) skewed Student-t, standardized to zero mean / unit variance.

    Parameters are ``nu`` (tail thickness, > 2) and ``lam`` (asymmetry, in
    (-1, 1)).  Negative ``lam`` means a fatter left tail, which is the usual
    shape for equity index returns.
    """

    name: str = "skewt"
    param_names: tuple[str, ...] = ("nu", "lambda")

    def starting_params(self):
        return np.array([8.0, -0.05])

    def bounds(self):
        return [(2.2, 300.0), (-0.95, 0.95)]

    @staticmethod
    def _abc(nu: float, lam: float) -> tuple[float, float, float]:
        log_c = (
            special.gammaln(0.5 * (nu + 1.0))
            - special.gammaln(0.5 * nu)
            - 0.5 * np.log(np.pi * (nu - 2.0))
        )
        c = float(np.exp(log_c))
        a = 4.0 * lam * c * (nu - 2.0) / (nu - 1.0)
        b = float(np.sqrt(1.0 + 3.0 * lam * lam - a * a))
        return a, b, log_c

    def loglikelihood(self, z, params):
        z = np.asarray(z, dtype=float)
        nu, lam = float(params[0]), float(params[1])
        a, b, log_c = self._abc(nu, lam)
        # the density switches branch at the mode of the standardized variable
        denom = np.where(z < -a / b, 1.0 - lam, 1.0 + lam)
        kernel = (b * z + a) / denom
        return (
            log_c
            + np.log(b)
            - 0.5 * (nu + 1.0) * np.log1p(kernel * kernel / (nu - 2.0))
        )

    def cdf(self, z, params):
        z = np.asarray(z, dtype=float)
        nu, lam = float(params[0]), float(params[1])
        a, b, _ = self._abc(nu, lam)
        scale = np.sqrt(nu / (nu - 2.0))
        left = (1.0 - lam) * stats.t.cdf((b * z + a) / (1.0 - lam) * scale, df=nu)
        right = (1.0 - lam) / 2.0 + (1.0 + lam) * (
            stats.t.cdf((b * z + a) / (1.0 + lam) * scale, df=nu) - 0.5
        )
        return np.where(z < -a / b, left, right)

    def ppf(self, q, params):
        q = np.asarray(q, dtype=float)
        nu, lam = float(params[0]), float(params[1])
        a, b, _ = self._abc(nu, lam)
        scale = np.sqrt((nu - 2.0) / nu)
        split = (1.0 - lam) / 2.0
        # clip keeps the inner t-quantile arguments inside (0, 1) on the branch
        # that is discarded by the where()
        lower = (1.0 - lam) / b * scale * stats.t.ppf(
            np.clip(q / (1.0 - lam), 1e-12, 1.0 - 1e-12), df=nu
        ) - a / b
        upper = (1.0 + lam) / b * scale * stats.t.ppf(
            np.clip((q + lam) / (1.0 + lam), 1e-12, 1.0 - 1e-12), df=nu
        ) - a / b
        return np.where(q < split, lower, upper)

    def simulate(self, size, params, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        u = rng.random(size)
        return self.ppf(u, params)


_REGISTRY = {
    "normal": Normal,
    "gaussian": Normal,
    "t": StudentT,
    "studentt": StudentT,
    "student-t": StudentT,
    "skewt": SkewT,
    "skew-t": SkewT,
}


def get_distribution(name: str) -> Distribution:
    key = str(name).strip().lower().replace("_", "-")
    key = key.replace("-", "") if key.replace("-", "") in _REGISTRY else key
    if key not in _REGISTRY:
        raise ValueError(
            f"unknown distribution {name!r}; choose from "
            f"{sorted(set(k for k in _REGISTRY))}"
        )
    return _REGISTRY[key]()
