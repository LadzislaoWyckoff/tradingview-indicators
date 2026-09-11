"""GARCH-family conditional variance models with maximum-likelihood estimation.

The three volatility recursions here cover what you actually need for an
equity index:

``GARCH(p, q)``   symmetric baseline.
``GJRGARCH(p, q)``  adds a leverage term, so a down day raises tomorrow's
                    variance more than an up day of the same size.  On SPX this
                    is not a nicety -- the asymmetry term is usually the single
                    most significant coefficient in the model.
``EGARCH(p, q)``  models log-variance, so it needs no positivity constraints
                  and handles the asymmetry multiplicatively.

Returns are modelled as ``r_t = mu_t + e_t`` with ``e_t = sigma_t * z_t`` and
``z_t`` drawn from one of the standardized distributions in
:mod:`garchlab.distributions`.

Scaling note: everything is estimated on returns in *percent* (100 * log
return).  That keeps omega away from 1e-6 and the optimizer much better
conditioned; :func:`GARCHResult.forecast` converts back for you.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import optimize, stats

from .distributions import Distribution, get_distribution

__all__ = ["GARCH", "GJRGARCH", "EGARCH", "GARCHResult", "build_model"]

_SQRT_EPS = float(np.sqrt(np.finfo(float).eps))
#: weight on the quadratic constraint penalty used during estimation
_PENALTY_WEIGHT = 1e5


def _backcast(resids: np.ndarray, decay: float = 0.94, window: int = 75) -> float:
    """Exponentially weighted seed for the variance recursion.

    Using the sample variance as the seed biases early conditional variances
    toward the unconditional level; an EWMA of the first observations is much
    closer to what the recursion would have produced had the sample started
    earlier.
    """
    n = min(window, resids.size)
    weights = decay ** np.arange(n, dtype=float)
    weights /= weights.sum()
    value = float(np.dot(weights, resids[:n] ** 2))
    return max(value, 1e-8)


# ---------------------------------------------------------------------------
# mean models
# ---------------------------------------------------------------------------
class MeanModel:
    """The conditional mean.  Deliberately thin -- daily equity returns carry
    almost no predictable mean, and over-parameterising it only steals degrees
    of freedom from the variance equation."""

    def __init__(self, kind: str = "constant"):
        kind = str(kind).lower()
        if kind not in {"zero", "constant", "ar1"}:
            raise ValueError("mean must be one of 'zero', 'constant', 'ar1'")
        self.kind = kind

    @property
    def n_params(self) -> int:
        return {"zero": 0, "constant": 1, "ar1": 2}[self.kind]

    @property
    def param_names(self) -> tuple[str, ...]:
        return {"zero": (), "constant": ("mu",), "ar1": ("const", "phi")}[self.kind]

    def starting_params(self, y: np.ndarray) -> np.ndarray:
        if self.kind == "zero":
            return np.empty(0)
        if self.kind == "constant":
            return np.array([float(np.mean(y))])
        phi = float(np.corrcoef(y[:-1], y[1:])[0, 1])
        return np.array([float(np.mean(y)) * (1.0 - phi), phi])

    def bounds(self, y: np.ndarray) -> list[tuple[float, float]]:
        scale = 10.0 * float(np.std(y)) + 1.0
        if self.kind == "zero":
            return []
        if self.kind == "constant":
            return [(-scale, scale)]
        return [(-scale, scale), (-0.999, 0.999)]

    def resids(self, params: Sequence[float], y: np.ndarray) -> np.ndarray:
        if self.kind == "zero":
            return y
        if self.kind == "constant":
            return y - params[0]
        mu = np.empty_like(y)
        mu[0] = params[0] / (1.0 - params[1]) if abs(params[1]) < 1 else params[0]
        mu[1:] = params[0] + params[1] * y[:-1]
        return y - mu

    def next_mean(self, params: Sequence[float], last_y: float) -> float:
        if self.kind == "zero":
            return 0.0
        if self.kind == "constant":
            return float(params[0])
        return float(params[0] + params[1] * last_y)


# ---------------------------------------------------------------------------
# variance models
# ---------------------------------------------------------------------------
class VolatilityModel:
    name = "base"

    def __init__(self, p: int = 1, q: int = 1):
        if p < 1 or q < 0:
            raise ValueError("need p >= 1 and q >= 0")
        self.p = int(p)  # ARCH order (lags of the squared shock)
        self.q = int(q)  # GARCH order (lags of the variance)

    @property
    def n_params(self) -> int:
        raise NotImplementedError

    @property
    def param_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    def starting_params(self, resids: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def bounds(self, resids: np.ndarray) -> list[tuple[float, float]]:
        raise NotImplementedError

    def constraints(self, dist: Distribution) -> list[dict]:
        """Inequality constraints handed to SLSQP (each must be >= 0)."""
        return []

    def set_dist_state(self, dist: Distribution, dist_params: Sequence[float]) -> None:
        """Refresh any distribution-dependent constant used by the recursion.

        Only EGARCH needs this (it centres |z| on E|z|), but the estimator
        calls it unconditionally so the constant tracks the shape parameters
        as they move during optimization.
        """
        return None

    def compute_variance(
        self, params: Sequence[float], resids: np.ndarray, backcast: float
    ) -> np.ndarray:
        raise NotImplementedError

    def persistence(self, params: Sequence[float], dist_params=(), dist=None) -> float:
        raise NotImplementedError

    def forecast_variance(
        self,
        params: Sequence[float],
        horizon: int,
        last_sigma2: np.ndarray,
        last_resid: np.ndarray,
        dist: Distribution,
        dist_params: Sequence[float],
    ) -> np.ndarray:
        raise NotImplementedError


class GARCH(VolatilityModel):
    r"""GARCH(p, q):  h_t = omega + sum a_i e_{t-i}^2 + sum b_j h_{t-j}."""

    name = "GARCH"

    @property
    def n_params(self):
        return 1 + self.p + self.q

    @property
    def param_names(self):
        return (
            ("omega",)
            + tuple(f"alpha[{i + 1}]" for i in range(self.p))
            + tuple(f"beta[{j + 1}]" for j in range(self.q))
        )

    def starting_params(self, resids):
        var = float(np.var(resids))
        alpha_total, beta_total = 0.08, 0.90
        alphas = np.full(self.p, alpha_total / self.p)
        betas = np.full(self.q, beta_total / self.q) if self.q else np.empty(0)
        omega = var * (1.0 - alpha_total - beta_total)
        return np.concatenate(([max(omega, 1e-6)], alphas, betas))

    def bounds(self, resids):
        var = float(np.var(resids))
        return [(1e-12, 10.0 * var)] + [(0.0, 1.0)] * (self.p + self.q)

    def constraints(self, dist):
        p, q = self.p, self.q

        def stationarity(theta):
            return 1.0 - 1e-6 - np.sum(theta[1 : 1 + p + q])

        return [{"type": "ineq", "fun": stationarity}]

    def compute_variance(self, params, resids, backcast):
        params = np.asarray(params, dtype=float)
        omega = params[0]
        alphas = params[1 : 1 + self.p]
        betas = params[1 + self.p :]
        t = resids.size
        sigma2 = np.empty(t)
        e2 = resids * resids
        max_lag = max(self.p, self.q)
        for i in range(t):
            value = omega
            for k in range(self.p):
                value += alphas[k] * (e2[i - k - 1] if i - k - 1 >= 0 else backcast)
            for k in range(self.q):
                value += betas[k] * (sigma2[i - k - 1] if i - k - 1 >= 0 else backcast)
            sigma2[i] = value
        del max_lag
        return sigma2

    def persistence(self, params, dist_params=(), dist=None):
        return float(np.sum(np.asarray(params)[1:]))

    def forecast_variance(self, params, horizon, last_sigma2, last_resid, dist, dist_params):
        params = np.asarray(params, dtype=float)
        omega = params[0]
        alphas = params[1 : 1 + self.p]
        betas = params[1 + self.p :]
        # buffers hold the most recent values, index 0 = most recent lag
        e2_hist = list(last_resid[::-1] ** 2)
        h_hist = list(last_sigma2[::-1])
        out = np.empty(horizon)
        for h in range(horizon):
            value = omega
            for k in range(self.p):
                value += alphas[k] * e2_hist[k]
            for k in range(self.q):
                value += betas[k] * h_hist[k]
            out[h] = value
            # beyond the sample E[e^2] = E[h], so the two buffers coincide
            e2_hist.insert(0, value)
            h_hist.insert(0, value)
        return out


class GJRGARCH(VolatilityModel):
    r"""GJR-GARCH(p, q) of Glosten, Jagannathan & Runkle (1993):

    h_t = omega + sum (a_i + g_i * 1[e_{t-i} < 0]) e_{t-i}^2 + sum b_j h_{t-j}

    ``g > 0`` is the leverage effect: the same-sized loss feeds more variance
    into tomorrow than the gain does.
    """

    name = "GJR-GARCH"

    @property
    def n_params(self):
        return 1 + 2 * self.p + self.q

    @property
    def param_names(self):
        return (
            ("omega",)
            + tuple(f"alpha[{i + 1}]" for i in range(self.p))
            + tuple(f"gamma[{i + 1}]" for i in range(self.p))
            + tuple(f"beta[{j + 1}]" for j in range(self.q))
        )

    def starting_params(self, resids):
        var = float(np.var(resids))
        alphas = np.full(self.p, 0.02 / self.p)
        gammas = np.full(self.p, 0.10 / self.p)
        betas = np.full(self.q, 0.90 / self.q) if self.q else np.empty(0)
        omega = var * (1.0 - 0.02 - 0.05 - 0.90)
        return np.concatenate(([max(omega, 1e-6)], alphas, gammas, betas))

    def bounds(self, resids):
        var = float(np.var(resids))
        return (
            [(1e-12, 10.0 * var)]
            + [(0.0, 1.0)] * self.p          # alpha >= 0
            + [(-1.0, 2.0)] * self.p         # gamma may be negative in principle
            + [(0.0, 1.0)] * self.q
        )

    def constraints(self, dist):
        p, q = self.p, self.q

        def stationarity(theta):
            # kappa = P(z < 0); 0.5 under a symmetric innovation
            return 1.0 - 1e-6 - (
                np.sum(theta[1 : 1 + p])
                + 0.5 * np.sum(theta[1 + p : 1 + 2 * p])
                + np.sum(theta[1 + 2 * p : 1 + 2 * p + q])
            )

        def positivity(theta):
            # alpha + gamma >= 0 keeps the news impact curve non-negative
            return theta[1 : 1 + p] + theta[1 + p : 1 + 2 * p]

        return [{"type": "ineq", "fun": stationarity}, {"type": "ineq", "fun": positivity}]

    def compute_variance(self, params, resids, backcast):
        params = np.asarray(params, dtype=float)
        omega = params[0]
        alphas = params[1 : 1 + self.p]
        gammas = params[1 + self.p : 1 + 2 * self.p]
        betas = params[1 + 2 * self.p :]
        t = resids.size
        e2 = resids * resids
        neg = (resids < 0.0).astype(float)
        sigma2 = np.empty(t)
        for i in range(t):
            value = omega
            for k in range(self.p):
                j = i - k - 1
                if j >= 0:
                    value += (alphas[k] + gammas[k] * neg[j]) * e2[j]
                else:
                    # the pre-sample shock is unsigned, so split it by P(z<0)=1/2
                    value += (alphas[k] + 0.5 * gammas[k]) * backcast
            for k in range(self.q):
                j = i - k - 1
                value += betas[k] * (sigma2[j] if j >= 0 else backcast)
            sigma2[i] = value
        return sigma2

    def persistence(self, params, dist_params=(), dist=None):
        params = np.asarray(params, dtype=float)
        kappa = 0.5 if dist is None else dist.prob_negative(dist_params)
        return float(
            np.sum(params[1 : 1 + self.p])
            + kappa * np.sum(params[1 + self.p : 1 + 2 * self.p])
            + np.sum(params[1 + 2 * self.p :])
        )

    def forecast_variance(self, params, horizon, last_sigma2, last_resid, dist, dist_params):
        params = np.asarray(params, dtype=float)
        omega = params[0]
        alphas = params[1 : 1 + self.p]
        gammas = params[1 + self.p : 1 + 2 * self.p]
        betas = params[1 + 2 * self.p :]
        kappa = dist.prob_negative(dist_params)
        e2_hist = list(last_resid[::-1] ** 2)
        neg_hist = list((last_resid[::-1] < 0.0).astype(float))
        h_hist = list(last_sigma2[::-1])
        out = np.empty(horizon)
        for h in range(horizon):
            value = omega
            for k in range(self.p):
                # realised lags carry the actual sign; lags that are themselves
                # forecasts carry only the probability of a down day
                value += (alphas[k] + gammas[k] * neg_hist[k]) * e2_hist[k]
            for k in range(self.q):
                value += betas[k] * h_hist[k]
            out[h] = value
            e2_hist.insert(0, value)
            neg_hist.insert(0, kappa)
            h_hist.insert(0, value)
        return out


class EGARCH(VolatilityModel):
    r"""EGARCH(p, q) of Nelson (1991), on the log variance:

    log h_t = omega + sum a_i (|z_{t-i}| - E|z|) + sum g_i z_{t-i}
                    + sum b_j log h_{t-j}

    Because the recursion lives in logs there is nothing to constrain for
    positivity; ``sum b_j < 1`` is the only stationarity requirement.
    """

    name = "EGARCH"

    def __init__(self, p: int = 1, q: int = 1):
        super().__init__(p=p, q=q)
        # E|z| under a standard normal; overwritten by set_dist_state
        self._e_abs_z = float(np.sqrt(2.0 / np.pi))
        self._abs_cache: dict = {}

    def set_dist_state(self, dist, dist_params):
        # E|z| moves very slowly in the shape parameters, so a coarse cache key
        # keeps the quadrature out of the inner likelihood loop
        key = (dist.name, tuple(np.round(np.asarray(dist_params, dtype=float), 4)))
        if key not in self._abs_cache:
            if len(self._abs_cache) > 4096:
                self._abs_cache.clear()
            self._abs_cache[key] = dist.mean_abs(dist_params)
        self._e_abs_z = self._abs_cache[key]

    @property
    def n_params(self):
        return 1 + 2 * self.p + self.q

    @property
    def param_names(self):
        return (
            ("omega",)
            + tuple(f"alpha[{i + 1}]" for i in range(self.p))
            + tuple(f"gamma[{i + 1}]" for i in range(self.p))
            + tuple(f"beta[{j + 1}]" for j in range(self.q))
        )

    def starting_params(self, resids):
        log_var = float(np.log(np.var(resids)))
        beta_total = 0.95
        alphas = np.full(self.p, 0.10 / self.p)
        gammas = np.full(self.p, -0.05 / self.p)
        betas = np.full(self.q, beta_total / self.q) if self.q else np.empty(0)
        omega = log_var * (1.0 - beta_total)
        return np.concatenate(([omega], alphas, gammas, betas))

    def bounds(self, resids):
        return (
            [(-10.0, 10.0)]
            + [(-1.0, 1.0)] * self.p
            + [(-1.0, 1.0)] * self.p
            + [(-0.999, 0.999)] * self.q
        )

    def constraints(self, dist):
        q = self.q
        if q == 0:
            return []

        p = self.p

        def stationarity(theta):
            return 1.0 - 1e-6 - np.sum(np.abs(theta[1 + 2 * p : 1 + 2 * p + q]))

        return [{"type": "ineq", "fun": stationarity}]

    def compute_variance(self, params, resids, backcast):
        params = np.asarray(params, dtype=float)
        omega = params[0]
        alphas = params[1 : 1 + self.p]
        gammas = params[1 + self.p : 1 + 2 * self.p]
        betas = params[1 + 2 * self.p :]
        e_abs_z = self._e_abs_z
        t = resids.size
        log_h = np.empty(t)
        sigma2 = np.empty(t)
        log_backcast = float(np.log(backcast))
        for i in range(t):
            value = omega
            for k in range(self.p):
                j = i - k - 1
                if j >= 0:
                    z = resids[j] / np.sqrt(sigma2[j])
                    value += alphas[k] * (abs(z) - e_abs_z) + gammas[k] * z
                # a pre-sample shock contributes its expectation, i.e. zero
            for k in range(self.q):
                j = i - k - 1
                value += betas[k] * (log_h[j] if j >= 0 else log_backcast)
            log_h[i] = min(max(value, -30.0), 30.0)
            sigma2[i] = np.exp(log_h[i])
        return sigma2

    def persistence(self, params, dist_params=(), dist=None):
        return float(np.sum(np.asarray(params)[1 + 2 * self.p :]))

    def forecast_variance(self, params, horizon, last_sigma2, last_resid, dist, dist_params):
        """Simulated forecast.

        Unlike GARCH/GJR, E[h_{t+k}] has no clean closed form here because the
        recursion is linear in *log* h.  Averaging exp(log h) over simulated
        paths is the honest answer; the analytic shortcut would understate the
        forecast systematically.
        """
        params = np.asarray(params, dtype=float)
        omega = params[0]
        alphas = params[1 : 1 + self.p]
        gammas = params[1 + self.p : 1 + 2 * self.p]
        betas = params[1 + 2 * self.p :]
        self.set_dist_state(dist, dist_params)
        e_abs_z = self._e_abs_z

        if horizon == 1:
            # one step ahead the recursion is deterministic given the state --
            # no simulation needed, which matters a great deal because the
            # walk-forward backtest calls this once per day
            z_hist = (last_resid / np.sqrt(last_sigma2))[::-1]
            log_h_hist = np.log(last_sigma2[::-1])
            value = omega
            for k in range(self.p):
                value += alphas[k] * (abs(z_hist[k]) - e_abs_z) + gammas[k] * z_hist[k]
            for k in range(self.q):
                value += betas[k] * log_h_hist[k]
            return np.array([float(np.exp(min(max(value, -30.0), 30.0)))])

        n_sim = 10000
        rng = np.random.default_rng(20240917)

        log_h_hist = np.log(last_sigma2[::-1])
        z_hist = (last_resid / np.sqrt(last_sigma2))[::-1]
        log_h_buf = np.tile(log_h_hist, (n_sim, 1))
        z_buf = np.tile(z_hist, (n_sim, 1))
        out = np.empty(horizon)
        for h in range(horizon):
            value = np.full(n_sim, omega)
            for k in range(self.p):
                value += alphas[k] * (np.abs(z_buf[:, k]) - e_abs_z) + gammas[k] * z_buf[:, k]
            for k in range(self.q):
                value += betas[k] * log_h_buf[:, k]
            value = np.clip(value, -30.0, 30.0)
            out[h] = float(np.mean(np.exp(value)))
            log_h_buf = np.column_stack([value, log_h_buf])
            z_buf = np.column_stack([dist.simulate(n_sim, dist_params, rng), z_buf])
        return out


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass
class GARCHResult:
    """Everything a fitted model knows about itself."""

    params: np.ndarray
    param_names: tuple[str, ...]
    loglikelihood: float
    conditional_variance: np.ndarray
    resids: np.ndarray
    returns: np.ndarray
    mean_model: MeanModel
    vol_model: VolatilityModel
    dist: Distribution
    backcast: float
    scale: float
    periods_per_year: float
    converged: bool
    message: str
    std_errors: np.ndarray | None = None
    index: object = None
    meta: dict = field(default_factory=dict)

    # -- parameter views --------------------------------------------------
    @property
    def n_mean(self) -> int:
        return self.mean_model.n_params

    @property
    def mean_params(self) -> np.ndarray:
        return self.params[: self.n_mean]

    @property
    def vol_params(self) -> np.ndarray:
        return self.params[self.n_mean : self.n_mean + self.vol_model.n_params]

    @property
    def dist_params(self) -> np.ndarray:
        return self.params[self.n_mean + self.vol_model.n_params :]

    @property
    def nobs(self) -> int:
        return int(self.returns.size)

    @property
    def n_params(self) -> int:
        return int(self.params.size)

    @property
    def aic(self) -> float:
        return -2.0 * self.loglikelihood + 2.0 * self.n_params

    @property
    def bic(self) -> float:
        return -2.0 * self.loglikelihood + np.log(self.nobs) * self.n_params

    @property
    def tvalues(self) -> np.ndarray | None:
        if self.std_errors is None:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.params / self.std_errors

    @property
    def pvalues(self) -> np.ndarray | None:
        t = self.tvalues
        if t is None:
            return None
        return 2.0 * (1.0 - stats.norm.cdf(np.abs(t)))

    # -- derived quantities ----------------------------------------------
    @property
    def std_resids(self) -> np.ndarray:
        return self.resids / np.sqrt(self.conditional_variance)

    @property
    def persistence(self) -> float:
        return self.vol_model.persistence(self.vol_params, self.dist_params, self.dist)

    @property
    def half_life(self) -> float:
        """Trading days for a variance shock to decay halfway back to normal."""
        rho = self.persistence
        if not 0.0 < rho < 1.0:
            return float("inf")
        return float(np.log(0.5) / np.log(rho))

    @property
    def conditional_vol(self) -> np.ndarray:
        """In-sample conditional volatility, annualized, in the original units."""
        return (
            np.sqrt(self.conditional_variance)
            / self.scale
            * np.sqrt(self.periods_per_year)
        )

    def unconditional_variance(self) -> float:
        """Long-run variance the recursion reverts to (in the estimation scale)."""
        vp = np.asarray(self.vol_params, dtype=float)
        if isinstance(self.vol_model, EGARCH):
            beta_sum = self.vol_model.persistence(vp)
            if beta_sum >= 1.0:
                return float("nan")
            return float(np.exp(vp[0] / (1.0 - beta_sum)))
        rho = self.persistence
        if rho >= 1.0:
            return float("nan")
        return float(vp[0] / (1.0 - rho))

    @property
    def long_run_vol(self) -> float:
        """Annualized long-run volatility in the original units (e.g. 0.16)."""
        var = self.unconditional_variance()
        return float(np.sqrt(var) / self.scale * np.sqrt(self.periods_per_year))

    # -- forecasting ------------------------------------------------------
    def forecast(self, horizon: int = 22, annualize: bool = True) -> dict:
        """Forecast conditional variance ``horizon`` steps past the sample end.

        Returns a dict with:
        ``variance``      per-step variance forecasts (estimation scale)
        ``vol``           per-step volatility, annualized if requested
        ``cumulative_vol``  volatility of the *h-day aggregate* return, which is
                            what an option or a position sizer actually needs
        ``mean``          per-step conditional mean forecast
        """
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        vm = self.vol_model
        n_lag = max(vm.p, vm.q, 1)
        var_path = vm.forecast_variance(
            self.vol_params,
            horizon,
            self.conditional_variance[-n_lag:],
            self.resids[-n_lag:],
            self.dist,
            self.dist_params,
        )
        step_vol = np.sqrt(var_path) / self.scale
        cum_var = np.cumsum(var_path) / (self.scale ** 2)
        mean_path = np.full(horizon, self.mean_model.next_mean(self.mean_params, self.returns[-1]))
        mean_path = mean_path / self.scale
        steps = np.arange(1, horizon + 1, dtype=float)
        factor = np.sqrt(self.periods_per_year) if annualize else 1.0
        # vol of the h-day aggregate return: variances add across days
        agg_vol = np.sqrt(cum_var)
        return {
            "horizon": steps.astype(int),
            "variance": var_path,
            "vol": step_vol * factor,
            # the h-day vol re-expressed per period, so it is comparable to
            # `vol` and to an implied-vol quote
            "cumulative_vol": agg_vol / np.sqrt(steps) * factor if annualize else agg_vol,
            "cumulative_vol_raw": agg_vol,
            "mean": mean_path,
        }

    def summary(self) -> str:
        lines = []
        lines.append(f"{self.vol_model.name}({self.vol_model.p}, {self.vol_model.q})"
                     f"  mean={self.mean_model.kind}  dist={self.dist.name}")
        lines.append(f"observations: {self.nobs}    log-likelihood: {self.loglikelihood:,.3f}")
        lines.append(f"AIC: {self.aic:,.3f}    BIC: {self.bic:,.3f}")
        lines.append("")
        header = f"{'parameter':<14}{'estimate':>12}{'std.err':>12}{'t':>10}{'p':>10}"
        lines.append(header)
        lines.append("-" * len(header))
        se = self.std_errors if self.std_errors is not None else np.full(self.n_params, np.nan)
        tv = self.tvalues if self.tvalues is not None else np.full(self.n_params, np.nan)
        pv = self.pvalues if self.pvalues is not None else np.full(self.n_params, np.nan)
        for name, value, s, t, p in zip(self.param_names, self.params, se, tv, pv):
            lines.append(f"{name:<14}{value:>12.6f}{s:>12.6f}{t:>10.3f}{p:>10.4f}")
        lines.append("-" * len(header))
        lines.append(f"persistence: {self.persistence:.6f}    half-life: {self.half_life:,.1f} days")
        lines.append(f"long-run annualized vol: {self.long_run_vol * 100:.2f}%")
        if not self.converged:
            lines.append(f"WARNING: optimizer did not converge -- {self.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# estimation
# ---------------------------------------------------------------------------
_VOL_REGISTRY = {"garch": GARCH, "gjr": GJRGARCH, "gjrgarch": GJRGARCH,
                 "gjr-garch": GJRGARCH, "egarch": EGARCH}


def build_model(kind: str = "garch", p: int = 1, q: int = 1) -> VolatilityModel:
    key = str(kind).strip().lower()
    if key not in _VOL_REGISTRY:
        raise ValueError(f"unknown model {kind!r}; choose from {sorted(_VOL_REGISTRY)}")
    return _VOL_REGISTRY[key](p=p, q=q)


def _negative_loglikelihood(
    theta, y, mean_model, vol_model, dist, backcast, var_bounds, per_obs=False
):
    """Negative log-likelihood, kept finite everywhere on purpose.

    SLSQP builds its search direction from finite differences, so a likelihood
    that returns a huge sentinel on infeasible parameters produces an absurd
    gradient and throws the optimizer into a corner it never leaves.  Clipping
    the conditional variance into ``var_bounds`` keeps the surface continuous
    and merely flat where we do not want it, which the optimizer can walk back
    out of.
    """
    n_mean = mean_model.n_params
    n_vol = vol_model.n_params
    mean_params = theta[:n_mean]
    vol_params = theta[n_mean : n_mean + n_vol]
    dist_params = theta[n_mean + n_vol :]

    resids = mean_model.resids(mean_params, y)
    vol_model.set_dist_state(dist, dist_params)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        sigma2 = vol_model.compute_variance(vol_params, resids, backcast)
        sigma2 = np.where(np.isfinite(sigma2), sigma2, var_bounds[1])
        sigma2 = np.clip(sigma2, var_bounds[0], var_bounds[1])
        z = resids / np.sqrt(sigma2)
        ll = dist.loglikelihood(z, dist_params) - 0.5 * np.log(sigma2)
    ll = np.where(np.isfinite(ll), ll, -1e4)
    return -ll if per_obs else -float(np.sum(ll))


def _numeric_jacobian(fun, theta, eps_scale=_SQRT_EPS):
    """Per-observation score matrix by central differences (T x k)."""
    theta = np.asarray(theta, dtype=float)
    base = fun(theta)
    jac = np.empty((base.size, theta.size))
    for i in range(theta.size):
        step = eps_scale * max(abs(theta[i]), 1e-2)
        up, dn = theta.copy(), theta.copy()
        up[i] += step
        dn[i] -= step
        jac[:, i] = (fun(up) - fun(dn)) / (2.0 * step)
    return jac


def _numeric_hessian(fun, theta, eps_scale=(_SQRT_EPS ** 0.5)):
    theta = np.asarray(theta, dtype=float)
    k = theta.size
    steps = np.array([eps_scale * max(abs(theta[i]), 1e-2) for i in range(k)])
    hess = np.empty((k, k))
    f0 = fun(theta)
    for i in range(k):
        for j in range(i, k):
            tp = theta.copy(); tp[i] += steps[i]; tp[j] += steps[j]
            tm = theta.copy(); tm[i] -= steps[i]; tm[j] -= steps[j]
            tpm = theta.copy(); tpm[i] += steps[i]; tpm[j] -= steps[j]
            tmp = theta.copy(); tmp[i] -= steps[i]; tmp[j] += steps[j]
            if i == j:
                value = (fun(tp) - 2.0 * f0 + fun(tm)) / (4.0 * steps[i] * steps[j])
            else:
                value = (fun(tp) - fun(tpm) - fun(tmp) + fun(tm)) / (
                    4.0 * steps[i] * steps[j]
                )
            hess[i, j] = hess[j, i] = value
    return hess


def fit(
    returns,
    model: str | VolatilityModel = "gjr",
    p: int = 1,
    q: int = 1,
    dist: str | Distribution = "skewt",
    mean: str = "constant",
    periods_per_year: float = 252.0,
    scale: float = 100.0,
    robust_errors: bool = True,
    restarts: int = 2,
) -> GARCHResult:
    """Estimate a GARCH-family model by (quasi) maximum likelihood.

    Parameters
    ----------
    returns
        1-D array (or pandas Series) of *simple log returns*, e.g. 0.0123 for
        +1.23%.  They are multiplied by ``scale`` internally.
    model
        ``"garch"``, ``"gjr"`` or ``"egarch"``, or a ready
        :class:`VolatilityModel`.
    dist
        ``"normal"``, ``"t"`` or ``"skewt"``.
    robust_errors
        Bollerslev-Wooldridge sandwich standard errors.  These stay valid when
        the innovation distribution is misspecified, which it always somewhat
        is -- prefer them to the naive inverse-Hessian errors.
    restarts
        Extra optimizer runs from perturbed starting values.  GARCH likelihoods
        are mildly multi-modal; the best of a few runs is kept.
    """
    index = getattr(returns, "index", None)
    y_raw = np.asarray(getattr(returns, "values", returns), dtype=float).ravel()
    if np.any(~np.isfinite(y_raw)):
        raise ValueError("returns contain NaN or inf; clean them before fitting")
    if y_raw.size < 100:
        raise ValueError(f"need at least 100 observations, got {y_raw.size}")
    y = y_raw * scale

    vol_model = model if isinstance(model, VolatilityModel) else build_model(model, p, q)
    distribution = dist if isinstance(dist, Distribution) else get_distribution(dist)
    mean_model = MeanModel(mean)

    theta0 = np.concatenate(
        [
            mean_model.starting_params(y),
            vol_model.starting_params(y - np.mean(y)),
            distribution.starting_params(),
        ]
    )
    bounds = (
        mean_model.bounds(y) + vol_model.bounds(y - np.mean(y)) + distribution.bounds()
    )
    backcast = _backcast(y - np.mean(y))
    sample_var = float(np.var(y))
    # ~15 orders of magnitude of headroom: wide enough never to bind on a sane
    # fit, tight enough that an explosive recursion yields a large finite
    # penalty instead of inf
    var_bounds = (sample_var * 1e-8, sample_var * 1e7)

    n_mean = mean_model.n_params
    n_vol_params = vol_model.n_params
    # the constraint callables are written against the volatility block alone,
    # so slice it out exactly -- an open-ended slice would sweep the
    # distribution's shape parameters into the stationarity sum
    constraint_funs = [
        (lambda th, f=con["fun"]: np.atleast_1d(f(th[n_mean : n_mean + n_vol_params])))
        for con in vol_model.constraints(distribution)
    ]

    def constraint_violation(theta) -> float:
        """Total squared violation of the model's inequality constraints."""
        total = 0.0
        for fun in constraint_funs:
            g = fun(theta)
            total += float(np.sum(np.minimum(g, 0.0) ** 2))
        return total

    def objective(theta):
        """Plain negative log-likelihood; SLSQP enforces the constraints."""
        return _negative_loglikelihood(
            theta, y, mean_model, vol_model, distribution, backcast, var_bounds
        )

    def penalized(theta):
        """Negative log-likelihood with the constraints folded in as a smooth
        quadratic penalty, for the derivative-free polish step, which takes
        bounds but no constraints.  The penalty is inactive at any sane
        optimum, so it does not bias the estimates."""
        return objective(theta) + _PENALTY_WEIGHT * constraint_violation(theta)

    best = None
    rng = np.random.default_rng(0)
    starts = [theta0]

    # Two-stage start.  Jointly optimizing the variance recursion and a fat
    # tailed, skewed innovation from a cold start is where GARCH fits go to
    # die: the likelihood has a corner where nu collapses toward 2, variance
    # dynamics switch off and the fat tail explains everything.  Gaussian QMLE
    # is consistent for the variance parameters even when the true innovation
    # is not normal, so we fit that first and hand the result over.
    if distribution.param_names:
        try:
            stage1 = fit(
                y_raw,
                model=vol_model,
                dist="normal",
                mean=mean,
                periods_per_year=periods_per_year,
                scale=scale,
                robust_errors=False,
                restarts=0,
            )
            z1 = stage1.std_resids

            def dist_only(shape):
                value = -float(np.sum(distribution.loglikelihood(z1, shape)))
                return value if np.isfinite(value) else 1e12

            shape0 = optimize.minimize(
                dist_only,
                distribution.starting_params(),
                method="Nelder-Mead",
                options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 400},
            ).x
            shape0 = np.array(
                [min(max(v, lo + 1e-6), hi - 1e-6)
                 for v, (lo, hi) in zip(shape0, distribution.bounds())]
            )
            starts.insert(0, np.concatenate([stage1.params, shape0]))
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            pass

    for _ in range(max(0, int(restarts))):
        jitter = theta0 * (1.0 + 0.15 * rng.standard_normal(theta0.size))
        jitter = np.array(
            [min(max(v, lo + 1e-9), hi - 1e-9) for v, (lo, hi) in zip(jitter, bounds)]
        )
        starts.append(jitter)

    constraints = [{"type": "ineq", "fun": fun} for fun in constraint_funs]
    for start in starts:
        try:
            res = optimize.minimize(
                objective,
                start,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-10},
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            # a starting value the recursion cannot survive; try the next one
            continue
        if not np.all(np.isfinite(res.x)):
            continue
        # never hand back a point worse than the one we started from
        start_fun = objective(start)
        if start_fun < res.fun:
            res = optimize.OptimizeResult(
                x=np.asarray(start, dtype=float),
                fun=float(start_fun),
                success=False,
                message="optimizer failed to improve on the starting value",
            )
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        raise RuntimeError("optimization failed from every starting value")

    # Derivative-free polish.  L-BFGS-B stops on a gradient criterion computed
    # from finite differences, which on a 6000-term likelihood is noisy enough
    # to leave a little on the table; Nelder-Mead from the winner picks it up.
    try:
        polished = optimize.minimize(
            penalized,
            best.x,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-9},
        )
        if np.all(np.isfinite(polished.x)) and polished.fun < best.fun:
            best = optimize.OptimizeResult(
                x=polished.x, fun=float(polished.fun),
                success=bool(best.success or polished.success),
                message=str(polished.message),
            )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        pass

    theta = np.asarray(best.x, dtype=float)
    n_vol = vol_model.n_params
    resids = mean_model.resids(theta[:n_mean], y)
    vol_model.set_dist_state(distribution, theta[n_mean + n_vol :])
    sigma2 = vol_model.compute_variance(theta[n_mean : n_mean + n_vol], resids, backcast)

    std_errors = None
    try:
        hess = _numeric_hessian(objective, theta)
        hess_inv = np.linalg.pinv(hess)
        if robust_errors:
            scores = -_numeric_jacobian(
                lambda th: _negative_loglikelihood(
                    th, y, mean_model, vol_model, distribution, backcast,
                    var_bounds, per_obs=True
                ),
                theta,
            )
            opg = scores.T @ scores
            cov = hess_inv @ opg @ hess_inv
        else:
            cov = hess_inv
        diag = np.diag(cov)
        std_errors = np.where(diag > 0, np.sqrt(np.abs(diag)), np.nan)
    except Exception:  # pragma: no cover
        std_errors = None

    param_names = (
        mean_model.param_names + vol_model.param_names + distribution.param_names
    )
    violation = constraint_violation(theta)
    converged = bool(best.success) and violation < 1e-8
    message = str(best.message)
    if violation >= 1e-8:
        message = (
            f"{message}; stationarity/positivity constraint violated by "
            f"{np.sqrt(violation):.2e}"
        )
    clean_ll = -_negative_loglikelihood(
        theta, y, mean_model, vol_model, distribution, backcast, var_bounds
    )
    return GARCHResult(
        params=theta,
        param_names=param_names,
        loglikelihood=float(clean_ll),
        conditional_variance=sigma2,
        resids=resids,
        returns=y,
        mean_model=mean_model,
        vol_model=vol_model,
        dist=distribution,
        backcast=backcast,
        scale=scale,
        periods_per_year=periods_per_year,
        converged=converged,
        message=message,
        std_errors=std_errors,
        index=index,
    )


def simulate(
    n: int,
    model: str | VolatilityModel = "gjr",
    params: Sequence[float] | None = None,
    dist: str | Distribution = "normal",
    dist_params: Sequence[float] = (),
    mu: float = 0.0,
    burn: int = 1000,
    scale: float = 100.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a return path from a GARCH-family data generating process.

    Returns ``(returns, conditional_vol)`` with returns on the *original*
    scale (divide-by-``scale`` already applied), so a simulated series can be
    fed straight back into :func:`fit`.  This is what the test suite uses to
    check that the estimator recovers parameters it is supposed to recover.
    """
    vol_model = model if isinstance(model, VolatilityModel) else build_model(model)
    distribution = dist if isinstance(dist, Distribution) else get_distribution(dist)
    rng = np.random.default_rng(seed)
    if params is None:
        params = vol_model.starting_params(rng.standard_normal(1000))
    params = np.asarray(params, dtype=float)
    if params.size != vol_model.n_params:
        raise ValueError(
            f"{vol_model.name}({vol_model.p}, {vol_model.q}) needs "
            f"{vol_model.n_params} parameters {vol_model.param_names}, got {params.size}"
        )

    total = int(n) + int(burn)
    z = distribution.simulate(total, dist_params, rng)
    e = np.zeros(total)
    h = np.zeros(total)
    p_, q_ = vol_model.p, vol_model.q
    is_egarch = isinstance(vol_model, EGARCH)
    if is_egarch:
        e_abs_z = distribution.mean_abs(dist_params)
        beta_sum = float(np.sum(params[1 + 2 * p_ :]))
        h0 = float(np.exp(params[0] / (1.0 - beta_sum))) if beta_sum < 1 else 1.0
    else:
        rho = vol_model.persistence(params, dist_params, distribution)
        h0 = float(params[0] / (1.0 - rho)) if rho < 1 else 1.0

    for i in range(total):
        lag = max(p_, q_)
        if i < lag:
            h[i] = h0
        elif is_egarch:
            alphas = params[1 : 1 + p_]
            gammas = params[1 + p_ : 1 + 2 * p_]
            betas = params[1 + 2 * p_ :]
            log_h = params[0]
            for k in range(p_):
                zz = e[i - k - 1] / np.sqrt(h[i - k - 1])
                log_h += alphas[k] * (abs(zz) - e_abs_z) + gammas[k] * zz
            for k in range(q_):
                log_h += betas[k] * np.log(h[i - k - 1])
            h[i] = float(np.exp(min(max(log_h, -30.0), 30.0)))
        elif isinstance(vol_model, GJRGARCH):
            alphas = params[1 : 1 + p_]
            gammas = params[1 + p_ : 1 + 2 * p_]
            betas = params[1 + 2 * p_ :]
            value = params[0]
            for k in range(p_):
                j = i - k - 1
                value += (alphas[k] + gammas[k] * (e[j] < 0.0)) * e[j] ** 2
            for k in range(q_):
                value += betas[k] * h[i - k - 1]
            h[i] = value
        else:
            alphas = params[1 : 1 + p_]
            betas = params[1 + p_ :]
            value = params[0]
            for k in range(p_):
                value += alphas[k] * e[i - k - 1] ** 2
            for k in range(q_):
                value += betas[k] * h[i - k - 1]
            h[i] = value
        e[i] = np.sqrt(h[i]) * z[i]

    returns = (mu * scale + e[burn:]) / scale
    return returns, np.sqrt(h[burn:]) / scale
