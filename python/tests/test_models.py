"""Estimation tests.

The load-bearing test here is parameter recovery: simulate from a known data
generating process, estimate, and check the estimator finds it back.  Anything
subtler than that -- a wrong constraint, a mis-sliced parameter vector, a
broken recursion -- shows up as a fit that misses the truth.
"""

import numpy as np
import pytest

from garchlab.distributions import get_distribution
from garchlab.models import (
    EGARCH,
    GARCH,
    GJRGARCH,
    MeanModel,
    build_model,
    fit,
    simulate,
)


@pytest.fixture(scope="module")
def garch_series():
    return simulate(5000, "garch", [0.05, 0.08, 0.90], seed=7)[0]


@pytest.fixture(scope="module")
def gjr_series():
    return simulate(5000, "gjr", [0.03, 0.02, 0.12, 0.90], seed=11)[0]


def test_garch_recovers_its_own_parameters(garch_series):
    res = fit(garch_series, model="garch", dist="normal", restarts=0, robust_errors=False)
    omega, alpha, beta = res.vol_params
    assert res.converged
    assert alpha == pytest.approx(0.08, abs=0.03)
    assert beta == pytest.approx(0.90, abs=0.03)
    assert omega == pytest.approx(0.05, abs=0.03)


def test_gjr_recovers_the_leverage_term(gjr_series):
    res = fit(gjr_series, model="gjr", dist="normal", restarts=0, robust_errors=False)
    _, alpha, gamma, beta = res.vol_params
    assert res.converged
    assert gamma == pytest.approx(0.12, abs=0.04)
    assert gamma > alpha          # the whole point: down days matter more
    assert beta == pytest.approx(0.90, abs=0.04)


def test_egarch_recovers_its_own_parameters():
    r = simulate(5000, "egarch", [-0.08, 0.12, -0.09, 0.97], seed=11)[0]
    res = fit(r, model="egarch", dist="normal", restarts=0, robust_errors=False)
    _, alpha, gamma, beta = res.vol_params
    assert res.converged
    assert beta == pytest.approx(0.97, abs=0.03)
    assert gamma == pytest.approx(-0.09, abs=0.05)
    assert alpha == pytest.approx(0.12, abs=0.06)


@pytest.mark.parametrize("dist,params", [("t", (6.0,)), ("skewt", (7.0, -0.25))])
def test_fat_tailed_fits_recover_both_blocks(dist, params):
    """Regression test for a real bug.

    The constraint callables slice the parameter vector, and an open-ended
    slice used to sweep the distribution's shape parameters into the
    stationarity sum -- so the optimizer satisfied ``alpha + gamma/2 + beta +
    nu + lambda < 1`` by driving nu to its lower bound and switching the
    variance dynamics off entirely.  The fit looked plausible and was garbage.
    """
    r = simulate(5000, "gjr", [0.03, 0.02, 0.12, 0.90], dist=dist,
                 dist_params=params, seed=11)[0]
    res = fit(r, model="gjr", dist=dist, restarts=0, robust_errors=False)
    assert res.converged
    assert res.vol_params[3] == pytest.approx(0.90, abs=0.05)     # beta survived
    assert res.dist_params[0] == pytest.approx(params[0], rel=0.4)
    assert res.dist_params[0] > 3.0                                # not at the bound


def test_stationarity_constraint_ignores_the_distribution_block():
    """The constraint must read only the volatility slice, whatever follows."""
    model = build_model("gjr")
    dist = get_distribution("skewt")
    vol_params = np.array([0.03, 0.02, 0.12, 0.90])
    (stationarity, _positivity) = model.constraints(dist)
    padded = np.concatenate([vol_params, [8.0, -0.2]])
    assert stationarity["fun"](vol_params) == pytest.approx(stationarity["fun"](padded))
    assert stationarity["fun"](vol_params) > 0


def test_egarch_stationarity_constraint_ignores_the_distribution_block():
    model = build_model("egarch")
    vol_params = np.array([-0.08, 0.12, -0.09, 0.97])
    (stationarity,) = model.constraints(get_distribution("t"))
    padded = np.concatenate([vol_params, [8.0]])
    assert stationarity["fun"](vol_params) == pytest.approx(stationarity["fun"](padded))


def test_persistence_and_half_life_agree(garch_series):
    res = fit(garch_series, model="garch", dist="normal", restarts=0, robust_errors=False)
    rho = res.persistence
    assert 0.0 < rho < 1.0
    # a shock decays to half after `half_life` steps, by definition
    assert rho ** res.half_life == pytest.approx(0.5, rel=1e-9)


def test_forecast_reverts_to_the_unconditional_variance(garch_series):
    res = fit(garch_series, model="garch", dist="normal", restarts=0, robust_errors=False)
    path = res.forecast(horizon=4000, annualize=False)["variance"]
    assert path[-1] == pytest.approx(res.unconditional_variance(), rel=1e-3)
    assert np.all(np.diff(path) * np.sign(path[-1] - path[0]) >= -1e-12)  # monotone


@pytest.mark.parametrize("kind", ["garch", "gjr", "egarch"])
def test_one_step_forecast_equals_the_next_recursion_step(kind):
    """Forecasting one step must reproduce what the recursion would have done."""
    params = {"garch": [0.05, 0.08, 0.90], "gjr": [0.03, 0.02, 0.12, 0.90],
              "egarch": [-0.08, 0.12, -0.09, 0.97]}[kind]
    r = simulate(600, kind, params, seed=3)[0]
    res = fit(r[:-1], model=kind, dist="normal", restarts=0, robust_errors=False)
    one_step = res.vol_model.forecast_variance(
        res.vol_params, 1, res.conditional_variance[-1:], res.resids[-1:],
        res.dist, res.dist_params,
    )[0]
    # rerun the full recursion with one more observation and compare the tail
    extended = res.mean_model.resids(res.mean_params, r * res.scale)
    res.vol_model.set_dist_state(res.dist, res.dist_params)
    full = res.vol_model.compute_variance(res.vol_params, extended, res.backcast)
    assert one_step == pytest.approx(full[-1], rel=1e-6)


def test_cumulative_forecast_aggregates_the_daily_variances(garch_series):
    res = fit(garch_series, model="garch", dist="normal", restarts=0, robust_errors=False)
    fc = res.forecast(horizon=10, annualize=False)
    expected = np.sqrt(np.cumsum(fc["variance"]) / res.scale ** 2)
    assert fc["cumulative_vol_raw"] == pytest.approx(expected)


def test_annualized_cumulative_vol_is_per_period(garch_series):
    res = fit(garch_series, model="garch", dist="normal", restarts=0, robust_errors=False)
    fc = res.forecast(horizon=10, annualize=True)
    raw = fc["cumulative_vol_raw"]
    steps = np.arange(1, 11)
    assert fc["cumulative_vol"] == pytest.approx(raw / np.sqrt(steps) * np.sqrt(252.0))


def test_fat_tails_beat_a_normal_on_a_fat_tailed_series():
    r = simulate(4000, "garch", [0.05, 0.08, 0.90], dist="t",
                 dist_params=(5.0,), seed=21)[0]
    normal = fit(r, model="garch", dist="normal", restarts=0, robust_errors=False)
    student = fit(r, model="garch", dist="t", restarts=0, robust_errors=False)
    assert student.loglikelihood > normal.loglikelihood
    assert student.bic < normal.bic


def test_robust_standard_errors_are_finite_and_positive(gjr_series):
    res = fit(gjr_series, model="gjr", dist="normal", restarts=0, robust_errors=True)
    assert res.std_errors is not None
    assert np.all(np.isfinite(res.std_errors))
    assert np.all(res.std_errors > 0)
    assert abs(res.tvalues[3]) > 2.0     # gamma is the true leverage term


def test_ar1_mean_model_residuals_use_the_lagged_return():
    mean = MeanModel("ar1")
    y = np.array([1.0, 2.0, 3.0, 4.0])
    resids = mean.resids([0.5, 0.25], y)
    assert resids[1:] == pytest.approx(y[1:] - (0.5 + 0.25 * y[:-1]))
    assert mean.next_mean([0.5, 0.25], 4.0) == pytest.approx(1.5)


def test_simulate_rejects_a_wrong_length_parameter_vector():
    with pytest.raises(ValueError, match="needs 4 parameters"):
        simulate(500, "gjr", [0.05, 0.08, 0.90], seed=1)


def test_fit_rejects_short_and_dirty_input():
    with pytest.raises(ValueError, match="at least 100"):
        fit(np.zeros(50), model="garch")
    bad = simulate(400, "garch", [0.05, 0.08, 0.90], seed=1)[0]
    bad[10] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fit(bad, model="garch")


def test_build_model_registry():
    assert isinstance(build_model("garch"), GARCH)
    assert isinstance(build_model("gjr"), GJRGARCH)
    assert isinstance(build_model("egarch"), EGARCH)
    with pytest.raises(ValueError, match="unknown model"):
        build_model("figarch")
