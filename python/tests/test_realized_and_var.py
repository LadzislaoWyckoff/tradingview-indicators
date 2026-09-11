"""Realized-variance proxies and the VaR risk layer."""

import numpy as np
import pandas as pd
import pytest

from garchlab.distributions import get_distribution
from garchlab.realized import (
    annualize,
    garman_klass,
    parkinson,
    realized_variance,
    rogers_satchell,
    squared_return,
    yang_zhang,
)
from garchlab.var import (
    backtest_var,
    christoffersen_independence,
    conditional_coverage,
    expected_shortfall,
    kupiec_pof,
    value_at_risk,
)


@pytest.fixture(scope="module")
def ohlc():
    """Intraday paths simulated at a known volatility, aggregated to bars."""
    rng = np.random.default_rng(0)
    n, steps, sigma = 4000, 400, 0.011
    path = np.cumsum(rng.standard_normal((n, steps)) * sigma / np.sqrt(steps), axis=1)
    close_rel = np.exp(path[:, -1])
    level = np.cumprod(close_rel)
    open_ = level / close_rel
    return pd.DataFrame({
        "open": open_,
        "high": open_ * np.exp(np.maximum(path.max(axis=1), 0.0)),
        "low": open_ * np.exp(np.minimum(path.min(axis=1), 0.0)),
        "close": level,
    }), sigma


@pytest.mark.parametrize("fn", [parkinson, garman_klass, rogers_satchell])
def test_range_proxies_land_near_the_true_volatility(ohlc, fn):
    frame, sigma = ohlc
    est = float(np.sqrt(np.nanmean(fn(frame))))
    # discrete sampling always clips the true extremes a little
    assert 0.75 * sigma < est <= 1.05 * sigma


def test_range_proxies_are_less_noisy_than_the_squared_return(ohlc):
    frame, _ = ohlc
    r = np.log(frame["close"]).diff().to_numpy()
    noisy = squared_return(r[1:])
    for fn in (parkinson, garman_klass):
        quiet = fn(frame)[1:]
        assert np.nanstd(quiet) / np.nanmean(quiet) < np.nanstd(noisy) / np.nanmean(noisy)


def test_yang_zhang_single_day_adds_the_gap_to_rogers_satchell(ohlc):
    frame, _ = ohlc
    yz = yang_zhang(frame, window=1)
    gap = np.log(frame["open"].to_numpy() / np.concatenate(
        ([np.nan], frame["close"].to_numpy()[:-1])))
    assert yz[1:] == pytest.approx((gap ** 2 + rogers_satchell(frame))[1:])


def test_realized_variance_is_never_negative_or_nan_where_data_exists(ohlc):
    frame, _ = ohlc
    for method in ("squared", "parkinson", "garman_klass", "rogers_satchell"):
        rv = realized_variance(frame, method, returns=np.log(frame["close"]).diff())
        assert np.all(rv[np.isfinite(rv)] > 0)


def test_realized_variance_rejects_bad_input(ohlc):
    frame, _ = ohlc
    with pytest.raises(ValueError, match="unknown proxy"):
        realized_variance(frame, "magic")
    with pytest.raises(TypeError, match="OHLC"):
        realized_variance(np.zeros(10), "parkinson")


def test_annualize_scales_by_the_period_count():
    assert annualize(np.array([1e-4]), 252.0)[0] == pytest.approx(np.sqrt(252e-4))


# --- VaR -------------------------------------------------------------------
def test_value_at_risk_is_negative_and_scales_with_sigma():
    normal = get_distribution("normal")
    v = value_at_risk(np.array([0.01, 0.02]), 0.01, normal)
    assert np.all(v < 0)
    assert v[1] == pytest.approx(2.0 * v[0])
    assert v[0] == pytest.approx(0.01 * normal.ppf(np.array([0.01]), ())[0])


def test_expected_shortfall_is_worse_than_value_at_risk():
    for name, params in [("normal", ()), ("t", (5.0,)), ("skewt", (6.0, -0.2))]:
        dist = get_distribution(name)
        sigma = np.array([0.012])
        assert expected_shortfall(sigma, 0.01, dist, params)[0] < \
               value_at_risk(sigma, 0.01, dist, params)[0]


def test_fat_tails_give_a_wider_one_percent_var_than_a_normal():
    sigma = np.array([0.01])
    normal = value_at_risk(sigma, 0.01, get_distribution("normal"))[0]
    student = value_at_risk(sigma, 0.01, get_distribution("t"), (4.0,))[0]
    assert student < normal


def test_kupiec_accepts_the_correct_rate_and_rejects_a_wrong_one():
    rng = np.random.default_rng(0)
    correct = (rng.random(5000) < 0.01).astype(int)
    assert kupiec_pof(correct, 0.01)[1] > 0.05
    too_many = (rng.random(5000) < 0.03).astype(int)
    assert kupiec_pof(too_many, 0.01)[1] < 0.01


def test_christoffersen_rejects_clustered_breaches():
    hits = np.zeros(4000, dtype=int)
    hits[1000:1040] = 1                       # every breach in one stretch
    assert christoffersen_independence(hits)[1] < 0.01
    rng = np.random.default_rng(1)
    scattered = (rng.random(4000) < 0.01).astype(int)
    assert christoffersen_independence(scattered)[1] > 0.05


def test_conditional_coverage_adds_the_two_statistics():
    rng = np.random.default_rng(2)
    hits = (rng.random(3000) < 0.05).astype(int)
    cc = conditional_coverage(hits, 0.05)[0]
    assert cc == pytest.approx(kupiec_pof(hits, 0.05)[0] + christoffersen_independence(hits)[0])


def test_var_backtest_passes_a_correctly_specified_model():
    rng = np.random.default_rng(3)
    sigma = np.full(6000, 0.01)
    r = rng.standard_normal(6000) * sigma
    normal = get_distribution("normal")
    report = backtest_var(r, value_at_risk(sigma, 0.01, normal),
                          expected_shortfall(sigma, 0.01, normal), 0.01)
    assert report.kupiec[1] > 0.05
    assert abs(report.breach_rate - 0.01) < 0.005
    assert 0.8 < report.es_backtest["ratio"] < 1.2
    assert "Kupiec" in str(report)


def test_var_backtest_rejects_a_normal_var_against_fat_tailed_reality():
    rng = np.random.default_rng(4)
    sigma = np.full(6000, 0.01)
    r = get_distribution("t").simulate(6000, (3.5,), rng) * sigma
    report = backtest_var(r, value_at_risk(sigma, 0.01, get_distribution("normal")),
                          None, 0.01)
    assert report.kupiec[1] < 0.01
    assert report.breach_rate > 0.01


def test_var_backtest_requires_aligned_inputs():
    with pytest.raises(ValueError, match="align"):
        backtest_var(np.zeros(10), np.zeros(9))
