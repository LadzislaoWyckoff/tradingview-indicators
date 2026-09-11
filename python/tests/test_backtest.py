"""Backtest engine: loss functions, benchmarks, and above all no look-ahead."""

import numpy as np
import pandas as pd
import pytest

from garchlab.backtest import (
    diebold_mariano,
    ewma_forecast,
    mae_vol,
    mincer_zanowitz,
    mse_variance,
    qlike,
    rolling_std_forecast,
    walk_forward,
)
from garchlab.models import simulate


def test_qlike_is_minimized_at_the_truth():
    rng = np.random.default_rng(0)
    true_var = np.full(20000, 1.5)
    rv = true_var * rng.chisquare(1, 20000)          # an unbiased, noisy proxy
    at_truth = qlike(rv, true_var)
    for wrong in (0.5, 0.9, 1.1, 2.0, 4.0):
        assert qlike(rv, true_var * wrong) > at_truth


def test_mse_is_minimized_at_the_truth():
    rng = np.random.default_rng(1)
    true_var = np.full(20000, 1.5)
    rv = true_var * rng.chisquare(1, 20000)
    assert mse_variance(rv, true_var) < mse_variance(rv, true_var * 1.5)


def test_qlike_punishes_under_forecasting_harder_than_over_forecasting():
    rv = np.full(1000, 1.0)
    assert qlike(rv, np.full(1000, 0.5)) > qlike(rv, np.full(1000, 2.0))


def test_losses_are_zero_for_a_perfect_forecast():
    rv = np.abs(np.random.default_rng(2).standard_normal(500)) + 0.1
    assert qlike(rv, rv) == pytest.approx(0.0)
    assert mse_variance(rv, rv) == pytest.approx(0.0)
    assert mae_vol(rv, rv) == pytest.approx(0.0)


def test_mincer_zanowitz_on_a_perfect_forecast_gives_alpha_zero_beta_one():
    rv = np.abs(np.random.default_rng(3).standard_normal(2000)) + 0.1
    mz = mincer_zanowitz(rv, rv)
    assert mz["alpha"] == pytest.approx(0.0, abs=1e-8)
    assert mz["beta"] == pytest.approx(1.0, abs=1e-8)
    assert mz["r2"] == pytest.approx(1.0)


def test_mincer_zanowitz_detects_a_scaled_forecast():
    rng = np.random.default_rng(4)
    truth = np.exp(rng.standard_normal(4000) * 0.4)
    rv = truth * rng.chisquare(4, 4000) / 4.0
    mz = mincer_zanowitz(rv, truth * 2.0)
    assert mz["beta"] == pytest.approx(0.5, abs=0.1)   # forecast twice too big
    assert mz["pvalue"] < 0.01                        # and the test says so


def test_diebold_mariano_picks_the_better_forecast():
    rng = np.random.default_rng(5)
    truth = np.exp(rng.standard_normal(4000) * 0.5)
    rv = truth * rng.chisquare(2, 4000) / 2.0
    good = truth
    bad = np.full(4000, float(truth.mean()))
    dm = diebold_mariano(rv, good, bad, loss="qlike")
    assert dm["statistic"] < 0 and dm["better"] == "A"
    assert dm["pvalue"] < 0.01
    flipped = diebold_mariano(rv, bad, good, loss="qlike")
    assert flipped["statistic"] == pytest.approx(-dm["statistic"], rel=1e-9)


def test_diebold_mariano_finds_no_difference_between_identical_forecasts():
    rng = np.random.default_rng(6)
    rv = np.abs(rng.standard_normal(2000)) + 0.1
    f = np.full(2000, 1.0)
    dm = diebold_mariano(rv, f, f.copy(), loss="qlike")
    assert not np.isfinite(dm["statistic"]) or abs(dm["statistic"]) < 1e-6


def test_ewma_forecast_uses_only_past_data():
    r = np.random.default_rng(7).standard_normal(500) * 0.01
    base = ewma_forecast(r)
    bumped = r.copy()
    bumped[300:] *= 10.0
    assert ewma_forecast(bumped)[:301] == pytest.approx(base[:301])


def test_rolling_std_forecast_uses_only_past_data():
    r = np.random.default_rng(8).standard_normal(500) * 0.01
    base = rolling_std_forecast(r, window=60)
    bumped = r.copy()
    bumped[300:] *= 10.0
    got = rolling_std_forecast(bumped, window=60)[:301]
    assert np.allclose(got, base[:301], equal_nan=True)


def test_ewma_matches_the_riskmetrics_recursion():
    r = np.random.default_rng(9).standard_normal(200) * 0.01
    out = ewma_forecast(r, lam=0.94, warmup=60)
    lam = 0.94
    h = float(np.var(r[:60]))
    for t in range(200):
        assert out[t] == pytest.approx(h)
        h = lam * h + (1.0 - lam) * r[t] ** 2


@pytest.fixture(scope="module")
def series():
    return simulate(1200, "garch", [0.05, 0.08, 0.90], seed=13)[0]


def test_walk_forward_forecasts_do_not_use_future_returns(series):
    """The test that matters.

    Changing the tail of the series must leave every earlier forecast exactly
    where it was.  If it does not, the backtest is peeking and every number it
    produces is worthless.
    """
    cut = 900
    base = walk_forward(series, model="garch", dist="normal", train=600,
                        refit=100, benchmarks=False)
    tampered = series.copy()
    tampered[cut:] *= 6.0
    other = walk_forward(tampered, model="garch", dist="normal", train=600,
                         refit=100, benchmarks=False)
    a = base.forecasts["garch"].loc[base.forecasts.index < cut]
    b = other.forecasts["garch"].loc[other.forecasts.index < cut]
    assert len(a) > 100
    assert a.to_numpy() == pytest.approx(b.to_numpy(), rel=1e-12)


def test_walk_forward_beats_the_unconditional_benchmark(series):
    bt = walk_forward(series, model="garch", dist="normal", train=600, refit=100)
    assert bt.losses.loc["garch", "qlike"] < bt.losses.loc["unconditional", "qlike"]
    assert bt.settings["n_failed_refits"] == 0
    assert "unconditional" in bt.dm and np.isfinite(bt.dm["unconditional"]["statistic"])


def test_walk_forward_refits_on_the_stated_schedule(series):
    bt = walk_forward(series, model="garch", dist="normal", train=600,
                      refit=50, benchmarks=False)
    assert bt.refits[0] == 600
    assert np.all(np.diff(bt.refits) == 50)


def test_multi_period_horizon_targets_the_aggregate_variance(series):
    bt = walk_forward(series, model="garch", dist="normal", train=600,
                      refit=100, horizon=5, benchmarks=True)
    # a 5-day variance should sit near 5x the 1-day variance
    one_day = walk_forward(series, model="garch", dist="normal", train=600,
                           refit=100, horizon=1, benchmarks=False)
    ratio = bt.forecasts["garch"].mean() / one_day.forecasts["garch"].mean()
    assert 3.0 < ratio < 7.0
    assert bt.settings["horizon"] == 5


def test_rolling_window_keeps_the_sample_length_fixed(series):
    bt = walk_forward(series, model="garch", dist="normal", train=600,
                      refit=100, window="rolling", benchmarks=False)
    assert bt.settings["window"] == "rolling"
    assert bt.settings["n_forecasts"] > 500


def test_walk_forward_rejects_impossible_settings(series):
    with pytest.raises(ValueError, match="need more than"):
        walk_forward(series, train=len(series), horizon=1)
    with pytest.raises(ValueError, match="expanding"):
        walk_forward(series, train=600, window="sliding")


def test_summary_renders(series):
    bt = walk_forward(series, model="garch", dist="normal", train=600, refit=100)
    text = bt.summary()
    assert "QLIKE" in text and "Diebold-Mariano" in text and "ewma(0.94)" in text
