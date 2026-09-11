"""The distributions must be standardized -- everything downstream assumes it."""

import numpy as np
import pytest

from garchlab.distributions import Distribution, Normal, SkewT, StudentT, get_distribution

CASES = [(Normal(), ()), (StudentT(), (6.0,)), (StudentT(), (20.0,)),
         (SkewT(), (6.0, -0.3)), (SkewT(), (10.0, 0.2))]


@pytest.mark.parametrize("dist,params", CASES)
def test_density_integrates_to_one_with_zero_mean_unit_variance(dist, params):
    z = np.linspace(-40.0, 40.0, 800_001)
    f = np.exp(dist.loglikelihood(z, params))
    dz = z[1] - z[0]
    assert np.trapezoid(f, z) == pytest.approx(1.0, abs=1e-4)
    assert float(np.sum(z * f) * dz) == pytest.approx(0.0, abs=2e-3)
    assert float(np.sum(z * z * f) * dz) == pytest.approx(1.0, abs=3e-3)


@pytest.mark.parametrize("dist,params", CASES)
def test_ppf_inverts_cdf(dist, params):
    q = np.array([0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
    assert dist.cdf(dist.ppf(q, params), params) == pytest.approx(q, abs=1e-9)


@pytest.mark.parametrize("dist,params", CASES)
def test_simulated_draws_match_the_stated_moments(dist, params):
    z = dist.simulate(400_000, params, np.random.default_rng(0))
    assert float(z.mean()) == pytest.approx(0.0, abs=0.02)
    assert float(z.std()) == pytest.approx(1.0, abs=0.03)


def test_student_t_mean_abs_closed_form_matches_quadrature():
    for nu in (3.0, 6.0, 10.0, 50.0):
        closed = StudentT().mean_abs((nu,))
        quad = Distribution.mean_abs(StudentT(), (nu,))
        assert closed == pytest.approx(quad, rel=2e-3)


def test_normal_mean_abs_is_sqrt_two_over_pi():
    assert Normal().mean_abs(()) == pytest.approx(np.sqrt(2.0 / np.pi))


def test_negative_skew_puts_more_mass_below_zero_in_the_tail():
    left = SkewT()
    assert left.expected_shortfall(0.05, (6.0, -0.3)) < left.expected_shortfall(0.05, (6.0, 0.0))
    # a left-skewed density has its median above zero, so P(z < 0) drops
    assert left.prob_negative((6.0, -0.3)) < 0.5
    assert left.prob_negative((6.0, 0.3)) > 0.5


def test_symmetric_skewt_collapses_to_student_t():
    z = np.linspace(-5.0, 5.0, 101)
    assert SkewT().loglikelihood(z, (8.0, 0.0)) == pytest.approx(
        StudentT().loglikelihood(z, (8.0,)), abs=1e-10
    )


def test_expected_shortfall_is_below_value_at_risk():
    for dist, params in CASES:
        q = float(dist.ppf(np.array([0.05]), params)[0])
        assert dist.expected_shortfall(0.05, params) < q


def test_registry_rejects_unknown_names():
    assert isinstance(get_distribution("skew-t"), SkewT)
    assert isinstance(get_distribution("Normal"), Normal)
    with pytest.raises(ValueError, match="unknown distribution"):
        get_distribution("cauchy")
