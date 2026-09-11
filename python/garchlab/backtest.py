"""Out-of-sample evaluation of volatility forecasts.

In-sample fit tells you almost nothing about a volatility model.  Every GARCH
variant will track realized volatility beautifully over the window it was
estimated on.  The only question that matters is whether it beats the cheap
alternatives -- a RiskMetrics EWMA and a rolling standard deviation -- on data
it has never seen, and that is what this module measures.

Two details make the difference between an honest backtest and a flattering
one:

* **Parameters are re-estimated only every ``refit`` days**, and between
  refits the recursion is simply carried forward on new data.  Refitting daily
  is both unrealistic and slow; freezing parameters for a month is what you
  would actually do.
* **Losses are QLIKE and MSE.**  Patton (2011) shows these two are robust to
  noise in the realized-variance proxy: the model that minimizes them on a
  noisy proxy is the same one that would minimize them on the true, unobserved
  variance.  Most other loss functions -- including anything computed on
  volatility rather than variance, and R-squared -- do not have that property
  and can rank models wrongly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .models import GARCHResult, build_model, fit
from .realized import realized_variance

__all__ = [
    "qlike", "mse_variance", "mae_vol", "mincer_zanowitz", "diebold_mariano",
    "ewma_forecast", "rolling_std_forecast", "walk_forward", "BacktestResult",
]


# ---------------------------------------------------------------------------
# loss functions (all on *variance*, all lower-is-better)
# ---------------------------------------------------------------------------
def qlike(realized, forecast) -> float:
    """QLIKE loss: rv/f - log(rv/f) - 1.

    Asymmetric on purpose -- it punishes under-forecasting variance much harder
    than over-forecasting, which matches the asymmetry of the consequences.
    Minimized at f = rv, and robust to noise in ``realized``.
    """
    rv = np.asarray(realized, dtype=float)
    f = np.asarray(forecast, dtype=float)
    ok = np.isfinite(rv) & np.isfinite(f) & (rv > 0) & (f > 0)
    ratio = rv[ok] / f[ok]
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mse_variance(realized, forecast) -> float:
    rv = np.asarray(realized, dtype=float)
    f = np.asarray(forecast, dtype=float)
    ok = np.isfinite(rv) & np.isfinite(f)
    return float(np.mean((rv[ok] - f[ok]) ** 2))


def mae_vol(realized, forecast) -> float:
    """Mean absolute error in volatility units.  Easy to read, but not a robust
    loss -- use it for reporting, rank models on QLIKE."""
    rv = np.asarray(realized, dtype=float)
    f = np.asarray(forecast, dtype=float)
    ok = np.isfinite(rv) & np.isfinite(f) & (rv >= 0) & (f >= 0)
    return float(np.mean(np.abs(np.sqrt(rv[ok]) - np.sqrt(f[ok]))))


def _loss_series(realized, forecast, loss: str) -> np.ndarray:
    rv = np.asarray(realized, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if loss == "qlike":
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = rv / f
            out = ratio - np.log(ratio) - 1.0
        return out
    if loss == "mse":
        return (rv - f) ** 2
    if loss == "mae":
        return np.abs(np.sqrt(np.maximum(rv, 0)) - np.sqrt(np.maximum(f, 0)))
    raise ValueError(f"unknown loss {loss!r}")


def _newey_west_var(d: np.ndarray, lags: int | None = None) -> float:
    """HAC variance of a mean, Bartlett kernel."""
    d = d[np.isfinite(d)]
    n = d.size
    if n < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(lags, n - 2))
    e = d - d.mean()
    gamma0 = float(np.dot(e, e)) / n
    total = gamma0
    for k in range(1, lags + 1):
        gamma = float(np.dot(e[k:], e[:-k])) / n
        total += 2.0 * (1.0 - k / (lags + 1.0)) * gamma
    return max(total, 1e-300) / n


def diebold_mariano(realized, forecast_a, forecast_b, loss: str = "qlike",
                    lags: int | None = None) -> dict:
    """Diebold-Mariano test of equal predictive accuracy.

    Negative statistic means model A has the lower loss.  Standard errors are
    Newey-West, because loss differentials at overlapping horizons are
    autocorrelated by construction.
    """
    la = _loss_series(realized, forecast_a, loss)
    lb = _loss_series(realized, forecast_b, loss)
    d = la - lb
    d = d[np.isfinite(d)]
    if d.size < 10:
        return {"statistic": float("nan"), "pvalue": float("nan"), "n": int(d.size)}
    var = _newey_west_var(d, lags)
    stat = float(np.mean(d) / np.sqrt(var))
    return {
        "statistic": stat,
        "pvalue": float(2.0 * (1.0 - stats.norm.cdf(abs(stat)))),
        "mean_diff": float(np.mean(d)),
        "n": int(d.size),
        "better": "A" if stat < 0 else "B",
    }


def mincer_zanowitz(realized, forecast) -> dict:
    """Regression rv = a + b * forecast, with a joint test of (a, b) = (0, 1).

    ``b`` below 1 is the usual finding and means the forecast over-reacts:
    scaling it toward its own mean would improve it.  Reported with HAC
    standard errors.
    """
    rv = np.asarray(realized, dtype=float)
    f = np.asarray(forecast, dtype=float)
    ok = np.isfinite(rv) & np.isfinite(f)
    rv, f = rv[ok], f[ok]
    n = rv.size
    x = np.column_stack([np.ones(n), f])
    beta, *_ = np.linalg.lstsq(x, rv, rcond=None)
    resid = rv - x @ beta
    xtx_inv = np.linalg.pinv(x.T @ x)
    meat = (x * resid[:, None]).T @ (x * resid[:, None])
    cov = xtx_inv @ meat @ xtx_inv
    diff = beta - np.array([0.0, 1.0])
    try:
        wald = float(diff @ np.linalg.pinv(cov) @ diff)
        pvalue = float(stats.chi2.sf(wald, 2))
    except np.linalg.LinAlgError:  # pragma: no cover
        wald, pvalue = float("nan"), float("nan")
    ss_tot = float(np.sum((rv - rv.mean()) ** 2))
    r2 = 0.0 if ss_tot == 0 else 1.0 - float(np.sum(resid ** 2)) / ss_tot
    return {
        "alpha": float(beta[0]),
        "beta": float(beta[1]),
        "se_alpha": float(np.sqrt(abs(cov[0, 0]))),
        "se_beta": float(np.sqrt(abs(cov[1, 1]))),
        "r2": r2,
        "wald": wald,
        "pvalue": pvalue,
        "n": n,
    }


# ---------------------------------------------------------------------------
# cheap benchmarks worth beating
# ---------------------------------------------------------------------------
def ewma_forecast(returns, lam: float = 0.94, warmup: int = 60) -> np.ndarray:
    """RiskMetrics EWMA one-step variance forecasts.

    ``out[t]`` is the forecast for day ``t`` made with data through ``t-1``.
    This is the benchmark to beat; it is one line of code and it is good.
    """
    r = np.asarray(getattr(returns, "values", returns), dtype=float)
    n = r.size
    out = np.full(n, np.nan)
    h = float(np.var(r[:warmup])) if n > warmup else float(np.var(r))
    for t in range(n):
        out[t] = h
        h = lam * h + (1.0 - lam) * r[t] ** 2
    return out


def rolling_std_forecast(returns, window: int = 60) -> np.ndarray:
    """Rolling sample-variance forecast, lagged so it uses only past data."""
    r = pd.Series(np.asarray(getattr(returns, "values", returns), dtype=float))
    return r.rolling(window).var(ddof=1).shift(1).to_numpy()


# ---------------------------------------------------------------------------
# walk-forward engine
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    forecasts: pd.DataFrame
    losses: pd.DataFrame
    mz: dict
    dm: dict
    settings: dict
    refits: list = field(default_factory=list)

    def summary(self) -> str:
        s = self.settings
        lines = [
            f"Walk-forward backtest: {s['model']}({s['p']},{s['q']})-{s['dist']}"
            f"  horizon={s['horizon']}d",
            f"  {s['n_forecasts']} out-of-sample forecasts, train={s['train']}, "
            f"refit every {s['refit']} days ({len(self.refits)} refits, "
            f"{s['n_failed_refits']} failed)",
            f"  realized-variance proxy: {s['proxy']}",
            "",
        ]
        header = f"{'model':<22}{'QLIKE':>12}{'MSE':>14}{'MAE(vol)':>12}{'MZ R2':>9}{'MZ beta':>9}"
        lines.append(header)
        lines.append("-" * len(header))
        for name, row in self.losses.iterrows():
            mz = self.mz.get(name, {})
            lines.append(
                f"{name:<22}{row['qlike']:>12.6f}{row['mse']:>14.3e}"
                f"{row['mae_vol']:>12.6f}{mz.get('r2', float('nan')):>9.3f}"
                f"{mz.get('beta', float('nan')):>9.3f}"
            )
        lines.append("-" * len(header))
        lines.append("lower QLIKE / MSE is better; MZ beta of 1.0 means unbiased")
        if self.dm:
            lines.append("")
            lines.append("Diebold-Mariano vs the GARCH model (QLIKE loss):")
            for name, d in self.dm.items():
                if not np.isfinite(d.get("statistic", np.nan)):
                    continue
                verdict = (
                    "GARCH better" if d["statistic"] < 0 else f"{name} better"
                ) + (" (significant)" if d["pvalue"] < 0.05 else " (not significant)")
                lines.append(
                    f"  vs {name:<18} DM={d['statistic']:7.3f}  p={d['pvalue']:.4f}   {verdict}"
                )
        return "\n".join(lines)


def _aggregate_realized(rv: np.ndarray, horizon: int) -> np.ndarray:
    """Forward-looking sum of the next ``horizon`` daily realized variances."""
    if horizon == 1:
        return rv
    n = rv.size
    out = np.full(n, np.nan)
    csum = np.concatenate(([0.0], np.nancumsum(rv)))
    for t in range(n - horizon + 1):
        window = rv[t : t + horizon]
        if np.all(np.isfinite(window)):
            out[t] = csum[t + horizon] - csum[t]
    return out


def walk_forward(
    returns,
    model: str = "gjr",
    p: int = 1,
    q: int = 1,
    dist: str = "skewt",
    mean: str = "constant",
    train: int = 1000,
    refit: int = 21,
    horizon: int = 1,
    window: str = "expanding",
    ohlc: pd.DataFrame | None = None,
    proxy: str = "squared",
    benchmarks: bool = True,
    periods_per_year: float = 252.0,
    verbose: bool = False,
) -> BacktestResult:
    """Run a walk-forward volatility forecast backtest.

    Parameters
    ----------
    returns
        Log returns, one per period, in original units.
    train
        Length of the initial estimation sample.
    refit
        Re-estimate the parameters every this many observations.  Between
        refits the variance recursion is rolled forward on the new data with
        the parameters held fixed -- the same thing you would do live.
    horizon
        Forecast horizon in periods.  With ``horizon > 1`` the target is the
        variance of the h-period aggregate return, so the forecast windows
        overlap and the Diebold-Mariano test widens its HAC lags accordingly.
    window
        ``"expanding"`` uses all history at each refit; ``"rolling"`` keeps the
        sample length fixed at ``train``.
    ohlc / proxy
        Supply an OHLC frame and a range-based ``proxy`` to score against a far
        less noisy realized-variance estimate than the squared return.
    """
    r = np.asarray(getattr(returns, "values", returns), dtype=float)
    index = getattr(returns, "index", pd.RangeIndex(r.size))
    n = r.size
    if n <= train + horizon:
        raise ValueError(f"need more than train + horizon = {train + horizon} observations")
    if window not in {"expanding", "rolling"}:
        raise ValueError("window must be 'expanding' or 'rolling'")

    vol_model = build_model(model, p, q)
    n_lag = max(vol_model.p, vol_model.q, 1)
    scale = 100.0

    forecasts = np.full(n, np.nan)     # variance of the h-period aggregate return
    current: GARCHResult | None = None
    refit_points: list[int] = []
    failed = 0

    # state carried between refits, all on the estimation scale
    resid_state = np.empty(0)
    sigma2_state = np.empty(0)

    t = train
    while t < n - horizon + 1:
        need_refit = current is None or (t - train) % refit == 0
        if need_refit:
            start = 0 if window == "expanding" else max(0, t - train)
            try:
                current = fit(
                    r[start:t], model=model, p=p, q=q, dist=dist, mean=mean,
                    periods_per_year=periods_per_year, scale=scale,
                    robust_errors=False, restarts=0,
                )
                refit_points.append(t)
                resid_state = current.resids.copy()
                sigma2_state = current.conditional_variance.copy()
                if verbose:
                    print(f"  refit at {t}: persistence={current.persistence:.4f}")
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                failed += 1
                if current is None:
                    t += 1
                    continue

        assert current is not None
        # forecast made with information through t-1, for days t .. t+h-1
        var_path = current.vol_model.forecast_variance(
            current.vol_params, horizon,
            sigma2_state[-n_lag:], resid_state[-n_lag:],
            current.dist, current.dist_params,
        )
        forecasts[t] = float(np.sum(var_path)) / (scale ** 2)

        # roll the recursion forward one day on the newly observed return.
        # var_path[0] *is* the one-step-ahead conditional variance for day t,
        # so it becomes sigma2_t; no second forecast call needed.
        y_t = r[t] * scale
        prev_y = r[t - 1] * scale
        resid_t = y_t - current.mean_model.next_mean(current.mean_params, prev_y)
        resid_state = np.append(resid_state, resid_t)
        sigma2_state = np.append(sigma2_state, var_path[0])
        t += 1

    # -- the target ------------------------------------------------------
    rv_daily = realized_variance(
        ohlc if ohlc is not None else r, method=proxy, returns=r
    )
    target = _aggregate_realized(rv_daily, horizon)

    columns = {"realized": target, model: forecasts}
    if benchmarks:
        ewma = ewma_forecast(r) * horizon
        roll = rolling_std_forecast(r, window=60) * horizon
        uncond = pd.Series(r).expanding(min_periods=train).var(ddof=1).shift(1).to_numpy() * horizon
        columns["ewma(0.94)"] = ewma
        columns["rolling_std(60)"] = roll
        columns["unconditional"] = uncond

    frame = pd.DataFrame(columns, index=index)
    mask = np.isfinite(frame["realized"].to_numpy()) & np.isfinite(forecasts)
    frame = frame.loc[mask]

    model_names = [c for c in frame.columns if c != "realized"]
    rv = frame["realized"].to_numpy()
    losses = pd.DataFrame(
        {
            "qlike": [qlike(rv, frame[c].to_numpy()) for c in model_names],
            "mse": [mse_variance(rv, frame[c].to_numpy()) for c in model_names],
            "mae_vol": [mae_vol(rv, frame[c].to_numpy()) for c in model_names],
        },
        index=model_names,
    )
    mz = {c: mincer_zanowitz(rv, frame[c].to_numpy()) for c in model_names}
    dm = {
        c: diebold_mariano(rv, frame[model].to_numpy(), frame[c].to_numpy(),
                           loss="qlike", lags=None if horizon == 1 else horizon + 5)
        for c in model_names
        if c != model
    }

    return BacktestResult(
        forecasts=frame,
        losses=losses,
        mz=mz,
        dm=dm,
        refits=refit_points,
        settings={
            "model": model, "p": p, "q": q, "dist": dist, "mean": mean,
            "train": train, "refit": refit, "horizon": horizon,
            "window": window, "proxy": proxy,
            "n_forecasts": int(frame.shape[0]),
            "n_failed_refits": failed,
        },
    )
