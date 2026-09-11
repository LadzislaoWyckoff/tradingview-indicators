"""Generate a synthetic S&P-500-like OHLC series.

This exists so every example in the README runs on a machine with no market
data access.  It is *simulated*, not real: the parameters are chosen to look
like SPX daily data (roughly 16% long-run volatility, a strong leverage
effect, near-unit persistence, fat and left-skewed innovations), but nothing
here is an observation of the actual index.  Point the CLI at a real CSV
export for real work.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from garchlab.models import simulate

# GJR-GARCH(1,1) on percent returns with skewed-t innovations.
# persistence = alpha + kappa*gamma + beta ~ 0.96, which reproduces the
# headline statistics of real SPX daily data: ~16% annualized volatility,
# kurtosis around 11, skew around -0.7 and a worst day near -11%.
PARAMS = [0.025, 0.015, 0.09, 0.905]
DIST_PARAMS = (9.0, -0.10)
DRIFT = 0.08 / 252.0          # ~8% a year
N_DAYS = 7000                 # about 28 years of trading days


def main(out: str = "sample_spx.csv", seed: int = 20240917) -> Path:
    returns, vol = simulate(
        N_DAYS, "gjr", PARAMS, dist="skewt", dist_params=DIST_PARAMS,
        mu=DRIFT, seed=seed,
    )
    rng = np.random.default_rng(seed + 1)
    close = 350.0 * np.exp(np.cumsum(returns))
    prev_close = np.concatenate(([350.0], close[:-1]))

    # split the day's move into an overnight gap and an intraday path, then
    # read the high and low off a Brownian bridge so the range is consistent
    # with the day's own volatility instead of being made up independently
    gap_share = rng.beta(2.0, 5.0, N_DAYS)
    open_ = prev_close * np.exp(np.log(close / prev_close) * gap_share)
    steps = 390                                   # one point a minute
    walk = np.cumsum(rng.standard_normal((N_DAYS, steps)), axis=1) / np.sqrt(steps)
    clock = np.linspace(0.0, 1.0, steps)
    bridge = walk - clock * walk[:, -1:]          # pinned at both ends
    intraday = np.log(close / open_)[:, None] * clock + bridge * (
        vol[:, None] * np.sqrt(1.0 - gap_share)[:, None]
    )
    high = open_ * np.exp(np.maximum(intraday.max(axis=1), np.maximum(0.0, np.log(close / open_))))
    low = open_ * np.exp(np.minimum(intraday.min(axis=1), np.minimum(0.0, np.log(close / open_))))

    dates = pd.bdate_range("1998-01-02", periods=N_DAYS)
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": open_.round(4),
            "High": high.round(4),
            "Low": low.round(4),
            "Close": close.round(4),
            "Volume": (rng.lognormal(18.0, 0.35, N_DAYS)).round(0),
        }
    )
    path_out = Path(__file__).resolve().parent / out
    frame.to_csv(path_out, index=False)
    ann = float(np.std(returns) * np.sqrt(252))
    print(f"wrote {path_out}  ({len(frame):,} rows, realized annualized vol {ann:.2%})")
    return path_out


if __name__ == "__main__":
    main()
