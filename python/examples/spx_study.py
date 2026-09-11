"""End-to-end volatility study on a daily price series.

Runs the whole pipeline in the order you would actually work through it:

1. fit every model/distribution pair and rank them in-sample (BIC)
2. run the specification battery on the winner
3. walk-forward backtest against EWMA and a rolling standard deviation
4. VaR backtest at 1% and 5%
5. print the current forecast

    python examples/spx_study.py                      # bundled synthetic data
    python examples/spx_study.py --csv real_spx.csv   # your own export
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from garchlab import (
    backtest_var,
    diagnose,
    expected_shortfall,
    fit,
    load_prices,
    to_returns,
    value_at_risk,
    walk_forward,
)

DEFAULT_CSV = Path(__file__).resolve().parent / "sample_spx.csv"


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--start")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--refit", type=int, default=63)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--proxy", default="garman_klass")
    args = ap.parse_args()

    prices = load_prices(csv=args.csv, start=args.start)
    returns = to_returns(prices)
    print(f"{len(returns):,} returns, {returns.index[0].date()} -> {returns.index[-1].date()}")
    print(f"sample annualized volatility: {float(returns.std()) * np.sqrt(252):.2%}")

    rule("1. model selection (in-sample, ranked by BIC)")
    rows, fitted = [], {}
    for model in ("garch", "gjr", "egarch"):
        for dist in ("normal", "t", "skewt"):
            t0 = time.time()
            res = fit(returns, model=model, dist=dist)
            key = f"{model}-{dist}"
            fitted[key] = res
            rows.append({
                "model": key, "loglik": res.loglikelihood, "BIC": res.bic,
                "persistence": res.persistence, "half-life": res.half_life,
                "long-run vol": res.long_run_vol, "secs": time.time() - t0,
            })
            print(f"  fitted {key:<16} BIC={res.bic:12,.1f}  ({time.time() - t0:.1f}s)")
    table = pd.DataFrame(rows).sort_values("BIC").reset_index(drop=True)
    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    best_name = table.iloc[0]["model"]
    best = fitted[best_name]
    print(f"\nbest in-sample: {best_name}")

    rule(f"2. specification tests for {best_name}")
    print(best.summary())
    print()
    for test in diagnose(best).values():
        print(f"  {test}")

    rule(f"3. walk-forward backtest (horizon={args.horizon}d, proxy={args.proxy})")
    model, dist = best_name.split("-")
    t0 = time.time()
    bt = walk_forward(
        returns, model=model, dist=dist, train=args.train, refit=args.refit,
        horizon=args.horizon, ohlc=prices.loc[returns.index], proxy=args.proxy,
    )
    print(bt.summary())
    print(f"\n({time.time() - t0:.0f}s)")

    rule("4. value-at-risk backtest")
    one_step = bt if args.horizon == 1 else walk_forward(
        returns, model=model, dist=dist, train=args.train, refit=args.refit,
        horizon=1, benchmarks=False,
    )
    sigma = np.sqrt(one_step.forecasts[model].to_numpy())
    realized = returns.loc[one_step.forecasts.index].to_numpy()
    for alpha in (0.01, 0.05):
        levels = value_at_risk(sigma, alpha, best.dist, best.dist_params)
        es = expected_shortfall(sigma, alpha, best.dist, best.dist_params)
        print(backtest_var(realized, levels, es, alpha))
        print()

    rule("5. current forecast")
    fc = best.forecast(horizon=22)
    print(f"as of {returns.index[-1].date()}")
    print(f"  conditional volatility now : {best.conditional_vol[-1]:.2%}")
    print(f"  1-day ahead                : {fc['vol'][0]:.2%}")
    print(f"  22-day average             : {fc['cumulative_vol'][-1]:.2%}")
    print(f"  long-run level             : {best.long_run_vol:.2%}"
          f"  (half-life {best.half_life:.0f} days)")
    sigma_now = float(np.sqrt(fc["variance"][0]) / best.scale)
    for alpha in (0.01, 0.05):
        v = float(value_at_risk(np.array([sigma_now]), alpha, best.dist, best.dist_params)[0])
        e = float(expected_shortfall(np.array([sigma_now]), alpha, best.dist, best.dist_params)[0])
        print(f"  VaR({alpha:.0%}) tomorrow      : {v:.2%}   ES: {e:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
