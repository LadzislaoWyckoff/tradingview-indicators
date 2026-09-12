"""Command-line interface.

    python -m garchlab fit      --csv spx.csv --model gjr --dist skewt
    python -m garchlab forecast --csv spx.csv --horizon 22
    python -m garchlab backtest --csv spx.csv --train 2000 --refit 21
    python -m garchlab var      --csv spx.csv --alpha 0.01
    python -m garchlab compare  --csv spx.csv
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import __version__
from .backtest import walk_forward
from .data import load_prices, to_returns
from .diagnostics import diagnose
from .models import fit
from .realized import realized_variance
from .var import backtest_var, expected_shortfall, value_at_risk

MODELS = ("garch", "gjr", "egarch")
DISTS = ("normal", "t", "skewt")


def _add_data_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--csv", help="local OHLC csv; skips the network entirely")
    sub.add_argument("--symbol", default="^GSPC", help="ticker when downloading")
    sub.add_argument("--source", default="auto",
                     choices=("auto", "fmp", "yfinance", "stooq"),
                     help="fmp needs FMP_API_KEY in the environment")
    sub.add_argument("--start", help="first date, YYYY-MM-DD")
    sub.add_argument("--end", help="last date, YYYY-MM-DD")
    sub.add_argument("--periods-per-year", type=float, default=252.0)


def _add_model_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--model", default="gjr", choices=MODELS)
    sub.add_argument("--dist", default="skewt", choices=DISTS)
    sub.add_argument("--mean", default="constant", choices=("zero", "constant", "ar1"))
    sub.add_argument("-p", type=int, default=1, help="ARCH order")
    sub.add_argument("-q", type=int, default=1, help="GARCH order")


def _load(args) -> tuple[pd.DataFrame, pd.Series]:
    prices = load_prices(
        csv=args.csv, symbol=args.symbol, source=args.source,
        start=args.start, end=args.end,
    )
    returns = to_returns(prices)
    print(
        f"loaded {len(returns):,} returns  "
        f"{returns.index[0].date()} -> {returns.index[-1].date()}  "
        f"(sample annualized vol {float(returns.std()) * np.sqrt(args.periods_per_year):.2%})",
        file=sys.stderr,
    )
    return prices, returns


def _fit(args, returns):
    return fit(
        returns, model=args.model, p=args.p, q=args.q, dist=args.dist,
        mean=args.mean, periods_per_year=args.periods_per_year,
    )


def cmd_fit(args) -> int:
    _, returns = _load(args)
    res = _fit(args, returns)
    print(res.summary())
    print("\nspecification tests on the standardized residuals")
    for test in diagnose(res).values():
        print(f"  {test}")
    print(
        f"\ncurrent conditional volatility: {res.conditional_vol[-1]:.2%} annualized"
        f"   (long-run {res.long_run_vol:.2%})"
    )
    return 0


def cmd_forecast(args) -> int:
    _, returns = _load(args)
    res = _fit(args, returns)
    fc = res.forecast(horizon=args.horizon)
    print(res.summary())
    print(f"\nvolatility forecast, {args.horizon} periods ahead (annualized)")
    print(f"{'step':>6}{'per-day vol':>14}{'cumulative vol':>17}")
    for i, step in enumerate(fc["horizon"]):
        if step <= 10 or step % 5 == 0 or step == args.horizon:
            print(f"{step:>6}{fc['vol'][i]:>13.2%}{fc['cumulative_vol'][i]:>16.2%}")
    print(
        f"\nlast observed conditional vol {res.conditional_vol[-1]:.2%}"
        f"  ->  {args.horizon}-day average {fc['cumulative_vol'][-1]:.2%}"
        f"  (reverting to {res.long_run_vol:.2%}, half-life {res.half_life:.0f} days)"
    )
    if args.json:
        payload = {
            "as_of": str(returns.index[-1].date()),
            "model": f"{args.model}({args.p},{args.q})-{args.dist}",
            "params": dict(zip(res.param_names, map(float, res.params))),
            "persistence": res.persistence,
            "half_life_days": res.half_life,
            "long_run_vol": res.long_run_vol,
            "current_vol": float(res.conditional_vol[-1]),
            "horizon": [int(h) for h in fc["horizon"]],
            "vol": [float(v) for v in fc["vol"]],
            "cumulative_vol": [float(v) for v in fc["cumulative_vol"]],
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}", file=sys.stderr)
    return 0


def cmd_backtest(args) -> int:
    prices, returns = _load(args)
    ohlc = prices.loc[returns.index] if args.proxy != "squared" else None
    result = walk_forward(
        returns, model=args.model, p=args.p, q=args.q, dist=args.dist,
        mean=args.mean, train=args.train, refit=args.refit, horizon=args.horizon,
        window=args.window, ohlc=ohlc, proxy=args.proxy,
        periods_per_year=args.periods_per_year, verbose=args.verbose,
    )
    print(result.summary())
    if args.out:
        result.forecasts.to_csv(args.out)
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


def cmd_var(args) -> int:
    prices, returns = _load(args)
    result = walk_forward(
        returns, model=args.model, p=args.p, q=args.q, dist=args.dist,
        mean=args.mean, train=args.train, refit=args.refit, horizon=1,
        window=args.window, benchmarks=False,
        periods_per_year=args.periods_per_year,
    )
    frame = result.forecasts
    sigma = np.sqrt(frame[args.model].to_numpy())
    realized = returns.loc[frame.index].to_numpy()
    res = fit(returns, model=args.model, p=args.p, q=args.q, dist=args.dist,
              mean=args.mean, periods_per_year=args.periods_per_year)
    for alpha in args.alpha:
        levels = value_at_risk(sigma, alpha, res.dist, res.dist_params)
        es = expected_shortfall(sigma, alpha, res.dist, res.dist_params)
        print(backtest_var(realized, levels, es, alpha))
        print()
    sigma_now = float(np.sqrt(res.forecast(1, annualize=False)["variance"][0]) / res.scale)
    print(f"next-day forecast: sigma = {sigma_now:.2%}")
    for alpha in args.alpha:
        v = float(value_at_risk(np.array([sigma_now]), alpha, res.dist, res.dist_params)[0])
        e = float(expected_shortfall(np.array([sigma_now]), alpha, res.dist, res.dist_params)[0])
        print(f"  VaR({alpha:.0%}) = {v:.2%}    ES({alpha:.0%}) = {e:.2%}")
    return 0


def cmd_compare(args) -> int:
    _, returns = _load(args)
    rows = []
    fitted = {}
    for model in MODELS:
        for dist in DISTS:
            try:
                res = fit(returns, model=model, dist=dist, mean=args.mean,
                          periods_per_year=args.periods_per_year)
            except (ValueError, RuntimeError) as exc:
                print(f"  {model}-{dist} failed: {exc}", file=sys.stderr)
                continue
            tests = diagnose(res)
            key = f"{model}-{dist}"
            fitted[key] = res
            rows.append({
                "model": key,
                "loglik": res.loglikelihood,
                "AIC": res.aic,
                "BIC": res.bic,
                "persistence": res.persistence,
                "half_life": res.half_life,
                "long_run_vol": res.long_run_vol,
                "ARCH-LM p": tests["arch_lm"].pvalue,
                "converged": res.converged,
            })
    frame = pd.DataFrame(rows).sort_values("BIC").reset_index(drop=True)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(frame.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    best = frame.iloc[0]["model"]
    print(f"\nbest by BIC: {best}")
    print("BIC ranks in-sample fit only -- confirm with `backtest` before trusting it.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="garchlab",
        description="GARCH volatility modelling for the S&P 500 (and anything else daily).",
    )
    parser.add_argument("--version", action="version", version=f"garchlab {__version__}")
    subs = parser.add_subparsers(dest="command", required=True)

    p_fit = subs.add_parser("fit", help="estimate a model and run specification tests")
    _add_data_args(p_fit); _add_model_args(p_fit)
    p_fit.set_defaults(func=cmd_fit)

    p_fc = subs.add_parser("forecast", help="fit, then forecast volatility forward")
    _add_data_args(p_fc); _add_model_args(p_fc)
    p_fc.add_argument("--horizon", type=int, default=22, help="periods ahead")
    p_fc.add_argument("--json", help="also write the forecast to this json file")
    p_fc.set_defaults(func=cmd_forecast)

    p_bt = subs.add_parser("backtest", help="walk-forward out-of-sample evaluation")
    _add_data_args(p_bt); _add_model_args(p_bt)
    p_bt.add_argument("--train", type=int, default=1500)
    p_bt.add_argument("--refit", type=int, default=21)
    p_bt.add_argument("--horizon", type=int, default=1)
    p_bt.add_argument("--window", default="expanding", choices=("expanding", "rolling"))
    p_bt.add_argument("--proxy", default="squared",
                      choices=("squared", "parkinson", "garman_klass",
                               "rogers_satchell", "yang_zhang"))
    p_bt.add_argument("--out", help="write the forecast series to this csv")
    p_bt.add_argument("--verbose", action="store_true")
    p_bt.set_defaults(func=cmd_backtest)

    p_var = subs.add_parser("var", help="value-at-risk backtest and next-day levels")
    _add_data_args(p_var); _add_model_args(p_var)
    p_var.add_argument("--train", type=int, default=1500)
    p_var.add_argument("--refit", type=int, default=21)
    p_var.add_argument("--window", default="expanding", choices=("expanding", "rolling"))
    p_var.add_argument("--alpha", type=float, nargs="+", default=[0.01, 0.05])
    p_var.set_defaults(func=cmd_var)

    p_cmp = subs.add_parser("compare", help="fit every model/distribution pair and rank")
    _add_data_args(p_cmp)
    p_cmp.add_argument("--mean", default="constant", choices=("zero", "constant", "ar1"))
    p_cmp.set_defaults(func=cmd_compare)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
