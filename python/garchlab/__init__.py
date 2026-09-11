"""garchlab -- GARCH-family volatility modelling and forecasting.

Built for one job: take a daily price series (the S&P 500 index, by default),
estimate a conditional-variance model on it, forecast volatility forward, and
then prove out-of-sample whether that forecast is worth anything.

Quick start
-----------
>>> from garchlab import load_prices, to_returns, fit, walk_forward
>>> prices = load_prices(csv="spx.csv")
>>> r = to_returns(prices)
>>> res = fit(r, model="gjr", dist="skewt")
>>> print(res.summary())
>>> print(res.forecast(horizon=22)["vol"][:5])      # annualized, next 5 days
>>> print(walk_forward(r, model="gjr", train=2000).summary())
"""

from .backtest import (
    BacktestResult,
    diebold_mariano,
    ewma_forecast,
    mincer_zanowitz,
    mse_variance,
    qlike,
    rolling_std_forecast,
    walk_forward,
)
from .data import SP500_SYMBOLS, download, load_csv, load_prices, to_returns
from .diagnostics import arch_lm, diagnose, ljung_box, news_impact_curve
from .distributions import Normal, SkewT, StudentT, get_distribution
from .models import (
    EGARCH,
    GARCH,
    GARCHResult,
    GJRGARCH,
    build_model,
    fit,
    simulate,
)
from .realized import realized_variance
from .var import backtest_var, expected_shortfall, value_at_risk

__version__ = "0.1.0"

__all__ = [
    "fit", "simulate", "build_model", "GARCH", "GJRGARCH", "EGARCH", "GARCHResult",
    "walk_forward", "BacktestResult", "qlike", "mse_variance", "diebold_mariano",
    "mincer_zanowitz", "ewma_forecast", "rolling_std_forecast",
    "load_prices", "load_csv", "download", "to_returns", "SP500_SYMBOLS",
    "diagnose", "ljung_box", "arch_lm", "news_impact_curve",
    "realized_variance", "value_at_risk", "expected_shortfall", "backtest_var",
    "get_distribution", "Normal", "StudentT", "SkewT",
    "__version__",
]
