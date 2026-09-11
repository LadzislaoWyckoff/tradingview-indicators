"""Realized-volatility proxies.

A volatility forecast has to be scored against *something*, and the something
is never observed.  The default choice -- the squared daily return -- is an
unbiased proxy for the day's variance but an extremely noisy one: its
signal-to-noise ratio is roughly 1:2, which is why naive R-squared numbers for
volatility forecasts look so bad even when the forecasts are good.

Range-based estimators use the high and low as well as the close and are far
more efficient (Parkinson is about five times as efficient as the squared
return, Garman-Klass about seven).  When OHLC data is available, prefer them:
the same forecast will score dramatically better, not because it improved but
because the yardstick stopped shaking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "squared_return",
    "parkinson",
    "garman_klass",
    "rogers_satchell",
    "yang_zhang",
    "realized_variance",
    "annualize",
]

_LOG4 = float(np.log(4.0))


def _col(frame: pd.DataFrame, name: str) -> np.ndarray:
    lowered = {str(c).lower(): c for c in frame.columns}
    if name not in lowered:
        raise KeyError(f"frame has no {name!r} column; got {list(frame.columns)}")
    return frame[lowered[name]].to_numpy(dtype=float)


def squared_return(returns) -> np.ndarray:
    """The unbiased but very noisy default proxy."""
    r = np.asarray(getattr(returns, "values", returns), dtype=float)
    return r * r


def parkinson(frame: pd.DataFrame) -> np.ndarray:
    """Parkinson (1980) high-low range variance.

    Assumes a driftless diffusion and ignores overnight gaps, so it
    understates variance on gap-heavy days.
    """
    hi, lo = _col(frame, "high"), _col(frame, "low")
    return (np.log(hi / lo) ** 2) / (4.0 * np.log(2.0))


def garman_klass(frame: pd.DataFrame) -> np.ndarray:
    """Garman-Klass (1980), using open, high, low and close."""
    op, hi, lo, cl = (_col(frame, c) for c in ("open", "high", "low", "close"))
    log_hl = np.log(hi / lo)
    log_co = np.log(cl / op)
    return 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2


def rogers_satchell(frame: pd.DataFrame) -> np.ndarray:
    """Rogers-Satchell (1991): unlike Parkinson and Garman-Klass it stays
    unbiased when the price has a drift."""
    op, hi, lo, cl = (_col(frame, c) for c in ("open", "high", "low", "close"))
    return (
        np.log(hi / cl) * np.log(hi / op) + np.log(lo / cl) * np.log(lo / op)
    )


def yang_zhang(frame: pd.DataFrame, window: int = 1) -> np.ndarray:
    """Yang-Zhang (2000): drift-independent *and* gap-aware.

    The estimator is properly defined over a window, because the ``k`` weight
    that combines its three components is only meaningful for variances
    measured across several days.  ``window=1`` therefore falls back to the
    natural single-day gap-aware estimator, squared overnight gap plus the
    Rogers-Satchell intraday term; adding the open-to-close square on top of
    Rogers-Satchell as well would count the intraday move twice.
    """
    op, cl = _col(frame, "open"), _col(frame, "close")
    prev_close = np.concatenate(([np.nan], cl[:-1]))
    overnight = np.log(op / prev_close)
    open_close = np.log(cl / op)
    rs = rogers_satchell(frame)
    if window <= 1:
        return overnight ** 2 + rs
    k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
    var_o = pd.Series(overnight).rolling(window).var(ddof=1).to_numpy()
    var_c = pd.Series(open_close).rolling(window).var(ddof=1).to_numpy()
    var_rs = pd.Series(rs).rolling(window).mean().to_numpy()
    return var_o + k * var_c + (1.0 - k) * var_rs


_PROXIES = {
    "squared": "squared",
    "squared_return": "squared",
    "close": "squared",
    "parkinson": parkinson,
    "garman_klass": garman_klass,
    "garman-klass": garman_klass,
    "gk": garman_klass,
    "rogers_satchell": rogers_satchell,
    "rs": rogers_satchell,
    "yang_zhang": yang_zhang,
    "yz": yang_zhang,
}


def realized_variance(
    data, method: str = "squared", returns=None, floor: float = 1e-12
) -> np.ndarray:
    """Daily realized variance by ``method``.

    ``data`` is an OHLC frame for the range estimators, or a return series for
    ``"squared"``.  A small floor is applied because QLIKE divides by the
    proxy, and a literal zero (a day where high == low) would blow it up.
    """
    key = str(method).strip().lower()
    if key not in _PROXIES:
        raise ValueError(f"unknown proxy {method!r}; choose from {sorted(_PROXIES)}")
    fn = _PROXIES[key]
    if fn == "squared":
        series = returns if returns is not None else data
        rv = squared_return(series)
    else:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"proxy {method!r} needs an OHLC DataFrame")
        rv = fn(data)
    rv = np.asarray(rv, dtype=float)
    # range estimators can go slightly negative on rounding; treat as ~0
    return np.where(np.isfinite(rv), np.maximum(rv, floor), np.nan)


def annualize(variance, periods_per_year: float = 252.0) -> np.ndarray:
    """Daily variance -> annualized volatility."""
    return np.sqrt(np.asarray(variance, dtype=float) * periods_per_year)
