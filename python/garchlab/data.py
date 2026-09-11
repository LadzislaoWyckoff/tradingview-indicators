"""Price loading.

Three sources, tried in this order when you do not name one:

``csv``       a local file -- always works, and the only thing that works
              behind a locked-down network.
``yfinance``  needs the optional ``yfinance`` package.
``stooq``     a plain CSV endpoint, no key, no extra dependency.

Everything returns the same shape: a DataFrame indexed by date with whatever
of ``open/high/low/close/volume`` the source provides, sorted ascending and
free of duplicate dates.
"""

from __future__ import annotations

import io
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["load_csv", "download", "load_prices", "to_returns", "SP500_SYMBOLS"]

#: The index itself, under the tickers each source expects.
SP500_SYMBOLS = {
    "yfinance": {"index": "^GSPC", "etf": "SPY", "vix": "^VIX"},
    "stooq": {"index": "^spx", "etf": "spy.us", "vix": "^vix"},
}

_OHLC = ("open", "high", "low", "close", "volume")


def _normalize(frame: pd.DataFrame, date_col: str | None = None) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if date_col is None:
        for candidate in ("date", "datetime", "time", "timestamp"):
            if candidate in frame.columns:
                date_col = candidate
                break
    if date_col is not None and date_col.lower() in frame.columns:
        frame.index = pd.to_datetime(frame[date_col.lower()])
        frame = frame.drop(columns=[date_col.lower()])
    else:
        frame.index = pd.to_datetime(frame.index)
    if "adj close" in frame.columns and "close" in frame.columns:
        # Use the adjusted series: unadjusted closes put fake jumps at every
        # dividend and split, and a volatility model reads those as real moves.
        # Scale the whole bar by the same factor rather than replacing the
        # close alone -- otherwise the close can land outside its own high/low
        # and every range-based realized-variance proxy goes wrong.
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = pd.to_numeric(frame["adj close"], errors="coerce") / pd.to_numeric(
                frame["close"], errors="coerce"
            )
        factor = factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        for column in ("open", "high", "low", "close"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce") * factor
    keep = [c for c in _OHLC if c in frame.columns]
    if "close" not in keep:
        raise ValueError(f"no close column found; got {list(frame.columns)}")
    frame = frame[keep].apply(pd.to_numeric, errors="coerce")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame.dropna(subset=["close"])


def load_csv(path, date_col: str | None = None) -> pd.DataFrame:
    """Load an OHLC (or close-only) CSV exported from anywhere."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return _normalize(pd.read_csv(path), date_col)


def _download_stooq(symbol: str, timeout: float = 30.0) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={urllib.parse.quote(symbol)}&i=d"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read().decode("utf-8", errors="replace")
    if "Date" not in payload.split("\n", 1)[0]:
        raise RuntimeError(f"stooq returned no data for {symbol!r}: {payload[:200]!r}")
    return _normalize(pd.read_csv(io.StringIO(payload)))


def _download_yfinance(symbol: str, start=None, end=None) -> pd.DataFrame:
    try:
        import yfinance
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("yfinance is not installed (pip install yfinance)") from exc
    frame = yfinance.download(
        symbol, start=start, end=end, auto_adjust=True, progress=False
    )
    if frame is None or frame.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol!r}")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    return _normalize(frame)


def download(symbol: str = "^GSPC", source: str = "auto", start=None, end=None) -> pd.DataFrame:
    """Fetch daily OHLC for ``symbol``.

    ``source="auto"`` tries yfinance first and falls back to stooq.  Both are
    network calls; behind a restrictive proxy neither will work and you should
    export a CSV once and use :func:`load_csv`.
    """
    errors = []
    order = {"auto": ("yfinance", "stooq")}.get(source, (source,))
    for src in order:
        try:
            if src == "yfinance":
                return _download_yfinance(symbol, start, end)
            if src == "stooq":
                mapped = symbol
                if source == "auto" and symbol in ("^GSPC",):
                    mapped = "^spx"
                return _download_stooq(mapped)
            raise ValueError(f"unknown source {src!r}")
        except Exception as exc:  # noqa: BLE001 - report every source's failure
            errors.append(f"{src}: {exc}")
    raise RuntimeError(
        "could not download prices.\n  " + "\n  ".join(errors)
        + "\nIf this machine has no market-data access, export a CSV and pass "
          "--csv instead."
    )


def load_prices(
    csv: str | None = None, symbol: str = "^GSPC", source: str = "auto",
    start=None, end=None, date_col: str | None = None,
) -> pd.DataFrame:
    """CSV if given, otherwise download.  Optionally trimmed to [start, end]."""
    frame = load_csv(csv, date_col) if csv else download(symbol, source, start, end)
    if start is not None:
        frame = frame.loc[frame.index >= pd.Timestamp(start)]
    if end is not None:
        frame = frame.loc[frame.index <= pd.Timestamp(end)]
    if frame.empty:
        raise ValueError("no rows left after applying the date filter")
    return frame


def to_returns(prices, kind: str = "log", dropna: bool = True) -> pd.Series:
    """Close-to-close returns.

    Log returns are the right input for a GARCH model: they add across time, so
    the h-day variance is the sum of the daily variances, which is exactly the
    aggregation the forecast performs.
    """
    close = prices["close"] if isinstance(prices, pd.DataFrame) else pd.Series(prices)
    close = close.astype(float)
    if kind == "log":
        out = np.log(close).diff()
    elif kind in ("simple", "pct"):
        out = close.pct_change()
    else:
        raise ValueError("kind must be 'log' or 'simple'")
    out.name = "return"
    return out.dropna() if dropna else out
