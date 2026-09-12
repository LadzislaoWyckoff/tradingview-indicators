"""Price loading.

Four sources, tried in this order when you do not name one:

``csv``       a local file -- always works, and the only thing that works
              behind a locked-down network.
``yfinance``  needs the optional ``yfinance`` package; free and unmetered, so
              it leads the automatic order.
``fmp``       Financial Modeling Prep; needs ``FMP_API_KEY`` in the
              environment.  The same daily bars as Yahoo (see below), so use it
              as a second opinion or when Yahoo is unavailable.
``stooq``     a plain CSV endpoint, no key -- but as of 2026 it sits behind a
              JavaScript proof-of-work challenge and returns HTML rather than
              CSV, so treat it as a long shot rather than a fallback.

A word on the open, because it silently decides whether half this library
works.  For ^GSPC the open is *synthesised from the previous close* over most
of the older history: measured as the share of days where ``open == previous
close``, it runs 76.6% in the 1990s, 96.2% in 2000-2005, 15.1% in 2006-2009,
6.2% in the 2010s and 0.1% from 2020.  Every proxy that reads the open
(Garman-Klass, Rogers-Satchell, Yang-Zhang) is therefore meaningless before
about 2006, and nothing warns you -- use ``proxy="squared"`` on long
histories, or restrict the range proxies to recent data.

This is a property of the index's recorded history, not of a vendor: FMP and
Yahoo return those same percentages decade for decade, and their closes agree
to within 0.01 index points across 9,241 bars.  Do not switch vendors hoping
to fix it.

Everything returns the same shape: a DataFrame indexed by date with whatever
of ``open/high/low/close/volume`` the source provides, sorted ascending and
free of duplicate dates.
"""

from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["load_csv", "download", "load_prices", "to_returns", "SP500_SYMBOLS"]

#: The index itself, under the tickers each source expects.
SP500_SYMBOLS = {
    "fmp": {"index": "^GSPC", "etf": "SPY", "vix": "^VIX"},
    "yfinance": {"index": "^GSPC", "etf": "SPY", "vix": "^VIX"},
    "stooq": {"index": "^spx", "etf": "spy.us", "vix": "^vix"},
}

_FMP_BASE = "https://financialmodelingprep.com/stable"
#: the endpoint truncates any window to this many bars, so long histories page
_FMP_MAX_ROWS = 5000
_FMP_MAX_PAGES = 12

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


def _fmp_page(symbol: str, start, end, key: str, timeout: float) -> list:
    """One FMP end-of-day window.  Returns [] when the window holds nothing."""
    query = urllib.parse.urlencode({
        "symbol": symbol,
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
        "apikey": key,
    })
    url = f"{_FMP_BASE}/historical-price-eod/full?{query}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        hint = " (rate limited)" if exc.code == 429 else ""
        raise RuntimeError(f"FMP returned HTTP {exc.code}{hint}") from exc
    if isinstance(payload, dict):
        # FMP answers 200 with {"Error Message": "Limit Reach ..."} on a used-up key
        message = payload.get("Error Message") or payload.get("error") or payload
        raise RuntimeError(f"FMP error: {str(message)[:200]}")
    return payload if isinstance(payload, list) else []


def _download_fmp(symbol: str, start=None, end=None, timeout: float = 60.0) -> pd.DataFrame:
    """Daily OHLCV from Financial Modeling Prep.

    Reads the key from the ``FMP_API_KEY`` environment variable.  Pages
    backwards because a single request is capped at 5,000 bars, which is about
    twenty years of dailies.

    On daily ^GSPC this returns the same series as yfinance -- closes match to
    0.01 index points and the synthetic-open share is identical decade by
    decade -- so it buys redundancy rather than better data.  Its real edge is
    intraday history, which Yahoo will not give you in bulk and which is what
    you would need to move past daily GARCH to a realized-volatility model.

    Prices are **not** dividend-adjusted.  That is exactly right for an index
    such as ^GSPC, whose level contains no dividends or splits to adjust away.
    It is wrong for an ETF: SPY's quarterly dividend shows up as a fabricated
    gap down, and because those fakes are always negative they land squarely on
    the leverage term of a GJR or EGARCH fit.  For ETFs use the index instead,
    or yfinance, which adjusts.
    """
    key = os.environ.get("FMP_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "FMP_API_KEY is not set; export it or pass --source yfinance"
        )
    floor = pd.Timestamp(start) if start is not None else pd.Timestamp("1900-01-01")
    cursor = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()

    pages = []
    for _ in range(_FMP_MAX_PAGES):
        rows = _fmp_page(symbol, floor, cursor, key, timeout)
        if not rows:
            break
        page = pd.DataFrame(rows)
        if "date" not in page.columns:
            raise RuntimeError(f"FMP returned no date column for {symbol!r}")
        page["date"] = pd.to_datetime(page["date"])
        pages.append(page)
        oldest = page["date"].min()
        # a short page means the window was covered in full, so we are done
        if len(rows) < _FMP_MAX_ROWS or oldest <= floor:
            break
        cursor = oldest - pd.Timedelta(days=1)
    if not pages:
        raise RuntimeError(f"FMP returned no rows for {symbol!r}")
    merged = pd.concat(pages, ignore_index=True).drop_duplicates(subset="date")
    return _normalize(merged)


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

    ``source="auto"`` tries yfinance, then FMP, then stooq, keeping whichever
    answers first.  yfinance leads because it is free and unmetered, while an
    FMP key is typically shared with other tooling and rate limited -- ask for
    ``source="fmp"`` explicitly when you want it.  All three are network calls;
    behind a restrictive proxy none will work and you should export a CSV once
    and use :func:`load_csv`.
    """
    errors = []
    order = {"auto": ("yfinance", "fmp", "stooq")}.get(source, (source,))
    for src in order:
        try:
            if src == "fmp":
                return _download_fmp(symbol, start, end)
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
