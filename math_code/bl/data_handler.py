"""
data_handler.py
Handles all market-data plumbing for the Black-Litterman optimizer:
  - pulling historical prices from yfinance
  - building the (annualized) covariance matrix
  - pulling market caps and turning them into market-implied weights (w_mkt)
Nothing in this module knows about NiceGUI or the optimization math -
it only produces plain numpy/pandas objects that black_litterman.py and
optimizer.py consume.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yfinance as yf

from curl_cffi import requests as cffi_requests


class DataFetchError(Exception):
    """Raised when we can't build a usable dataset for the requested tickers."""


## ---------- yfinance hardening -----------
# Yahoo's anti-bot layer is far stricter on the quoteSummary endpoint
# (which backs .info / .get_info(), used below for market caps) than on
# the chart/download endpoint used for price history. Cloud hosts like
# Render get flagged on quoteSummary even when the chart endpoint works
# fine, so every call below goes through a browser-impersonating session
# with retry/backoff.

def make_yf_session():
    """A fresh curl_cffi session that impersonates Chrome's TLS fingerprint."""
    return cffi_requests.Session(impersonate="chrome")


def yf_retry(func, *args, retries=4, base_delay=1.5, **kwargs):
    """Call func(*args, **kwargs), retrying with exponential backoff on any
    exception (429s, empty JSON bodies, transient connection errors, etc).
    Re-raises the last exception if every attempt fails."""
    last_exc = None
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    raise last_exc


def fetch_price_history(
    tickers: list[str],
    period: str = "3y",
    interval: str = "1wk",
    session=None,
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.
    Returns a DataFrame indexed by date, one column per ticker, with any
    rows containing NaNs dropped (keeps the covariance matrix well-defined
    across a common date range).
    """
    if not tickers:
        raise DataFetchError("No tickers were provided.")

    if session is None:
        session = make_yf_session()

    try:
        raw = yf_retry(
            yf.download,
            tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="column",
            session=session,
        )
    except Exception as e:
        raise DataFetchError(
            f"yfinance request failed for {tickers}: {type(e).__name__}: {e}"
        ) from e

    if raw is None or raw.empty:
        raise DataFetchError(
            f"yfinance returned no data for {tickers}. Check the ticker symbols."
        )
    # yfinance returns a MultiIndex column frame for >1 ticker, a flat frame for 1
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise DataFetchError("Downloaded data did not contain a 'Close' column.")
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers
    prices = prices.dropna(how="all").dropna(axis=0, how="any")
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise DataFetchError(f"No price data returned for: {', '.join(missing)}")
    if len(prices) < 10:
        raise DataFetchError(
            "Not enough overlapping price history across the selected tickers."
        )
    return prices[tickers]


def compute_covariance(price_data: pd.DataFrame, periods_per_year: int = 52) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute log returns and the annualized sample covariance matrix.
    periods_per_year defaults to 52 for weekly bars ('1wk' interval above).
    """
    log_returns = np.log(price_data / price_data.shift(1)).dropna()
    if log_returns.empty:
        raise DataFetchError("Could not compute returns from the downloaded price history.")
    cov_matrix = log_returns.cov() * periods_per_year
    return cov_matrix, log_returns


def fetch_market_caps(tickers: list[str], session=None) -> dict[str, float | None]:
    """
    Best-effort market cap lookup per ticker. Returns None for any ticker
    where yfinance doesn't expose a market cap (e.g. some ETFs/indices), or
    where every retry attempt fails, so the caller can decide how to handle
    the gap.

    This hits Yahoo's quoteSummary endpoint per-ticker, which is the most
    rate-limit-sensitive call yfinance makes - each lookup goes through
    yf_retry with backoff, and a small pause is added between tickers so a
    multi-ticker universe doesn't trip the rate limiter on its own.
    """
    if session is None:
        session = make_yf_session()

    caps: dict[str, float | None] = {}
    for i, t in enumerate(tickers):
        cap = None
        try:
            info = yf_retry(yf.Ticker(t, session=session).get_info)
            cap = info.get("marketCap")
        except Exception as e:
            print(f"yfinance error fetching market cap for '{t}': {type(e).__name__}: {e}")
            cap = None
        caps[t] = cap

        # Small pause between tickers to avoid tripping the rate limiter
        # on the shared/cloud IP, on top of the per-call backoff above.
        if i < len(tickers) - 1:
            time.sleep(0.75)

    return caps


def compute_market_weights(caps: dict[str, float | None]) -> tuple[np.ndarray, list[str], bool]:
    """
    Convert a {ticker: market_cap} dict into a market-cap-weighted vector
    that sums to 1, in the same order as caps.keys().
    Returns (weights, tickers_in_order, used_fallback) where used_fallback
    is True if one or more market caps were missing and we fell back to
    equal weighting across the whole universe.
    """
    tickers = list(caps.keys())
    values = [caps[t] for t in tickers]
    if any(v is None or v <= 0 for v in values):
        n = len(tickers)
        weights = np.full(n, 1.0 / n)
        return weights, tickers, True
    values_arr = np.array(values, dtype=float)
    weights = values_arr / values_arr.sum()
    return weights, tickers, False
