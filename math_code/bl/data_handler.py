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


class DataFetchError(Exception):
    """Raised when we can't build a usable dataset for the requested tickers."""


## ---------- yfinance hardening -----------
## yfinance (0.2.60+) already impersonates Chrome via curl_cffi internally
## by default - it builds its own `requests.Session(impersonate="chrome")`
## whenever you DON'T pass a session. Passing an external curl_cffi session
## in ourselves conflicts with yfinance's own cookie/crumb handshake and
## causes info calls to silently return None instead of raising, which is
## worse than doing nothing (this was the cause of the market-cap gaps).
## So: don't touch sessions at all, just retry with backoff on top of
## yfinance's built-in impersonation, since quoteSummary is still the most
## rate-limit-sensitive endpoint even when impersonation works correctly.

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
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.
    Returns a DataFrame indexed by date, one column per ticker, with any
    rows containing NaNs dropped (keeps the covariance matrix well-defined
    across a common date range).
    """
    if not tickers:
        raise DataFetchError("No tickers were provided.")

    try:
        raw = yf_retry(
            yf.download,
            tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="column",
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


def fetch_market_caps(tickers: list[str]) -> dict[str, float | None]:
    """
    Best-effort market cap lookup per ticker. Returns None for any ticker
    where we can't determine a market cap, so the caller can decide how to
    handle the gap.

    Uses fast_info instead of get_info()/.info: get_info() hits Yahoo's
    quoteSummary endpoint, which is blocked outright for some cloud host
    IPs (retries return None instead of a 429/timeout - an IP-level block,
    not a rate limit, so retrying alone doesn't help). fast_info derives
    its values from the chart/quote endpoints instead, which stay reachable
    even when quoteSummary doesn't. If fast_info doesn't expose marketCap
    directly for a given ticker, we fall back to shares outstanding times
    last price, both also sourced from fast_info.
    """
    caps: dict[str, float | None] = {}
    for i, t in enumerate(tickers):
        cap = None
        try:
            fi = yf_retry(lambda tk=t: yf.Ticker(tk).fast_info)
            try:
                cap = fi["marketCap"]
            except (KeyError, TypeError):
                cap = None
            if not cap:
                shares = fi.get("shares") if hasattr(fi, "get") else None
                last_price = fi.get("lastPrice") if hasattr(fi, "get") else None
                if shares and last_price:
                    cap = float(shares) * float(last_price)
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
