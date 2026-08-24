"""
data_handler.py

Handles all market-data plumbing for the Black-Litterman optimizer:
  - pulling historical prices from yfinance
  - building the (annualized) covariance matrix
  - turning user-entered market caps into market-implied weights (w_mkt)

Market caps are entered manually in the UI (yfinance's get_info()/quoteSummary
are unreliable in this hosting environment), so this module just normalizes
whatever {ticker: market_cap} dict it's handed - no network call involved.

Nothing in this module knows about NiceGUI or the optimization math -
it only produces plain numpy/pandas objects that black_litterman.py and
optimizer.py consume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

class DataFetchError(Exception):
    """Raised when we can't build a usable dataset for the requested tickers."""


def fetch_price_history(tickers: list[str], period: str = "3y", interval: str = "1wk") -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.

    Returns a DataFrame indexed by date, one column per ticker, with any
    rows contsaining NaNs dropped (keeps the covariance matrix well-defined
    across a common date range).
    """
    if not tickers:
        raise DataFetchError("No tickers were provided.")

    raw = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

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

    prices = prices.dropna(how="all")

    # Per-ticker valid (non-NaN) observation counts, captured before the
    # any-NaN drop below. If the resulting overlap is too short, whichever
    # ticker(s) have noticeably fewer valid points than the rest are the
    # ones truncating the common date range.
    valid_counts = prices.notna().sum()

    prices = prices.dropna(axis=0, how="any")

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise DataFetchError(f"No price data returned for: {', '.join(missing)}")

    if len(prices) < 10:
        max_valid = int(valid_counts.max()) if len(valid_counts) else 0
        culprits = [
            f"{t} ({int(valid_counts.get(t, 0))} bars)"
            for t in tickers
            if valid_counts.get(t, 0) < max_valid
        ]
        culprit_note = (
            f" Likely culprits (shortest/misaligned history): {', '.join(culprits)}."
            if culprits
            else ""
        )
        raise DataFetchError(
            "Not enough overlapping price history across the selected tickers."
            f"{culprit_note}"
        )

    return prices[tickers]


def compute_covariance(
    price_data: pd.DataFrame,
    periods_per_year: int = 52,
    horizon_years: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute log returns and the sample covariance matrix, scaled to the
    chosen forecast horizon.

    periods_per_year is the number of bars per year implied by the chosen
    interval (e.g. 252 for daily, 52 for weekly, 12 for monthly bars) and
    describes the *sampling* frequency used to estimate the per-bar
    covariance. horizon_years is how far forward the resulting Sigma (and
    everything downstream: Pi, mu_BL, expected return/vol) should represent
    - e.g. 1/12 for a 1-month-ahead view, 1.0 for a 1-year-ahead view.

    Sigma_horizon = Sigma_per_bar * (periods_per_year * horizon_years)

    This is the standard i.i.d. variance-scaling assumption (variance grows
    linearly with the number of periods), just applied to a partial-year
    horizon instead of always a full year.
    """
    log_returns = np.log(price_data / price_data.shift(1)).dropna()
    if log_returns.empty:
        raise DataFetchError("Could not compute returns from the downloaded price history.")

    scale = periods_per_year * horizon_years
    cov_matrix = log_returns.cov() * scale
    return cov_matrix, log_returns


def compute_market_weights(caps: dict[str, float | None]) -> tuple[np.ndarray, list[str], bool]:
    """
    Convert a {ticker: market_cap} dict (user-entered, any consistent unit)
    into a market-cap-weighted vector that sums to 1, in the same order as
    caps.keys(). This is a straight normalization: w_i = cap_i / sum(caps).

    Returns (weights, tickers_in_order, used_fallback) where used_fallback
    is True if one or more market caps were missing/zero/negative and we
    fell back to equal weighting across the whole universe.
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