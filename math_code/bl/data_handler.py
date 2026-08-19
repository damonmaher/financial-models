from __future__ import annotations

import yfinance as yf


def fetch_market_caps(tickers: list[str]) -> dict[str, float | None]:
  """Best-effort market cap lookup per ticker using robust fast_info attribute

  and key resolution.
  """
  caps: dict[str, float | None] = {}
  for t in tickers:
    cap = None
    try:
      ticker = yf.Ticker(t)
      fast = ticker.fast_info

      # Try attribute access
      cap = getattr(fast, "market_cap", None) or getattr(
          fast, "marketCap", None
      )

      # Try dictionary access
      if cap is None and hasattr(fast, "get"):
        cap = fast.get("market_cap") or fast.get("marketCap")

      # Fallback to info lookup
      if cap is None or cap <= 0:
        info = ticker.info
        cap = info.get("marketCap")
    except Exception as e:
      print(f"Error fetching market cap for {t}: {e}")
      cap = None

    caps[t] = cap
  return caps
