from __future__ import annotations

import datetime
import time
import numpy as np
from os import listdir
from nicegui import run, ui
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf

# Configure custom session to prevent Yahoo 403 blocking on Render
session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
        " like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
})


async def req_csv():
  try:
    ticker_str = ticker_input.value.strip().upper()
    if not ticker_str:
      ui.notify("Please enter a valid ticker symbol.", type="warning")
      return

    ticker_object = yf.Ticker(ticker_str, session=session)

    # Set current stock price using fast_info or history fallback
    s_nought = None
    try:
      fast = ticker_object.fast_info
      s_nought = getattr(fast, "last_price", None) or getattr(
          fast, "lastPrice", None
      )
    except Exception:
      pass

    if s_nought is None or np.isnan(s_nought):
      try:
        hist = await run.io_bound(ticker_object.history, period="1d")
        if not hist.empty:
          s_nought = hist["Close"].iloc[-1]
      except Exception:
        pass

    if s_nought is None:
      ui.notify(
          f"Could not fetch spot price for '{ticker_str}'.", type="negative"
      )
      return

    # Safe dividend yield lookup
    q = 0.0
    try:
      info = await run.io_bound(lambda: ticker_object.info)
      q = info.get("dividendYield", 0.0) or 0.0
    except Exception:
      q = 0.0

    ui.notify(f"Fetching option chain for {ticker_str}...")
    stats_card.set_visibility(False)

    # Fetch expirations in background thread
    try:
      expirations = await run.io_bound(lambda: ticker_object.options)
    except Exception:
      expirations = None

    if not expirations:
      ui.notify(
          f"No options data available or access blocked for {ticker_str}.",
          type="negative",
      )
      return

    today_date = datetime.date.today()
    num_expirations = []

    for exp in expirations:
      exp_date = pd.to_datetime(exp).date()
      days_to_exp = (exp_date - today_date).days
      if 180 <= days_to_exp <= 365:
        num_expirations.append(exp)

    num_expirations = num_expirations[:5]

    if not num_expirations:
      ui.notify("No expiration dates found between 180-365 DTE.", type="warning")
      return

    # Fetch option chains safely
    def fetch_all_chains():
      chain_frames = []
      for expiry in num_expirations:
        try:
          chain = ticker_object.option_chain(expiry)
          calls, puts = chain.calls.copy(), chain.puts.copy()
          calls["type"], puts["type"] = "Call", "Put"

          full_chain = pd.concat([calls, puts], ignore_index=True)
          full_chain["expiration"] = expiry

          expiry_date = pd.to_datetime(expiry).date()
          days_to_expiry = (expiry_date - pd.to_datetime("today").date()).days
          full_chain["T"] = max(days_to_expiry, 1) / 365.0

          clean_cols = [
              "expiration",
              "T",
              "strike",
              "type",
              "bid",
              "ask",
              "lastPrice",
              "volume",
          ]
          chain_frames.append(full_chain[clean_cols])
          time.sleep(0.5)
        except Exception as e:
          print(f"Error fetching chain for {expiry}: {e}")
          continue
      return chain_frames

    frames = await run.io_bound(fetch_all_chains)

    if not frames:
      ui.notify(
          f"Yahoo Finance blocked or returned empty option chains for"
          f" {ticker_str}.",
          type="negative",
      )
      return

    surface_df = pd.concat(frames, ignore_index=True)
    surface_df["C_market"] = (surface_df["bid"] + surface_df["ask"]) / 2
    surface_df["C_market"] = surface_df["C_market"].fillna(
        surface_df["lastPrice"]
    )
    surface_df.loc[surface_df["C_market"] <= 0, "C_market"] = surface_df[
        "lastPrice"
    ]

    upper_bound = s_nought * 1.20
    lower_bound = s_nought * 0.80

    filtered_df = surface_df[
        (surface_df["strike"] >= lower_bound)
        & (surface_df["strike"] <= upper_bound)
    ].copy()
    filtered_df = filtered_df[filtered_df["C_market"] > 0.01]

    if filtered_df.empty:
      ui.notify("Insufficient options quotes in the 20% moneyness window.")
      return

    strikes_axis, target_ttms, vol_matrix, params = await run.io_bound(
        pipeline, filtered_df, s_nought, ticker_str, q
    )

    X_grid, Y_grid = np.meshgrid(strikes_axis, target_ttms)
    Z_grid = np.array(vol_matrix)

    fig = go.Figure(
        data=[
            go.Surface(
                x=X_grid,
                y=Y_grid * 365,
                z=Z_grid,
                colorscale=[
                    [0, "#1b2a4a"],
                    [0.5, "#3d6fd6"],
                    [1, "#f2f3f5"],
                ],
                showscale=True,
                colorbar=dict(
                    title="IV",
                    tickfont=dict(color=MUTED),
                    title_font=dict(color=MUTED),
                ),
            )
        ]
    )

    fig.update_layout(
        **dark_layout(f"{ticker_str}  ·  Implied Volatility Surface")
    )
    fig.update_layout(autosize=True, height=700)

    # Standard NiceGUI Plotly element update
    if hasattr(plotly_display, "figure"):
      plotly_display.figure = fig
      plotly_display.update()
    elif hasattr(plotly_display, "update_figure"):
      plotly_display.update_figure(fig)

    kappa, theta, vol_of_vol, rho, v0 = params
    stats_kappa.set_text(f"{kappa:.4f}")
    stats_theta.set_text(f"{theta:.4f}")
    stats_volvol.set_text(f"{vol_of_vol:.4f}")
    stats_rho.set_text(f"{rho:.4f}")
    stats_v0.set_text(f"{v0:.4f}")
    stats_spot.set_text(f"${s_nought:,.2f}")
    stats_card.set_visibility(True)

  except Exception as main_err:
    print(f"Unhandled error in req_csv: {main_err}")
    ui.notify(f"Error processing request: {main_err}", type="negative")
