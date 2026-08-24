#!/usr/bin/env python3
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime
import itertools

import set_params
import learn
import hmm

from pathlib import Path
from nicegui import ui

## ----------- Theme constants -----------

BG_PAGE = '#08090b'          # near-black page background
BG_PANEL = '#0e1013'         # card background
BORDER = 'rgba(120, 210, 160, 0.22)'   # soft green panel border
BORDER_HOVER = 'rgba(120, 210, 160, 0.45)'
GREEN = '#7fd8a4'
WHITE = '#f2f3f5'
MUTED = '#8a8f98'
BLUE = '#6fa8ff'
ORANGE = '#ff9d4d'
RED = '#ff6b6b'

## ----------- Fetch Data -----------

def make_empty_figure():
    fig = go.Figure()
    fig.update_layout(**dark_layout(title="Enter a ticker to begin"))
    return fig

def dark_layout(title):
    return dict(
        title=dict(text=title, font=dict(family='JetBrains Mono, monospace', size=16, color=WHITE)),
        paper_bgcolor=BG_PANEL,
        plot_bgcolor=BG_PANEL,
        font=dict(family='JetBrains Mono, monospace', color=MUTED, size=12),
        xaxis=dict(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.08)',
                   linecolor='rgba(255,255,255,0.12)', title_font=dict(color=MUTED)),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.08)',
                   linecolor='rgba(255,255,255,0.12)', title_font=dict(color=MUTED)),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=MUTED)),
        margin=dict(l=50, r=30, t=50, b=40),
    )

global_fig = make_empty_figure()


def req_csv():
    ticker_symbol = ticker_input.value

    if not ticker_symbol:
        ui.notify('Please enter a ticker symbol.', type='warning')
        return

    ui.notify(f'Fetching data for {ticker_symbol}...')
    stats_card.set_visibility(False)

    # Fetching 5 years of OHLC ticker info from yfinance
    df = yf.download(ticker_symbol, period='5y')

    if df.empty:
        ui.notify(f'No data found for {ticker_symbol}', type='negative')
        return

    global_fig.data = []  # clear everything

    # Extract Open, High, Low, Close handling both MultiIndex and standard columns
    required_cols = ['Open', 'High', 'Low', 'Close']

    if isinstance(df.columns, pd.MultiIndex):
        # Select the specific ticker level from MultiIndex
        ohlc_df = df.xs(ticker_symbol, level=1, axis=1)[required_cols]
    else:
        ohlc_df = df[required_cols]

    # Drop any incomplete rows and convert to float64
    ohlc_df = ohlc_df.dropna().astype(float)

    # Extract close_prices directly from cleaned OHLC data for guaranteed index alignment
    close_prices = ohlc_df['Close']

    # ------ Instantiate Parameters -------

    # 1. Run Garman-Klass Heston calibration on full OHLC DataFrame
    kappa, theta, sigma = set_params.calc_params(ohlc_df)

    # 2. Pass calibrated parameters and aligned close prices to HMM matrix builder
    A, B, pi = learn.matrices(kappa, theta, sigma, close_prices)

    #Observations
    daily_log_returns = np.log(close_prices / close_prices.shift(1))
    rolling_drift_10d = daily_log_returns.rolling(window=10).mean()
    drift = rolling_drift_10d.dropna().values

    print(f"Mean Reversion: {kappa}, Long-Term Variance: {theta}, Vol-of-Vol: {sigma}")
    print(f"Transition: {A}, Emission: {B}, Pi: {pi}")

    #Discretize drifts by sorting into buckets
    drift_flat = drift.flatten()

    b_30 = 0.30 / 252
    b_15 = 0.15 / 252
    b_05 = 0.05 / 252

    obs = np.zeros(len(drift_flat), dtype=int)
    obs[drift_flat > b_30] = 0
    obs[(drift_flat > b_15) & (drift_flat <= b_30)] = 1
    obs[(drift_flat > b_05) & (drift_flat <= b_15)] = 2
    obs[(drift_flat >= -b_05) & (drift_flat <= b_05)] = 3
    obs[(drift_flat >= -b_15) & (drift_flat < -b_05)] = 4
    obs[(drift_flat >= -b_30) & (drift_flat < -b_15)] = 5
    obs[drift_flat < -b_30] = 6

    #Baum-Welch learning
    A_hmm, B_hmm, pi_hmm = hmm.baum_welch_vec(obs, A, B, pi)

    print(f"Transition: {A_hmm}, Emission: {B_hmm}, Pi: {pi_hmm}")

    # Isolate data in the past week
    past_week = close_prices.tail(5)
    last_date = past_week.index[-1]
    last_price = float(np.ravel(past_week.values)[-1])

    # Plot the historical data first
    global_fig.add_trace(go.Scatter(
        x=past_week.index.strftime('%Y-%m-%d').tolist(),
        y=np.ravel(past_week.values),
        mode='lines+markers',
        name='Past Week',
        line=dict(color=WHITE, width=3),
        marker=dict(size=6, color=WHITE, line=dict(width=1, color=BG_PANEL))
    ))

    # ----- 2. PREPARE THE FUTURE DATES -----
    # Generate the next 5 business days
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=5)

    # Prepend the last historical date to the future dates so the lines connect!
    raw_plot_dates = [last_date] + list(future_dates)

    plot_dates_str = [d.strftime('%Y-%m-%d') for d in raw_plot_dates]

    # ----- 3. CALCULATE ALL POSSIBLE 5-DAY PATH PROBABILITIES -----
    alpha, c = hmm.forward_vec(obs, A_hmm, B_hmm, pi_hmm)
    today_probs = alpha[-1] / (np.sum(alpha[-1]) + 1e-300)
    tomorrow_probs = today_probs @ A_hmm

    all_paths = list(itertools.product(range(7), repeat=5))
    path_probabilities = []

    for path in all_paths:
        prob = tomorrow_probs[path[0]]
        for i in range(4):
            prob *= A_hmm[path[i], path[i+1]]
        if prob > 1e-8:
            path_probabilities.append((prob, path))

    # Sort paths from Most Likely to Least Likely
    path_probabilities.sort(key=lambda x: x[0], reverse=True)
    top_n = min(50, len(path_probabilities))
    top_paths = path_probabilities[:top_n]

    # ----- 4. CALCULATE EXPECTED PRICES & PLOT GRADIENT -----
    bucket_means_annual = np.array([0.45, 0.225, 0.10, 0.0, -0.10, -0.225, -0.45])
    daily_drifts = bucket_means_annual / 252

    # Green (likely / near) -> red (unlikely / far), gradient encodes rank likelihood
    start_rgb = (127, 216, 164)   # GREEN
    end_rgb = (255, 107, 107)     # RED

    for rank, (prob, path) in enumerate(top_paths):
        # Start the price array with the last historical price to close the gap
        prices = [last_price]
        curr_price = last_price

        for state in path:
            curr_price *= np.exp(daily_drifts[state])
            prices.append(curr_price)

        ratio = rank / max(1, (top_n - 1))
        r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
        g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
        b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)

        opacity = max(0.12, 1.0 - (ratio * 0.85))
        color = f'rgba({r}, {g}, {b}, {opacity:.2f})'

        is_top = rank == 0

        # Add the line to the chart
        global_fig.add_trace(go.Scatter(
            x=plot_dates_str,
            y=prices,
            mode='lines',
            line=dict(color=color, width=4 if is_top else 1.4),
            showlegend=(rank == 0 or rank == top_n - 1),
            name='Most Likely Path' if rank == 0 else ('Least Likely (Top 50)' if rank == top_n - 1 else None),
            hoverinfo='skip'
        ))

    global_fig.update_layout(**dark_layout(
        title=f"{ticker_symbol.upper()}  ·  Past Week vs. Regime-Weighted Prediction Paths"
    ))
    global_fig.update_layout(xaxis_title="Date", yaxis_title="Price")

    plotly_display.update()

    # ----- Update the parameter / stats panel -----
    stats_kappa.set_text(f'{kappa:.5f}')
    stats_theta.set_text(f'{theta:.5f}')
    stats_sigma.set_text(f'{sigma:.5f}')
    stats_regime.set_text(REGIME_LABELS[int(np.argmax(pi_hmm))])
    stats_card.set_visibility(True)


REGIME_LABELS = [
    'Severely Bullish', 'Bullish', 'Slightly Bullish', 'Stagnant',
    'Slightly Bearish', 'Bearish', 'Severely Bearish'
]

# ----- Page setup / global styling -----

ui.dark_mode().enable()

ui.add_head_html('''
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=EB+Garamond:ital,wght@1,400;1,500&family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
<style>
    body {
        background: radial-gradient(circle at 20% 0%, #12151a 0%, #08090b 55%, #050506 100%) !important;
        font-family: 'Inter', sans-serif;
    }
    .q-page { background: transparent !important; }

    .panel-card {
        background: #0e1013;
        border: 1px solid rgba(120, 210, 160, 0.22);
        border-radius: 14px;
        box-shadow: 0 0 0 1px rgba(0,0,0,0.4), 0 12px 28px rgba(0,0,0,0.35);
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .panel-card:hover {
        border-color: rgba(120, 210, 160, 0.45);
        box-shadow: 0 0 24px rgba(120, 210, 160, 0.08), 0 12px 28px rgba(0,0,0,0.4);
    }

    .brand-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        letter-spacing: 0.02em;
        color: #f2f3f5;
        text-transform: uppercase;
    }
    .brand-subtitle {
        font-family: 'EB Garamond', serif;
        font-style: italic;
        color: #7fd8a4;
        font-size: 1.3rem;
    }
    .mono-label {
        font-family: 'JetBrains Mono', monospace;
        color: #8a8f98;
        font-size: 0.78rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .mono-value {
        font-family: 'JetBrains Mono', monospace;
        color: #f2f3f5;
        font-size: 1.3rem;
    }

    .q-field__control {
        background: #0b0c0f !important;
        border-radius: 8px !important;
    }
    .q-field--outlined .q-field__control:before {
        border-color: rgba(120, 210, 160, 0.3) !important;
    }
    .q-field--outlined.q-field--focused .q-field__control:after {
        border-color: #7fd8a4 !important;
    }
    .q-field__label { color: #8a8f98 !important; font-family: 'JetBrains Mono', monospace; }
    .q-field__native { color: #f2f3f5 !important; font-family: 'JetBrains Mono', monospace; }

    .run-btn {
        background: #7fd8a4 !important;
        color: #08090b !important;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        letter-spacing: 0.05em;
        border-radius: 8px !important;
    }
    .run-btn:hover { filter: brightness(1.08); }
</style>
''')

## ----- UI Layout -----

with ui.column().classes('w-full items-center').style('padding: 2.5rem 1.5rem;'):
    with ui.column().classes('items-center gap-1').style('max-width: 900px; margin-bottom: 2rem;'):
        ui.label('HIDDEN MARKOV REGIME PREDICTION').classes('brand-title text-3xl text-center')

    with ui.row().classes('items-center gap-4 panel-card').style('padding: 1.1rem 1.5rem; width: 100%; max-width: 900px;'):
        ticker_input = ui.input(
            label='Ticker',
            value='NVDA',
            placeholder='e.g. NVDA'
        ).props('outlined dense').classes('w-40')
        ui.button('RUN MODEL', on_click=req_csv).classes('run-btn').props('unelevated')
        ui.label('Heston vol-of-vol · Baum-Welch calibrated 7-state HMM').classes('mono-label').style('margin-left: auto;')

    with ui.column().classes('panel-card').style('width: 100%; max-width: 900px; margin-top: 1.5rem; padding: 0.75rem;'):
        plotly_display = ui.plotly(global_fig).classes('w-full').style('height: 560px;')

    with ui.row().classes('gap-4 flex-wrap').style('width: 100%; max-width: 900px; margin-top: 1.5rem;') as stats_card:
        with ui.column().classes('panel-card items-start').style('padding: 1rem 1.5rem; min-width: 180px; flex: 1;'):
            ui.label('κ  MEAN REVERSION').classes('mono-label')
            stats_kappa = ui.label('—').classes('mono-value')
        with ui.column().classes('panel-card items-start').style('padding: 1rem 1.5rem; min-width: 180px; flex: 1;'):
            ui.label('θ  LONG-TERM VARIANCE').classes('mono-label')
            stats_theta = ui.label('—').classes('mono-value')
        with ui.column().classes('panel-card items-start').style('padding: 1rem 1.5rem; min-width: 180px; flex: 1;'):
            ui.label('σ  VOL-OF-VOL').classes('mono-label')
            stats_sigma = ui.label('—').classes('mono-value')
        with ui.column().classes('panel-card items-start').style('padding: 1rem 1.5rem; min-width: 180px; flex: 1;'):
            ui.label('CURRENT REGIME').classes('mono-label')
            stats_regime = ui.label('—').classes('mono-value').style('color: #7fd8a4;')

    stats_card.set_visibility(False)

ui.run(native=True, title="Regime Prediction", window_size=(960, 900))