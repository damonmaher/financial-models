#!/usr/bin/env python3

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import time
import datetime

import set_params
import bs
import fft

from pathlib import Path
from nicegui import ui

#import set_params

## ---------- Theme constants -----------

BG_PANEL = '#0e1013'
BORDER = 'rgba(120, 210, 160, 0.22)'
WHITE = '#f2f3f5'
MUTED = '#8a8f98'
GREEN = '#7fd8a4'

def dark_scene():
    return dict(
        xaxis=dict(title='Strike Price ($)', backgroundcolor=BG_PANEL, gridcolor='rgba(255,255,255,0.08)',
                   color=MUTED, showbackground=True),
        yaxis=dict(title='Days to Maturity', backgroundcolor=BG_PANEL, gridcolor='rgba(255,255,255,0.08)',
                   color=MUTED, showbackground=True),
        zaxis=dict(title='Implied Volatility (IV)', backgroundcolor=BG_PANEL, gridcolor='rgba(255,255,255,0.08)',
                   color=MUTED, showbackground=True),
    )

def dark_layout(title):
    return dict(
        title=dict(text=title, font=dict(family='JetBrains Mono, monospace', size=16, color=WHITE)),
        paper_bgcolor=BG_PANEL,
        plot_bgcolor=BG_PANEL,
        font=dict(family='JetBrains Mono, monospace', color=MUTED, size=12),
        scene=dark_scene(),
        margin=dict(l=10, r=10, t=50, b=10),
    )

## ---------- Parameters -----------

cmarket_df = 0
s_nought = 0
r_free = 0
var = 0
vol_vol = 0
long_term_var = 0
mean_rev = 0

global_fig = go.Figure()
global_fig.update_layout(**dark_layout("Enter a ticker to begin"))



## -------- Event Listeners --------

def req_csv():
    ticker_object = yf.Ticker(ticker_input.value)

    #Set current stock price
    try:
        s_nought = ticker_object.history(period="1d")['Close'].iloc[-1]

        # fetch dividend yield q
        info = ticker_object.info
        q = info.get('dividendYield', 0.0)
        if q is None:
            q = 0.0

    except Exception:
        print("Incorrect Ticker Entered")
        ui.notify('Incorrect ticker entered.', type='negative')
        return None, None

    ui.notify(f'Fetching option chain for {ticker_input.value}...')
    stats_card.set_visibility(False)

    #Extract expiration dates
    expirations = ticker_object.options
    if not expirations:
        print("No Options Data for this Ticker")
        ui.notify('No options data for this ticker.', type='negative')
        return

    #Limit expirations
    today_date = datetime.date.today()
    num_expirations = []

    for exp in expirations:
        exp_date = pd.to_datetime(exp).date()
        days_to_exp = (exp_date-today_date).days

        if 180 <= days_to_exp <= 365:
            num_expirations.append(exp)


    num_expirations = num_expirations[:5]

    frames = []

    #Loop through all expirations
    for expiry in num_expirations:
        try:
            #Get a specific chain
            chain = ticker_object.option_chain(expiry)

            #Label calls and puts
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls['type'] = 'Call'
            puts['type'] = 'Put'

            #Combine calls and puts for this expiration
            full_chain = pd.concat([calls,puts],ignore_index=True)

            #Add expiration date as a column
            full_chain['expiration'] = expiry

            #Calculate TTE in years
            expiry_date = pd.to_datetime(expiry).date()
            today_date = pd.to_datetime('today').date()

            days_to_expiry = (expiry_date - today_date).days
            full_chain['T'] = max(days_to_expiry, 1) / 365.0

            #Keep only necessary columns
            clean_columns = ['expiration', 'T', 'strike', 'type', 'bid', 'ask', 'lastPrice', 'volume']
            frames.append(full_chain[clean_columns])

            #Pause to prevent rate limits being reached
            time.sleep(1)

        except Exception as e:
            print("Error")
            continue

    #Concatenate all rows into one master Data Frame
    surface_df = pd.concat(frames, ignore_index=True)

    #Calculate C_Market
    surface_df['C_market'] = (surface_df['bid']+surface_df['ask'])/2

    #If bid/ask is missing of zero, use last traded price
    surface_df['C_market'] = surface_df['C_market'].fillna(surface_df['lastPrice'])
    surface_df.loc[surface_df['C_market'] <= 0, 'C_market'] = surface_df['lastPrice']

    #Only want a 20% strike-asset price spread
    upper_bound = s_nought * 1.20
    lower_bound = s_nought * 0.80

    filtered_df = surface_df[
        (surface_df['strike'] >= lower_bound) &
        (surface_df['strike'] <= upper_bound)
    ].copy()

    #Remove quotes with an ask price of 0
    filtered_df = filtered_df[filtered_df['C_market'] > 0.01]
    strikes_axis, target_ttms, vol_matrix, params = pipeline(filtered_df, s_nought, ticker_input.value, q)
    X_grid, Y_grid = np.meshgrid(strikes_axis, target_ttms)
    Z_grid = np.array(vol_matrix)

    fig = go.Figure(data=[go.Surface(
        x=X_grid,
        y=Y_grid * 365, #turns back into days
        z=Z_grid,
        colorscale=[[0, '#1b2a4a'], [0.5, '#3d6fd6'], [1, '#f2f3f5']],
        showscale=True,
        colorbar=dict(title='IV', tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    )])

    fig.update_layout(**dark_layout(f'{ticker_input.value.upper()}  ·  Implied Volatility Surface'))
    fig.update_layout(autosize=True, height=700)

    plotly_display.update_figure(fig)

    #Update the parameter panel
    kappa, theta, vol_of_vol, rho, v0 = params
    stats_kappa.set_text(f'{kappa:.4f}')
    stats_theta.set_text(f'{theta:.4f}')
    stats_volvol.set_text(f'{vol_of_vol:.4f}')
    stats_rho.set_text(f'{rho:.4f}')
    stats_v0.set_text(f'{v0:.4f}')
    stats_spot.set_text(f'${s_nought:,.2f}')
    stats_card.set_visibility(True)



## -------- Master Process ---------

#Generate risk_free interest vals for each contract, append to df
def pipeline(filtered_df, s_nought, ticker, q):
    filtered_df['r'] = set_params.calc_sofr_rates(filtered_df['T'].values)
    filtered_df['vol_atm'] = bs.inv_bs(s_nought, filtered_df['strike'].values, filtered_df['r'].values, filtered_df['T'].values, filtered_df['C_market'].values, q)
    print(filtered_df)

    kappa, theta, vol_of_vol, rho, v0 = set_params.calc_params(ticker)
    print(f"Kappa: {kappa:.4f}, Theta: {theta:.4f}, Vol_of_Vol: {vol_of_vol:.4f}, Rho: {rho:.4f}")

    days = np.arange(180,365,10)
    target_ttms = days / 365.0
    strikes_axis = np.linspace(s_nought * 0.5, s_nought * 1.5, 100)
    vol_matrix = []

    for ttm in target_ttms:
        r = set_params.calc_sofr_rates(ttm)
        if isinstance(r, np.ndarray):
            r=r[0]
        fair_prices, strikes = fft.heston_fft_main(s_nought,ttm, kappa, theta, vol_of_vol, rho, v0, r, q, is_call=True)
        standardized_prices = np.interp(strikes_axis, strikes, fair_prices) #interpolate onto 100-point strikes to get more uniform values
        atm_vols = bs.inv_bs(s_nought, strikes_axis, r, ttm, standardized_prices, q)
        vol_matrix.append(atm_vols)

    return strikes_axis, target_ttms, vol_matrix, (kappa, theta, vol_of_vol, rho, v0)




#Build parameters
#Run fft



## ----------- Page setup / global styling -----------

ui.dark_mode().enable()

ui.add_head_html('''
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
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

## ----------- UI Layout -----------

with ui.column().classes('w-full items-center').style('padding: 2.5rem 1.5rem;'):
    with ui.column().classes('items-center gap-1').style('max-width: 900px; margin-bottom: 2rem;'):
        ui.label('HESTON-BASED PNL PREDICTION FOR OPTIONS').classes('brand-title text-3xl text-center')

    with ui.row().classes('items-center gap-4 panel-card').style('padding: 1.1rem 1.5rem; width: 100%; max-width: 900px;'):
        ticker_input = ui.input(
            label='Ticker Symbol',
            value='NVDA',
            placeholder='Enter Ticker Symbol'
        ).props('outlined dense').classes('w-40')
        ui.button('REQUEST CSV', on_click=req_csv).classes('run-btn').props('unelevated')
        ui.label('180–365 DTE chain · FFT-priced Heston surface').classes('mono-label').style('margin-left: auto;')

    with ui.column().classes('panel-card').style('width: 100%; max-width: 900px; margin-top: 1.5rem; padding: 0.75rem;'):
        plotly_display = ui.plotly(global_fig).classes('w-full').style('height: 700px;')

    with ui.row().classes('gap-4 flex-wrap').style('width: 100%; max-width: 900px; margin-top: 1.5rem;') as stats_card:
        with ui.column().classes('panel-card items-start').style('padding: 1rem 1.5rem; min-width: 140px; flex: 1;'):
            ui.label('SPOT').classes('mono-label')
            stats_spot = ui.label('—').classes('mono-value')
        with ui.column().classes('panel-card items-start').style('padding: 1rem 1.5rem; min-width: 140px; flex: 1;'):
            ui.label('κ  MEAN REVERSION').classes('mono-label')
            stats_kappa = ui.label('—').classes('mono-value')
        with ui.column().classes('panel-card items-start').style('padding: 1rem 1.5rem; min-width: 140px; flex: 1;'):
            ui.label('θ  LONG-TERM VAR').classes('mono-label')
            stats_theta = ui.label('—').classes('mono-value')
        with ui.column().classes('panel-card items-start').style('padding: 1rem 1.5rem; min-width: 140px; flex: 1;'):
            ui.label('σ  VOL-OF-VOL').classes('mono-label')
            stats_volvol = ui.label('—').classes('mono-value')
        with ui.column().classes('panel-card items-start').style('padding: 1rem 1.5rem; min-width: 140px; flex: 1;'):
            ui.label('ρ  CORRELATION').classes('mono-label')
            stats_rho = ui.label('—').classes('mono-value')
        with ui.column().classes('panel-card items-start').style('padding: 1rem 1.5rem; min-width: 140px; flex: 1;'):
            ui.label('v₀  INITIAL VAR').classes('mono-label')
            stats_v0 = ui.label('—').classes('mono-value')

    stats_card.set_visibility(False)

ui.run(native=True, title="Volatility Surface Viewer", window_size=(960, 950))
