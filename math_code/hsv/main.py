#!/usr/bin/env python3

import asyncio
import datetime
import time
from typing import Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from nicegui import run, ui

import fft
import set_params

# ---------- Styling & Constants ----------

class Config:
    BG_PANEL = '#0e1013'
    BORDER = 'rgba(120, 210, 160, 0.22)'
    WHITE = '#f2f3f5'
    MUTED = '#8a8f98'
    GREEN = '#7fd8a4'
    
    MAX_EXPIRATIONS = 3
    CACHE_TTL = 15 * 60  # 15 minutes

# ---------- Cache ----------

# Structure: ticker -> {'history': pd.DataFrame, 'spot': float, 'expirations': list, 'timestamp': float}
_MARKET_CACHE: Dict[str, Dict[str, Any]] = {}

# ---------- Plotly Helper Functions ----------

def create_dark_scene() -> dict:
    return dict(
        xaxis=dict(
            title='Strike Price ($)',
            backgroundcolor=Config.BG_PANEL,
            gridcolor='rgba(255,255,255,0.08)',
            color=Config.MUTED,
            showbackground=True,
        ),
        yaxis=dict(
            title='Days to Maturity',
            backgroundcolor=Config.BG_PANEL,
            gridcolor='rgba(255,255,255,0.08)',
            color=Config.MUTED,
            showbackground=True,
        ),
        zaxis=dict(
            title='Fair Call Price ($)',
            backgroundcolor=Config.BG_PANEL,
            gridcolor='rgba(255,255,255,0.08)',
            color=Config.MUTED,
            showbackground=True,
        ),
    )

def create_dark_layout(title: str) -> dict:
    return dict(
        title=dict(
            text=title,
            font=dict(family='JetBrains Mono, monospace', size=16, color=Config.WHITE),
        ),
        paper_bgcolor=Config.BG_PANEL,
        plot_bgcolor=Config.BG_PANEL,
        font=dict(family='JetBrains Mono, monospace', color=Config.MUTED, size=12),
        scene=create_dark_scene(),
        margin=dict(l=10, r=10, t=50, b=10),
    )

# ---------- Async Fetching Helpers ----------

async def async_retry(func, *args, retries=3, base_delay=2.0, max_delay=12.0, label="request", **kwargs):
    """Executes func in an io_bound thread with non-blocking exponential backoff retries."""
    last_exc = None
    for attempt in range(retries):
        try:
            return await run.io_bound(func, *args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                print(f"[{label}] Attempt {attempt + 1}/{retries} failed ({type(e).__name__}: {e}); retrying in {delay:.0f}s...")
                ui.notify(f'{label}: rate limited, retrying in {delay:.0f}s...', type='warning')
                await asyncio.sleep(delay)
    raise last_exc

# ---------- Calibration CPU Pipeline ----------

def pipeline(history_data: pd.DataFrame, s_nought: float, q: float):
    """Calibrates Heston parameters and calculates FFT option fair prices (no inverse BS)."""
    # 1. Calibrate EKF Heston Parameters using pre-fetched history
    kappa, theta, vol_of_vol, rho, v0 = set_params.calc_params(history_data)

    # 2. Compute Fair Prices across Strike and Maturity Grids
    days = np.arange(180, 365, 10)
    target_ttms = days / 365.0
    strikes_axis = np.linspace(s_nought * 0.70, s_nought * 1.30, 100)
    price_matrix = []

    for ttm in target_ttms:
        r = set_params.calc_sofr_rates(ttm)
        if isinstance(r, np.ndarray):
            r = r[0]
        
        # FFT pricing outputs fair contract prices in USD
        fair_prices, strikes = fft.heston_fft_main(
            s_nought, ttm, kappa, theta, vol_of_vol, rho, v0, r, q, is_call=True
        )
        # Interpolate directly onto standardized strike grid
        standardized_prices = np.interp(strikes_axis, strikes, fair_prices)
        price_matrix.append(standardized_prices)

    return strikes_axis, target_ttms, price_matrix, (kappa, theta, vol_of_vol, rho, v0)

# ---------- Core Event Handlers ----------

async def generate_surface():
    symbol = ticker_input.value.strip().upper()
    if not symbol:
        ui.notify('Please enter a valid ticker symbol.', type='warning')
        return

    try:
        q = float(q_input.value or 0.0) / 100.0
    except ValueError:
        ui.notify('Please enter a valid numeric dividend yield.', type='warning')
        return

    # Lock UI controls
    run_button.disable()
    spinner.set_visibility(True)
    stats_card.set_visibility(False)

    try:
        # 1. EARLY CACHE CHECK
        cached_entry = _MARKET_CACHE.get(symbol)
        now = time.time()

        if cached_entry and (now - cached_entry['timestamp']) < Config.CACHE_TTL:
            ui.notify(f'Using cached data for {symbol}')
            history_data = cached_entry['history']
            s_nought = cached_entry['spot']
        else:
            ui.notify(f'Fetching market data for {symbol}...')
            ticker_object = yf.Ticker(symbol)

            # Consolidated Single Fetch: Get 2 years of daily data (used for spot price & EKF calibration)
            history_data = await async_retry(
                lambda: ticker_object.history(period='2y'),
                retries=3, base_delay=2.0, max_delay=8.0, label="history data"
            )

            if history_data.empty:
                ui.notify(f'Failed to retrieve price history for {symbol}.', type='negative')
                return

            s_nought = float(history_data['Close'].iloc[-1])

            # Cache raw market data
            _MARKET_CACHE[symbol] = {
                'history': history_data,
                'spot': s_nought,
                'timestamp': time.time()
            }

        # 2. Run CPU Calibration Pipeline
        ui.notify('Calibrating Heston model & computing fair prices...')
        strikes_axis, target_ttms, price_matrix, params = await run.io_bound(
            pipeline, history_data, s_nought, q
        )

        # 3. Render Fair Price Surface
        X_grid, Y_grid = np.meshgrid(strikes_axis, target_ttms)
        Z_grid = np.array(price_matrix)

        fig = go.Figure(data=[go.Surface(
            x=X_grid,
            y=Y_grid * 365,
            z=Z_grid,
            colorscale=[[0, '#1b2a4a'], [0.5, '#3d6fd6'], [1, '#7fd8a4']],
            showscale=True,
            colorbar=dict(title='Price ($)', tickfont=dict(color=Config.MUTED), title_font=dict(color=Config.MUTED))
        )])
        fig.update_layout(**create_dark_layout(f'{symbol}  ·  Heston Fair Call Option Price Surface ($)'))
        fig.update_layout(autosize=True, height=700)

        plotly_display.update_figure(fig)

        # 4. Update UI Statistics
        kappa, theta, vol_of_vol, rho, v0 = params
        stats_spot.set_text(f'${s_nought:,.2f}')
        stats_kappa.set_text(f'{kappa:.4f}')
        stats_theta.set_text(f'{theta:.4f}')
        stats_volvol.set_text(f'{vol_of_vol:.4f}')
        stats_rho.set_text(f'{rho:.4f}')
        stats_v0.set_text(f'{v0:.4f}')
        stats_card.set_visibility(True)

    except Exception as e:
        print(f"Error processing '{symbol}': {type(e).__name__}: {e}")
        ui.notify(f'Failed to generate surface: {e}', type='negative')

    finally:
        run_button.enable()
        spinner.set_visibility(False)

# ---------- Global Theme Head Script ----------

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

# ---------- Layout Construction ----------

default_fig = go.Figure()
default_fig.update_layout(**create_dark_layout("Enter a ticker to begin"))

with ui.column().classes('w-full items-center').style('padding: 2.5rem 1.5rem;'):
    with ui.column().classes('items-center gap-1').style('max-width: 900px; margin-bottom: 2rem;'):
        ui.label('HESTON OPTION FAIR PRICE SURFACE').classes('brand-title text-3xl text-center')

    # Controls Bar
    with ui.row().classes('items-center gap-3 panel-card').style('padding: 1.1rem 1.5rem; width: 100%; max-width: 900px;'):
        ticker_input = ui.input(
            label='Ticker Symbol',
            value='NVDA',
            placeholder='Symbol'
        ).props('outlined dense').classes('w-32')

        q_input = ui.number(
            label='Div Yield %',
            value=0.0,
            format='%.2f',
            step=0.1
        ).props('outlined dense').classes('w-28')

        with ui.icon('help_outline', size='sm', color='grey-5').classes('cursor-pointer'):
            with ui.menu().classes('p-3 bg-zinc-900 border border-zinc-700 max-w-xs'):
                ui.markdown('''
                **Estimating Dividend Yield ($q$):**
                * **Growth/Tech (NVDA, AMZN):** `0.00%`
                * **Index ETFs (SPY, QQQ):** `1.20%`–`1.50%`
                * **Dividend Stock (KO, XOM):** Use Trailing 12M Yield from Finviz/Yahoo.
                ''').classes('text-xs text-gray-300')

        run_button = ui.button('GENERATE SURFACE', on_click=generate_surface).classes('run-btn').props('unelevated')
        spinner = ui.spinner(size='lg', color='green-4').classes('ml-2')
        spinner.set_visibility(False)

        ui.label('180–365 DTE · FFT Heston Pricing').classes('mono-label').style('margin-left: auto;')

    # 3D Plot Display
    with ui.column().classes('panel-card').style('width: 100%; max-width: 900px; margin-top: 1.5rem; padding: 0.75rem;'):
        plotly_display = ui.plotly(default_fig).classes('w-full').style('height: 700px;')

    # Parameter & Stats Card Row
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

# ---------- Application Entrypoint ----------

ui.run(native=True, title="Volatility Surface Viewer", window_size=(960, 950))
