import itertools
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from nicegui import run, ui

from math_code.bl import black_litterman as bl
from math_code.bl import data_handler as dh
from math_code.bl import optimizer as opt
from math_code.hmm import hmm as hmm_engine
from math_code.hmm import learn as hmm_learn
from math_code.hmm import set_params as hmm_params
from math_code.hsv import bs as bs_engine
from math_code.hsv import set_params as hsv_params


REGIME_LABELS = [
    "Severely Bullish",
    "Bullish",
    "Slightly Bullish",
    "Stagnant",
    "Slightly Bearish",
    "Bearish",
    "Severely Bearish",
]


def page_header(title: str, subtitle: str):
    ui.label("MAHINI & MAHER LIVE MODELS").classes("text-xs tracking-widest text-gray-400")
    ui.label(title).classes("text-3xl font-bold")
    ui.label(subtitle).classes("text-sm text-gray-400 mb-4")


def result_card(title: str):
    return ui.card().classes("w-full bg-gray-900 border border-gray-700")


@ui.page("/")
def home():
    page_header("Live Financial Models", "Run the actual Python models directly from the browser.")
    with ui.column().classes("w-full max-w-4xl mx-auto gap-4"):
        with result_card("Black-Litterman"):
            ui.label("Portfolio optimization using equilibrium returns, investor views, and Max-Sharpe weights.")
            ui.button("Open Black-Litterman", on_click=lambda: ui.navigate.to("/bl")).classes("bg-blue-600 text-white")
        with result_card("Hidden Markov Model"):
            ui.label("Calibrate Heston parameters, fit the seven-state HMM, and generate regime-weighted price paths.")
            ui.button("Open HMM", on_click=lambda: ui.navigate.to("/hmm")).classes("bg-green-600 text-white")
        with result_card("Heston Stochastic Volatility"):
            ui.label("Calibrate Heston parameters with the existing EKF/QMLE implementation.")
            ui.button("Open HSV", on_click=lambda: ui.navigate.to("/hsv")).classes("bg-orange-600 text-white")


@ui.page("/bl")
def bl_page():
    page_header(
        "Black-Litterman / Max-Sharpe",
        "Enter a universe, optionally express investor views, then run the real optimization engine.",
    )

    tickers_input = ui.input(
        label="Tickers (comma-separated)", value="AAPL, MSFT, SPY"
    ).classes("w-full")
    with ui.row().classes("w-full gap-4"):
        risk_aversion = ui.number("Risk aversion (lambda)", value=2.5, step=0.1, format="%.2f").classes("flex-1")
        tau = ui.number("Tau", value=0.05, step=0.01, format="%.3f").classes("flex-1")
        risk_free = ui.number("Risk-free rate", value=0.02, step=0.005, format="%.3f").classes("flex-1")
    with ui.row().classes("w-full gap-4"):
        period = ui.select({"1y": "1y", "2y": "2y", "3y": "3y", "5y": "5y", "10y": "10y"}, value="3y", label="Lookback").classes("flex-1")
        interval = ui.select({"1d": "Daily", "1wk": "Weekly", "1mo": "Monthly"}, value="1wk", label="Bar interval").classes("flex-1")

    ui.label("Investor view (optional)").classes("font-semibold mt-4")
    ui.label("Example: AAPL:1,MSFT:-1 means AAPL should outperform MSFT by the return below. Use AAPL:1 for an absolute view.").classes("text-xs text-gray-400")
    view_expression = ui.input("View weights", placeholder="AAPL:1,MSFT:-1").classes("w-full")
    view_return = ui.number("Expected return of the view (%)", value=5.0, step=0.5, format="%.2f").classes("w-full")

    status = ui.column().classes("w-full gap-1 mt-4")
    results = ui.column().classes("w-full gap-4 mt-4")

    def log(message: str, color: str = "text-gray-400"):
        with status:
            ui.label(message).classes(f"text-xs font-mono {color}")

    def parse_view(expression: str, tickers: list[str]):
        expression = (expression or "").strip()
        if not expression:
            return np.zeros((0, len(tickers))), np.zeros(0)
        weights = np.zeros(len(tickers))
        for part in expression.split(","):
            if ":" not in part:
                raise ValueError("View must use TICKER:WEIGHT format, e.g. AAPL:1,MSFT:-1")
            ticker, raw_weight = part.split(":", 1)
            ticker = ticker.strip().upper()
            if ticker not in tickers:
                raise ValueError(f"View ticker {ticker} is not in the universe.")
            weights[tickers.index(ticker)] = float(raw_weight)
        if np.allclose(weights, 0):
            raise ValueError("The view weights cannot all be zero.")
        return weights.reshape(1, -1), np.array([float(view_return.value) / 100.0])

    async def execute_bl():
        results.clear()
        status.clear()
        raw_tickers = [x.strip().upper() for x in tickers_input.value.split(",") if x.strip()]
        tickers = list(dict.fromkeys(raw_tickers))
        if len(tickers) < 2:
            ui.notify("Add at least two tickers.", type="warning")
            return

        run_button.props("loading")
        try:
            log(f"> universe: {', '.join(tickers)}")
            ppy = {"1d": 252, "1wk": 52, "1mo": 12}[interval.value]
            log("> downloading price history...")
            prices = await run.io_bound(dh.fetch_price_history, tickers, period.value, interval.value)
            log("> computing covariance matrix...")
            cov, _ = await run.io_bound(dh.compute_covariance, prices, ppy)
            log("> retrieving market caps...")
            caps = await run.io_bound(dh.fetch_market_caps, tickers)
            w_mkt, _, fallback = dh.compute_market_weights(caps)
            if fallback:
                log("! missing market cap -> equal-weight fallback", "text-yellow-400")
            log("> computing equilibrium returns...")
            pi = bl.implied_equilibrium_returns(cov.values, w_mkt, float(risk_aversion.value))
            P, Q = parse_view(view_expression.value, tickers)
            log(f"> applying {len(Q)} investor view(s)...")
            mu_bl, _ = bl.posterior_returns(cov.values, pi, P, Q, float(tau.value))
            log("> solving Max-Sharpe portfolio...")
            weights = opt.max_sharpe_weights(mu_bl, cov.values, float(risk_free.value))
            stats = opt.portfolio_stats(weights, mu_bl, cov.values, float(risk_free.value))

            with results:
                with ui.row().classes("w-full gap-4"):
                    for label, value in [
                        ("Expected return", f"{stats['expected_return'] * 100:.2f}%"),
                        ("Volatility", f"{stats['volatility'] * 100:.2f}%"),
                        ("Sharpe", f"{stats['sharpe_ratio']:.2f}"),
                    ]:
                        with result_card(label):
                            ui.label(label).classes("text-xs text-gray-400")
                            ui.label(value).classes("text-2xl font-bold text-blue-400")

                ui.label("Optimal portfolio weights").classes("font-semibold")
                ui.table(
                    columns=[
                        {"name": "ticker", "label": "TICKER", "field": "ticker"},
                        {"name": "market", "label": "MARKET WEIGHT", "field": "market"},
                        {"name": "pi", "label": "EQUILIBRIUM RETURN", "field": "pi"},
                        {"name": "bl", "label": "BL RETURN", "field": "bl"},
                        {"name": "weight", "label": "OPTIMAL WEIGHT", "field": "weight"},
                    ],
                    rows=[
                        {
                            "ticker": t,
                            "market": f"{wm * 100:.2f}%",
                            "pi": f"{p * 100:.2f}%",
                            "bl": f"{m * 100:.2f}%",
                            "weight": f"{w * 100:+.2f}%",
                        }
                        for t, wm, p, m, w in zip(tickers, w_mkt, pi, mu_bl, weights)
                    ],
                ).classes("w-full")

                fig = go.Figure(go.Heatmap(z=cov.values, x=tickers, y=tickers, colorscale="Viridis"))
                fig.update_layout(title="Annualized covariance matrix", height=420)
                ui.plotly(fig).classes("w-full")

            log("> done.", "text-green-400")
            ui.notify("Black-Litterman optimization complete.", type="positive")
        except Exception as exc:
            log(f"! {type(exc).__name__}: {exc}", "text-red-400")
            ui.notify(str(exc), type="negative")
        finally:
            run_button.props(remove="loading")

    run_button = ui.button("RUN BLACK-LITTERMAN MODEL", on_click=execute_bl).classes("bg-blue-600 text-white mt-4")
    ui.link("← Back to home", "/").classes("mt-4")


@ui.page("/hmm")
def hmm_page():
    page_header(
        "Hidden Markov Model",
        "Calibrate Heston parameters, fit the seven-state HMM, and generate five-day regime-weighted paths.",
    )
    ticker = ui.input("Ticker", value="SPY").classes("w-full")
    status = ui.column().classes("w-full gap-1 mt-4")
    results = ui.column().classes("w-full gap-4 mt-4")

    async def execute_hmm():
        status.clear()
        results.clear()
        symbol = ticker.value.strip().upper()
        if not symbol:
            ui.notify("Enter a ticker.", type="warning")
            return
        run_button.props("loading")
        try:
            with status:
                ui.label(f"> downloading 5 years of {symbol} data...").classes("text-xs font-mono")
            df = await run.io_bound(lambda: yf.download(symbol, period="5y", auto_adjust=True, progress=False))
            if df is None or df.empty:
                raise ValueError(f"No data found for {symbol}.")
            if isinstance(df.columns, pd.MultiIndex):
                close = df.xs(symbol, level=1, axis=1)["Close"] if symbol in df.columns.get_level_values(1) else df["Close"].iloc[:, 0]
                ohlc = df.xs(symbol, level=1, axis=1)[["Open", "High", "Low", "Close"]]
            else:
                close = df["Close"]
                ohlc = df[["Open", "High", "Low", "Close"]]
            ohlc = ohlc.dropna().astype(float)
            close = ohlc["Close"]
            if len(close) < 100:
                raise ValueError("Not enough historical data to fit the HMM.")

            with status:
                ui.label("> calibrating Heston parameters...").classes("text-xs font-mono")
            kappa, theta, sigma = await run.io_bound(hmm_params.calc_params, ohlc)
            with status:
                ui.label("> building HMM matrices...").classes("text-xs font-mono")
            A, B, pi = hmm_learn.matrices(kappa, sigma, theta, close)
            daily_log_returns = np.log(close / close.shift(1))
            drift = daily_log_returns.rolling(window=10).mean().dropna().values.flatten()
            b30, b15, b05 = 0.30 / 252, 0.15 / 252, 0.05 / 252
            obs = np.zeros(len(drift), dtype=int)
            obs[drift > b30] = 0
            obs[(drift > b15) & (drift <= b30)] = 1
            obs[(drift > b05) & (drift <= b15)] = 2
            obs[(drift >= -b05) & (drift <= b05)] = 3
            obs[(drift >= -b15) & (drift < -b05)] = 4
            obs[(drift >= -b30) & (drift < -b15)] = 5
            obs[drift < -b30] = 6

            with status:
                ui.label("> fitting Baum-Welch parameters...").classes("text-xs font-mono")
            A_hmm, B_hmm, pi_hmm = await run.io_bound(hmm_engine.baum_welch_vec, obs, A, B, pi)
            alpha, _ = hmm_engine.forward_vec(obs, A_hmm, B_hmm, pi_hmm)
            today_probs = alpha[-1] / (np.sum(alpha[-1]) + 1e-300)
            tomorrow_probs = today_probs @ A_hmm

            last_date = close.index[-1]
            last_price = float(close.iloc[-1])
            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=5)
            dates = [last_date] + list(future_dates)
            daily_drifts = np.array([0.45, 0.225, 0.10, 0.0, -0.10, -0.225, -0.45]) / 252
            paths = []
            for path in itertools.product(range(7), repeat=5):
                prob = tomorrow_probs[path[0]]
                for i in range(4):
                    prob *= A_hmm[path[i], path[i + 1]]
                if prob > 1e-8:
                    price_path = [last_price]
                    current = last_price
                    for state in path:
                        current *= np.exp(daily_drifts[state])
                        price_path.append(current)
                    paths.append((prob, path, price_path))
            paths.sort(key=lambda x: x[0], reverse=True)
            paths = paths[:10]

            with results:
                with ui.row().classes("w-full gap-4"):
                    for label, value in [
                        ("Kappa", f"{kappa:.5f}"),
                        ("Theta", f"{theta:.5f}"),
                        ("Vol-of-vol", f"{sigma:.5f}"),
                        ("Current regime", REGIME_LABELS[int(np.argmax(pi_hmm))]),
                    ]:
                        with result_card(label):
                            ui.label(label).classes("text-xs text-gray-400")
                            ui.label(value).classes("font-bold text-green-400")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[d.strftime("%Y-%m-%d") for d in close.tail(5).index], y=close.tail(5).values, mode="lines+markers", name="Past 5 days"))
                for rank, (prob, path, price_path) in enumerate(paths):
                    fig.add_trace(go.Scatter(x=[d.strftime("%Y-%m-%d") for d in dates], y=price_path, mode="lines", name=f"Path {rank + 1} ({prob:.1%})", opacity=max(0.2, 1 - rank / 12)))
                fig.update_layout(title=f"{symbol} — HMM five-day prediction paths", height=500)
                ui.plotly(fig).classes("w-full")

                ui.table(
                    columns=[
                        {"name": "state", "label": "STATE", "field": "state"},
                        {"name": "prob", "label": "CURRENT PROBABILITY", "field": "prob"},
                    ],
                    rows=[
                        {"state": label, "prob": f"{p:.2%}"}
                        for label, p in zip(REGIME_LABELS, today_probs)
                    ],
                ).classes("w-full")

            with status:
                ui.label("> done.").classes("text-xs font-mono text-green-400")
            ui.notify("HMM model complete.", type="positive")
        except Exception as exc:
            with status:
                ui.label(f"! {type(exc).__name__}: {exc}").classes("text-xs font-mono text-red-400")
            ui.notify(str(exc), type="negative")
        finally:
            run_button.props(remove="loading")

    run_button = ui.button("RUN HIDDEN MARKOV MODEL", on_click=execute_hmm).classes("bg-green-600 text-white mt-4")
    ui.link("← Back to home", "/").classes("mt-4")


@ui.page("/hsv")
def hsv_page():
    page_header(
        "Heston Stochastic Volatility",
        "Calibrate the existing EKF/QMLE Heston implementation against two years of market data.",
    )
    ticker = ui.input("Ticker", value="AAPL").classes("w-full")
    status = ui.column().classes("w-full gap-1 mt-4")
    results = ui.column().classes("w-full gap-4 mt-4")

    async def execute_hsv():
        status.clear()
        results.clear()
        symbol = ticker.value.strip().upper()
        if not symbol:
            ui.notify("Enter a ticker.", type="warning")
            return
        run_button.props("loading")
        try:
            with status:
                ui.label(f"> calibrating Heston parameters for {symbol}...").classes("text-xs font-mono")
            kappa, theta, vol_of_vol, rho, v0 = await run.io_bound(hsv_params.calc_params, symbol)
            spot = await run.io_bound(lambda: float(yf.Ticker(symbol).history(period="1d")["Close"].iloc[-1]))
            with results:
                with ui.row().classes("w-full gap-4 flex-wrap"):
                    values = [
                        ("Spot", f"${spot:,.2f}"),
                        ("Kappa", f"{kappa:.5f}"),
                        ("Theta", f"{theta:.5f}"),
                        ("Vol-of-vol", f"{vol_of_vol:.5f}"),
                        ("Rho", f"{rho:.4f}"),
                        ("V0", f"{v0:.5f}"),
                        ("Long-run vol", f"{np.sqrt(theta) * 100:.2f}%"),
                        ("Current vol", f"{np.sqrt(v0) * 100:.2f}%"),
                    ]
                    for label, value in values:
                        with result_card(label):
                            ui.label(label).classes("text-xs text-gray-400")
                            ui.label(value).classes("text-xl font-bold text-orange-400")

                ui.label("Example Black-Scholes call value using the calibrated long-run volatility").classes("font-semibold")
                strike = ui.number("Strike", value=spot, step=1).classes("w-full")
                ttm = ui.number("Time to maturity (years)", value=1.0, step=0.1).classes("w-full")
                rate = ui.number("Risk-free rate", value=0.05, step=0.005).classes("w-full")
                quote = ui.label("Set strike/time/rate, then calculate.")

                def calculate_call():
                    try:
                        price = float(bs_engine.raw_bs(spot, float(strike.value), float(rate.value), float(np.sqrt(theta)), float(ttm.value)))
                        quote.set_text(f"Black-Scholes call estimate: ${price:.2f}")
                    except Exception as exc:
                        quote.set_text(f"Error: {exc}")

                ui.button("CALCULATE CALL ESTIMATE", on_click=calculate_call).classes("bg-orange-600 text-white")

            with status:
                ui.label("> done.").classes("text-xs font-mono text-green-400")
            ui.notify("Heston calibration complete.", type="positive")
        except Exception as exc:
            with status:
                ui.label(f"! {type(exc).__name__}: {exc}").classes("text-xs font-mono text-red-400")
            ui.notify(str(exc), type="negative")
        finally:
            run_button.props(remove="loading")

    run_button = ui.button("RUN HESTON STOCHASTIC VOLATILITY MODEL", on_click=execute_hsv).classes("bg-orange-600 text-white mt-4")
    ui.link("← Back to home", "/").classes("mt-4")


port = int(os.environ.get("PORT", 10000))
ui.run(host="0.0.0.0", port=port, reload=False, title="Mahini & Maher Live Models")
