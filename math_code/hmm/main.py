"""
main.py

NiceGUI frontend for the Black-Litterman / Max-Sharpe portfolio optimizer.

Flow:
  1. User builds a universe of tickers.
  2. User manually enters a market cap per ticker (any consistent unit);
     w_mkt is just those caps normalized over their sum.
  3. User optionally adds "views" (I believe TICKER will return X%), each
     view maps to a row of the pick matrix P and an entry of Q.
  4. On "Run optimization":
       - pull price history from yfinance                    (data_handler)
       - normalize user-entered market caps into w_mkt        (data_handler)
       - compute Sigma, Pi, Omega, mu_BL                     (black_litterman)
       - compute max-Sharpe weights off mu_BL                (optimizer)
  5. Results render as a covariance heatmap, an equilibrium/BL returns
     table, and a treemap of the optimal weights.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from nicegui import ui, run

import data_handler as dh
import black_litterman as bl
import optimizer as opt
from style import (
    apply_theme,
    ACCENT,
    ACCENT_DIM,
    AMBER,
    POSITIVE,
    NEGATIVE,
    TEXT_PRIMARY,
    TEXT_MUTED,
    BG_PANEL,
    BG_PANEL_RAISED,
    BORDER,
    FONT_MONO,
)

PERIOD_OPTIONS = {"1y": "1y", "2y": "2y", "3y": "3y", "5y": "5y", "10y": "10y"}
INTERVAL_OPTIONS = {"1d": "Daily", "1wk": "Weekly", "1mo": "Monthly"}
PERIODS_PER_YEAR = {"1d": 252, "1wk": 52, "1mo": 12}

# How far forward the resulting expected returns / covariance should
# represent, as a fraction of a year. This is independent of "Lookback"
# (how much history to train on) and "Bar interval" (sampling frequency).
HORIZON_OPTIONS = {
    "5d": "5 Trading Days",
    "1m": "1 Month",
    "3m": "3 Months",
    "6m": "6 Months",
    "1y": "1 Year",
}
HORIZON_YEARS = {
    "5d": 5 / 252,
    "1m": 1 / 12,
    "3m": 3 / 12,
    "6m": 6 / 12,
    "1y": 1.0,
}


def plotly_dark_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=BG_PANEL,
        plot_bgcolor=BG_PANEL,
        font=dict(family=FONT_MONO, color=TEXT_PRIMARY, size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        height=height,
    )
    return fig


class PortfolioApp:
    def __init__(self):
        # ---- universe ----
        self.tickers: list[str] = []

        # ---- market caps: user-entered, {ticker: market_cap}. Any consistent
        # unit works (millions, billions, raw dollars) since w_mkt is just the
        # normalized proportions. ----
        self.market_caps: dict[str, float] = {}

        # ---- views: each {id, assets:list[str], weights:dict[str,float], ret_pct:float} ----
        self.views: list[dict] = []
        self._view_id_counter = 0

        # ---- parameters ----
        self.risk_aversion = 2.5
        self.tau = 0.05
        self.risk_free_rate = 0.02
        self.period = "3y"
        self.interval = "1wk"
        self.horizon = "1y"

        # ---- computed results ----
        self.cov_matrix = None
        self.w_mkt = None
        self.market_cap_fallback = False
        self.pi = None
        self.mu_bl = None
        self.weights = None
        self.stats = None

        self.status_log = None  # ui.column set up in build()

    # ------------------------------------------------------------------
    # state helpers
    # ------------------------------------------------------------------
    def add_ticker(self, raw: str):
        symbol = raw.strip().upper()
        if not symbol:
            return
        if symbol in self.tickers:
            ui.notify(f"{symbol} is already in the universe.", type="warning")
            return
        self.tickers.append(symbol)
        self.market_caps.setdefault(symbol, 0.0)
        self.ticker_chips.refresh()
        self.market_cap_rows.refresh()

    def remove_ticker(self, symbol: str):
        self.tickers = [t for t in self.tickers if t != symbol]
        self.market_caps.pop(symbol, None)
        # drop that ticker from any view that referenced it
        for v in self.views:
            if symbol in v["assets"]:
                v["assets"].remove(symbol)
                v["weights"].pop(symbol, None)
        self.ticker_chips.refresh()
        self.market_cap_rows.refresh()
        self.view_rows.refresh()

    def add_view(self):
        self._view_id_counter += 1
        self.views.append({
            "id": self._view_id_counter,
            "assets": [],
            "weights": {},
            "ret_pct": 0.0,
        })
        self.view_rows.refresh()

    def remove_view(self, view_id: int):
        self.views = [v for v in self.views if v["id"] != view_id]
        self.view_rows.refresh()

    def build_pick_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.tickers)
        valid = [
            v for v in self.views
            if v["assets"] and any(abs(v["weights"].get(a, 0.0)) > 1e-12 for a in v["assets"])
        ]
        if not valid:
            return np.zeros((0, n)), np.zeros((0,))

        P = np.zeros((len(valid), n))
        Q = np.zeros(len(valid))
        for i, v in enumerate(valid):
            for a in v["assets"]:
                j = self.tickers.index(a)
                P[i, j] = v["weights"].get(a, 0.0)
            Q[i] = v["ret_pct"] / 100.0
        return P, Q

    def log(self, message: str, kind: str = "muted"):
        color = {"muted": TEXT_MUTED, "accent": ACCENT, "amber": AMBER, "negative": NEGATIVE}.get(kind, TEXT_MUTED)
        with self.status_log:
            ui.label(message).classes("qp-mono").style(f"color:{color}; font-size:0.78rem;")

    # ------------------------------------------------------------------
    # pipeline
    # ------------------------------------------------------------------
    async def run_pipeline(self):
        if len(self.tickers) < 2:
            ui.notify("Add at least two tickers before running the optimizer.", type="warning")
            return

        self.status_log.clear()
        self.run_button.props("loading")
        self.log(f"> universe: {', '.join(self.tickers)}")

        try:
            periods_per_year = PERIODS_PER_YEAR[self.interval]
            horizon_years = HORIZON_YEARS[self.horizon]

            self.log("> pulling price history via yfinance...")
            price_data = await run.io_bound(dh.fetch_price_history, self.tickers, self.period, self.interval)

            self.log(f"> computing covariance matrix (horizon: {HORIZON_OPTIONS[self.horizon]})...")
            self.cov_matrix, _ = await run.io_bound(dh.compute_covariance, price_data, periods_per_year, horizon_years)

            self.log("> normalizing user-entered market caps...")
            caps = {t: self.market_caps.get(t, 0.0) for t in self.tickers}
            self.w_mkt, ordered_tickers, self.market_cap_fallback = dh.compute_market_weights(caps)
            # dh.compute_market_weights preserves dict order, which matches self.tickers
            if self.market_cap_fallback:
                self.log("! one or more market caps missing/zero -> falling back to equal weighting", "amber")

            self.log("> computing implied equilibrium returns (Pi)...")
            self.pi = bl.implied_equilibrium_returns(self.cov_matrix.values, self.w_mkt, self.risk_aversion)

            P, Q = self.build_pick_matrix()
            if P.shape[0] > 0:
                self.log(f"> blending {P.shape[0]} investor view(s) via Black-Litterman...")
            else:
                self.log("> no investor views supplied -> mu_BL = Pi")
            self.mu_bl, _ = bl.posterior_returns(self.cov_matrix.values, self.pi, P, Q, self.tau)

            self.log("> solving max-Sharpe weights (Schaible transform)...")
            self.weights = opt.max_sharpe_weights(self.mu_bl, self.cov_matrix.values, self.risk_free_rate)
            self.stats = opt.portfolio_stats(self.weights, self.mu_bl, self.cov_matrix.values, self.risk_free_rate)

            self.log("> done.", "accent")
            ui.notify("Optimization complete.", type="positive")

        except (dh.DataFetchError, bl.BlackLittermanError, opt.OptimizationError) as e:
            self.log(f"! {e}", "negative")
            ui.notify(str(e), type="negative")
            self.run_button.props(remove="loading")
            return
        except Exception as e:  # pragma: no cover - defensive
            self.log(f"! unexpected error: {e}", "negative")
            ui.notify(f"Unexpected error: {e}", type="negative")
            self.run_button.props(remove="loading")
            return

        self.run_button.props(remove="loading")
        self.results_tabs.set_visibility(True)
        self.no_results_placeholder.set_visibility(False)
        self.render_covariance.refresh()
        self.render_equilibrium.refresh()
        self.render_bl.refresh()
        self.render_weights.refresh()

    # ------------------------------------------------------------------
    # left panel: universe
    # ------------------------------------------------------------------
    @ui.refreshable
    def ticker_chips(self):
        if not self.tickers:
            ui.label("No tickers yet.").classes("qp-mono qp-muted").style("font-size:0.8rem;")
            return
        with ui.row().classes("gap-2 flex-wrap"):
            for t in self.tickers:
                ui.chip(text=t, removable=True, color=None).classes("qp-ticker-chip") \
                    .on_value_change(lambda e, sym=t: None if e.value else self.remove_ticker(sym))

    @ui.refreshable
    def market_cap_rows(self):
        if not self.tickers:
            ui.label("Add tickers above to enter their market caps.") \
                .classes("qp-mono qp-muted").style("font-size:0.78rem;")
            return
        ui.label(
            "Enter a market cap per ticker (any consistent unit — e.g. all in "
            "billions). w_mkt is just each cap normalized over the total."
        ).classes("qp-mono qp-muted").style("font-size:0.72rem;")
        with ui.column().classes("w-full gap-2"):
            for t in self.tickers:
                ui.number(
                    label=t, value=self.market_caps.get(t, 0.0), step=1, format="%.2f", min=0
                ).classes("qp-mono w-full").props("outlined dense") \
                    .on_value_change(lambda e, sym=t: self.market_caps.__setitem__(sym, e.value or 0.0))

    def build_universe_panel(self):
        with ui.column().classes("qp-panel w-full").style("padding:0;"):
            ui.label("01 — UNIVERSE").classes("qp-panel-header w-full")
            with ui.column().classes("w-full gap-3").style("padding:16px;"):
                with ui.row().classes("w-full items-end gap-2"):
                    ticker_input = ui.input(label="Ticker symbol").classes("qp-mono").style("flex:1;").props(
                        "outlined dense"
                    )
                    ticker_input.on(
                        "keydown.enter",
                        lambda: (self.add_ticker(ticker_input.value), ticker_input.set_value(""))
                    )
                    ui.button("ADD", on_click=lambda: (self.add_ticker(ticker_input.value), ticker_input.set_value(""))) \
                        .classes("qp-btn-ghost").props("unelevated dense")
                self.ticker_chips()

    def build_market_caps_panel(self):
        with ui.column().classes("qp-panel w-full").style("padding:0;"):
            ui.label("02 — MARKET CAPS").classes("qp-panel-header w-full")
            with ui.column().classes("w-full gap-3").style("padding:16px;"):
                self.market_cap_rows()

    # ------------------------------------------------------------------
    # left panel: parameters
    # ------------------------------------------------------------------
    def show_horizon_info(self):
        with ui.dialog() as dialog, ui.card().classes("qp-panel").style(f"max-width:480px; border:1px solid {BORDER};"):
            ui.label("FORECAST HORIZON").classes("qp-mono").style(f"color:{ACCENT}; font-size:0.75rem; letter-spacing:0.1em;")
            ui.label(
                "This scales Sigma (and everything downstream - Pi, mu_BL, "
                "expected return/vol) to represent the chosen number of days/"
                "months ahead, instead of always annualizing to a full year."
            ).classes("qp-mono").style("font-size:0.82rem; line-height:1.5;")
            ui.separator().classes("qp-divider")
            ui.label(
                "\"Lookback\" (how much history to train on) and \"Forecast "
                "horizon\" (how far ahead the output represents) are "
                "independent choices - don't default to the longest lookback "
                "just because more data is available."
            ).classes("qp-mono").style(f"font-size:0.82rem; line-height:1.5; color:{AMBER};")
            ui.label(
                "A long lookback smooths out noise but can blend in stale, "
                "regime-shifted data (rate cycles, past crises) that no "
                "longer describes the market you're forecasting - which is "
                "its own form of overfitting to a period that isn't "
                "representative going forward. A short lookback tracks "
                "current conditions but is noisier and more sensitive to "
                "a handful of outlier moves. Match the lookback to the "
                "horizon: short horizons are usually better served by "
                "shorter, more recent lookback windows, and vice versa."
            ).classes("qp-mono qp-muted").style("font-size:0.78rem; line-height:1.5;")
            ui.button("GOT IT", on_click=dialog.close).classes("qp-btn-primary w-full").props("unelevated dense")
        dialog.open()

    def build_parameters_panel(self):
        with ui.column().classes("qp-panel w-full").style("padding:0;"):
            ui.label("03 — PARAMETERS").classes("qp-panel-header w-full")
            with ui.column().classes("w-full gap-3").style("padding:16px;"):
                with ui.row().classes("w-full gap-3"):
                    ui.number(label="Risk aversion (lambda)", value=self.risk_aversion, step=0.1, format="%.2f") \
                        .classes("qp-mono").style("flex:1;").props("outlined dense") \
                        .bind_value(self, "risk_aversion")
                    ui.number(label="Tau", value=self.tau, step=0.01, format="%.3f") \
                        .classes("qp-mono").style("flex:1;").props("outlined dense") \
                        .bind_value(self, "tau")
                with ui.row().classes("w-full gap-3"):
                    ui.number(label="Risk-free rate (annual)", value=self.risk_free_rate, step=0.005, format="%.3f") \
                        .classes("qp-mono").style("flex:1;").props("outlined dense") \
                        .bind_value(self, "risk_free_rate")
                with ui.row().classes("w-full gap-3"):
                    ui.select(options=PERIOD_OPTIONS, value=self.period, label="Lookback") \
                        .classes("qp-mono").style("flex:1;").props("outlined dense") \
                        .bind_value(self, "period")
                    ui.select(options=INTERVAL_OPTIONS, value=self.interval, label="Bar interval") \
                        .classes("qp-mono").style("flex:1;").props("outlined dense") \
                        .bind_value(self, "interval")
                with ui.row().classes("w-full items-center gap-2"):
                    ui.select(options=HORIZON_OPTIONS, value=self.horizon, label="Forecast horizon") \
                        .classes("qp-mono").style("flex:1;").props("outlined dense") \
                        .bind_value(self, "horizon")
                    ui.button(icon="info", on_click=self.show_horizon_info) \
                        .props("flat dense round size=sm").classes("qp-muted")

    # ------------------------------------------------------------------
    # left panel: views
    # ------------------------------------------------------------------
    @ui.refreshable
    def view_rows(self):
        if not self.views:
            ui.label("No views added. mu_BL will default to the equilibrium returns (Pi).") \
                .classes("qp-mono qp-muted").style("font-size:0.78rem;")
            return

        for v in self.views:
            with ui.column().classes("w-full gap-2").style(
                f"padding:12px; border:1px dashed {BORDER}; border-radius:4px; background:{BG_PANEL_RAISED};"
            ):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label(f"VIEW {v['id']}").classes("qp-mono").style(f"color:{AMBER}; font-size:0.75rem; letter-spacing:0.08em;")
                    ui.button(icon="close", on_click=lambda vid=v["id"]: self.remove_view(vid)) \
                        .props("flat dense round size=sm").classes("qp-muted")

                asset_select = ui.select(
                    options=self.tickers, value=v["assets"], multiple=True,
                    label="Assets involved in this view"
                ).classes("qp-mono w-full").props("outlined dense use-chips")

                weight_container = ui.row().classes("w-full gap-2 flex-wrap")

                def redraw_weights(container=weight_container, view=v):
                    container.clear()
                    with container:
                        if not view["assets"]:
                            ui.label("Select assets above to assign pick-matrix weights.") \
                                .classes("qp-mono qp-muted").style("font-size:0.72rem;")
                        for a in view["assets"]:
                            ui.number(
                                label=a, value=view["weights"].get(a, 0.0), step=0.25, format="%.2f"
                            ).classes("qp-mono").style("width:110px;").props("outlined dense") \
                                .on_value_change(lambda e, asset=a, view=view: view["weights"].__setitem__(asset, e.value or 0.0))

                def on_assets_change(e, view=v, redraw=redraw_weights):
                    view["assets"] = e.value or []
                    for a in list(view["weights"].keys()):
                        if a not in view["assets"]:
                            view["weights"].pop(a)
                    redraw()

                asset_select.on_value_change(on_assets_change)
                redraw_weights()

                ui.label("Positive weight = expects outperformance, negative = underperformance " \
                          "(use +1/-1 for a simple relative view, or 1.0 alone for an absolute view).") \
                    .classes("qp-mono qp-muted").style("font-size:0.68rem;")

                ui.number(
                    label="Expected return of the view (%)", value=v["ret_pct"], step=0.5, format="%.2f"
                ).classes("qp-mono w-full").props("outlined dense") \
                    .bind_value(v, "ret_pct")

            ui.separator().classes("qp-divider")

    def build_views_panel(self):
        with ui.column().classes("qp-panel w-full").style("padding:0;"):
            with ui.row().classes("w-full items-center justify-between qp-panel-header"):
                ui.label("04 — INVESTOR VIEWS (OPTIONAL)")
                ui.button("+ ADD VIEW", on_click=self.add_view).classes("qp-btn-amber").props("unelevated dense size=sm")
            with ui.column().classes("w-full gap-3").style("padding:16px;"):
                self.view_rows()

    # ------------------------------------------------------------------
    # right panel: results
    # ------------------------------------------------------------------
    @ui.refreshable
    def render_covariance(self):
        if self.cov_matrix is None:
            return
        tickers = list(self.cov_matrix.columns)
        z = self.cov_matrix.values
        fig = go.Figure(data=go.Heatmap(
            z=z, x=tickers, y=tickers,
            colorscale=[[0, "#0e1420"], [0.5, ACCENT_DIM], [1, ACCENT]],
            colorbar=dict(title="Cov", tickfont=dict(color=TEXT_MUTED)),
            hovertemplate="%{y} × %{x}: %{z:.4f}<extra></extra>",
        ))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False, autorange="reversed")
        plotly_dark_layout(fig, height=420)
        ui.plotly(fig).classes("w-full")
        ui.label(f"Sample covariance matrix from log returns, scaled to a {HORIZON_OPTIONS[self.horizon]} horizon.") \
            .classes("qp-mono qp-muted").style("font-size:0.72rem;")

    @ui.refreshable
    def render_equilibrium(self):
        if self.pi is None:
            return
        rows = [
            {"ticker": t, "w_mkt": f"{w*100:.2f}%", "pi": f"{p*100:.2f}%"}
            for t, w, p in zip(self.tickers, self.w_mkt, self.pi)
        ]
        columns = [
            {"name": "ticker", "label": "TICKER", "field": "ticker", "align": "left"},
            {"name": "w_mkt", "label": "MARKET WEIGHT", "field": "w_mkt", "align": "right"},
            {"name": "pi", "label": "IMPLIED RETURN (Π)", "field": "pi", "align": "right"},
        ]
        ui.table(columns=columns, rows=rows, row_key="ticker").classes("qp-table w-full").props("flat dense")
        if self.market_cap_fallback:
            ui.label("Market cap missing or zero for one or more tickers — equal weighting used as w_mkt.") \
                .classes("qp-mono").style(f"color:{AMBER}; font-size:0.72rem;")

    @ui.refreshable
    def render_bl(self):
        if self.mu_bl is None:
            return
        rows = [
            {
                "ticker": t,
                "pi": f"{p*100:.2f}%",
                "mu_bl": f"{m*100:.2f}%",
                "delta": f"{(m-p)*100:+.2f}%",
            }
            for t, p, m in zip(self.tickers, self.pi, self.mu_bl)
        ]
        columns = [
            {"name": "ticker", "label": "TICKER", "field": "ticker", "align": "left"},
            {"name": "pi", "label": "EQUILIBRIUM Π", "field": "pi", "align": "right"},
            {"name": "mu_bl", "label": "BLACK-LITTERMAN μ", "field": "mu_bl", "align": "right"},
            {"name": "delta", "label": "SHIFT FROM VIEWS", "field": "delta", "align": "right"},
        ]
        ui.table(columns=columns, rows=rows, row_key="ticker").classes("qp-table w-full").props("flat dense")

    @ui.refreshable
    def render_weights(self):
        if self.weights is None:
            return

        with ui.row().classes("w-full gap-4"):
            for label, value in [
                ("EXP. RETURN", f"{self.stats['expected_return']*100:.2f}%"),
                ("VOLATILITY", f"{self.stats['volatility']*100:.2f}%"),
                ("SHARPE", f"{self.stats['sharpe_ratio']:.2f}"),
            ]:
                with ui.column().classes("qp-panel gap-0").style(f"padding:10px 16px; border-color:{BORDER};"):
                    ui.label(label).classes("qp-mono qp-muted").style("font-size:0.65rem; letter-spacing:0.1em;")
                    ui.label(value).classes("qp-mono").style(f"color:{ACCENT}; font-size:1.3rem; font-weight:700;")

        colors = [POSITIVE if w >= 0 else NEGATIVE for w in self.weights]
        fig = go.Figure(go.Treemap(
            labels=self.tickers,
            parents=[""] * len(self.tickers),
            values=[max(abs(w), 1e-6) for w in self.weights],
            text=[f"{t}<br>{w*100:+.1f}%" for t, w in zip(self.tickers, self.weights)],
            textinfo="text",
            marker=dict(colors=colors, line=dict(width=2, color=BG_PANEL)),
            textfont=dict(family=FONT_MONO, size=15, color="#06110f"),
        ))
        plotly_dark_layout(fig, height=420)
        fig.update_layout(margin=dict(l=4, r=4, t=4, b=4))
        ui.plotly(fig).classes("w-full")
        ui.label("Square area = |optimal weight|. Green = long, red = short.") \
            .classes("qp-mono qp-muted").style("font-size:0.72rem;")

        rows = [{"ticker": t, "weight": f"{w*100:+.2f}%"} for t, w in zip(self.tickers, self.weights)]
        columns = [
            {"name": "ticker", "label": "TICKER", "field": "ticker", "align": "left"},
            {"name": "weight", "label": "OPTIMAL WEIGHT", "field": "weight", "align": "right"},
        ]
        ui.table(columns=columns, rows=rows, row_key="ticker").classes("qp-table w-full").props("flat dense")

    def build_results_panel(self):
        with ui.column().classes("qp-panel w-full h-full").style("padding:0;"):
            ui.label("RESULTS").classes("qp-panel-header w-full")
            with ui.column().classes("w-full").style("padding:16px;"):
                self.no_results_placeholder = ui.column().classes("w-full items-center justify-center") \
                    .style("padding:60px 0;")
                with self.no_results_placeholder:
                    ui.icon("insights", size="2.5rem").classes("qp-muted")
                    ui.label("Configure the universe and hit RUN OPTIMIZATION.") \
                        .classes("qp-mono qp-muted").style("margin-top:8px; font-size:0.8rem;")

                self.results_tabs = ui.column().classes("w-full")
                self.results_tabs.set_visibility(False)
                with self.results_tabs:
                    with ui.tabs().classes("qp-tabs w-full") as tabs:
                        t1 = ui.tab("COVARIANCE")
                        t2 = ui.tab("EQUILIBRIUM")
                        t3 = ui.tab("BLACK-LITTERMAN")
                        t4 = ui.tab("OPTIMAL WEIGHTS")
                    with ui.tab_panels(tabs, value=t1).classes("qp-tab-panels w-full"):
                        with ui.tab_panel(t1):
                            self.render_covariance()
                        with ui.tab_panel(t2):
                            self.render_equilibrium()
                        with ui.tab_panel(t3):
                            self.render_bl()
                        with ui.tab_panel(t4):
                            self.render_weights()

    # ------------------------------------------------------------------
    # page
    # ------------------------------------------------------------------
    def build(self):
        apply_theme()

        with ui.column().classes("w-full items-center").style("padding:28px 20px 60px;"):
            with ui.column().classes("w-full gap-1").style("max-width:1320px; margin-bottom:20px;"):
                ui.label("PORTFOLIO OPTIMIZATION ENGINE").classes("qp-eyebrow")
                ui.label("BLACK-LITTERMAN / MAX-SHARPE").classes("qp-title").style("font-size:1.8rem;")
                ui.label(
                    "Equilibrium-anchored expected returns, blended with your views, "
                    "solved in closed form for the max-Sharpe frontier."
                ).classes("qp-mono qp-muted").style("font-size:0.82rem;")

            with ui.row().classes("w-full gap-5 items-start").style("max-width:1320px;"):
                with ui.column().classes("gap-5").style("width:400px; flex-shrink:0;"):
                    self.build_universe_panel()
                    self.build_market_caps_panel()
                    self.build_parameters_panel()
                    self.build_views_panel()

                    self.run_button = ui.button("RUN OPTIMIZATION", on_click=self.run_pipeline) \
                        .classes("qp-btn-primary w-full").props("unelevated size=lg")

                    with ui.column().classes("qp-panel w-full gap-1").style("padding:12px 16px; max-height:220px; overflow-y:auto;"):
                        self.status_log = ui.column().classes("w-full gap-1")

                with ui.column().classes("flex-1 gap-5").style("min-width:0;"):
                    self.build_results_panel()


app_instance = PortfolioApp()
app_instance.build()

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Black-Litterman Optimizer", dark=True, reload=False, port=8080)