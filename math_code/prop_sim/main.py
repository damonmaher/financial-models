"""Interactive prop-firm historical and Monte Carlo simulator."""

from copy import deepcopy
import csv
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Tuple

from nicegui import ui

from .models import Trade, row_to_trade
from .monte_carlo import run_monte_carlo_summaries
from .rules import DEFAULT_RULES
from .simulator import run_simulation


DATA_DIR = Path(__file__).resolve().parent / "data"
STRATEGY_OPTIONS = ["Built-in historical strategy"]


@lru_cache(maxsize=2)
def _load_realized_trades(filename: str, phase: str) -> Tuple[Trade, ...]:
    """Load realized trades, using only TradingView exit rows to avoid double counting."""
    trades: List[Trade] = []
    with (DATA_DIR / filename).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if not str(row.get("Type", "")).strip().lower().startswith("exit"):
                continue
            trades.append(row_to_trade(row, phase))
    trades.sort(key=lambda trade: trade.timestamp)
    return tuple(trades)


def _trade_window(years: int) -> Tuple[List[Trade], List[Trade], datetime, datetime]:
    eval_all = _load_realized_trades("eval_trades.csv", "eval")
    funded_all = _load_realized_trades("funded_trades.csv", "funded")
    end = max(eval_all[-1].timestamp, funded_all[-1].timestamp)
    try:
        start = end.replace(year=end.year - years)
    except ValueError:
        start = end.replace(year=end.year - years, day=28)
    return (
        [trade for trade in eval_all if start <= trade.timestamp <= end],
        [trade for trade in funded_all if start <= trade.timestamp <= end],
        start,
        end,
    )


def _rules_from_inputs(values: Dict[str, float]) -> Dict:
    rules = deepcopy(DEFAULT_RULES)
    rules["eval"].update({
        "profit_target": values["profit_target"],
        "consistency_best_day_pct": values["consistency_pct"] / 100.0,
        "consistency_min_days": int(values["consistency_days"]),
        "trailing_drawdown_limit": values["trailing_drawdown"],
        "daily_loss_limit": values["eval_daily_loss"],
        "reset_fee": values["reset_fee"],
    })
    rules["funded"].update({
        "static_max_loss": values["funded_max_loss"],
        "daily_loss_limit": values["funded_daily_loss"],
    })
    rules["funded"]["payout_standard"].update({
        "min_winning_days": int(values["payout_days"]),
        "min_win_per_day": values["winning_day_min"],
        "payout_cap": values["payout_cap"],
    })
    return rules


def _calculate(years: int, mode: str, simulations: int, values: Dict[str, float]) -> Dict:
    eval_trades, funded_trades, start, end = _trade_window(years)
    rules = _rules_from_inputs(values)
    sim_config = {"reset_fee": values["reset_fee"]}
    historical = run_simulation(
        eval_trades, funded_trades, values["starting_balance"], rules, sim_config
    )
    summaries = run_monte_carlo_summaries(
        eval_trades,
        funded_trades,
        values["starting_balance"],
        num_simulations=simulations,
        mode=mode,
        rules_config=rules,
        sim_config=sim_config,
    )
    nets = [float(item["net_profit"]) for item in summaries]

    def percentile(items: List[float], pct: float) -> float:
        ordered = sorted(items)
        position = (len(ordered) - 1) * pct
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "historical": historical["financial_summary"],
        "monte_carlo": {
            "mean": float(mean(nets)),
            "median": float(median(nets)),
            "p10": percentile(nets, 0.10),
            "p90": percentile(nets, 0.90),
            "profitable_pct": sum(value > 0 for value in nets) / len(nets) * 100.0,
            "mean_passes": float(mean(item["num_passes"] for item in summaries)),
            "mean_payouts": float(mean(item["num_payouts"] for item in summaries)),
            "mean_blows": float(mean(item["num_blows"] for item in summaries)),
            "nets": nets,
        },
        "start": start.strftime("%b %d, %Y"),
        "end": end.strftime("%b %d, %Y"),
        "eval_count": len(eval_trades),
        "funded_count": len(funded_trades),
    }


def _money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def _histogram(values: List[float], bins: int = 20) -> Tuple[List[str], List[int]]:
    low, high = min(values), max(values)
    if low == high:
        return [_money(low)], [len(values)]
    width = (high - low) / bins
    counts = [0] * bins
    for value in values:
        index = min(int((value - low) / width), bins - 1)
        counts[index] += 1
    labels = [_money(low + (index + 0.5) * width) for index in range(bins)]
    return labels, counts


def render_prop_sim() -> None:
    ui.add_head_html("""
        <style>
            body { background: #07111f; color: #e8eef8; }
            .ps-card { background: #101e31; border: 1px solid #263a55; border-radius: 16px; }
            .ps-stat { background: #0b1728; border: 1px solid #263a55; border-radius: 12px; }
            .q-field__native, .q-field__input, .q-field__label { color: #e8eef8 !important; }
        </style>
    """)

    with ui.column().classes("w-full max-w-7xl mx-auto p-4 md:p-8 gap-6"):
        ui.label("Prop Firm Strategy Simulator").classes("text-3xl md:text-5xl font-bold")
        ui.label(
            "Replay the included evaluation and funded strategies under adjustable account rules, "
            "then test how sensitive the outcome is to trade order and resampling."
        ).classes("text-base md:text-lg text-slate-300 max-w-4xl")
        ui.label(
            "Research simulation only. Historical and randomized outcomes are not forecasts."
        ).classes("text-sm text-amber-300")
        ui.label("Engine build: 2026").classes("text-xs text-slate-500")
        ui.label("Diagnostic build: 4").classes("hidden")
        ui.label(f"Process: {os.getpid()}").classes("hidden")

        with ui.card().classes("ps-card w-full p-5"):
            ui.label("Simulation setup").classes("text-2xl font-semibold")
            with ui.grid().classes("w-full grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4"):
                ui.select(STRATEGY_OPTIONS, value=STRATEGY_OPTIONS[0], label="Evaluation strategy").classes("w-full")
                ui.select(STRATEGY_OPTIONS, value=STRATEGY_OPTIONS[0], label="Funded strategy").classes("w-full")
                timeframe = ui.select({1: "1 year", 2: "2 years", 3: "3 years"}, value=3, label="Historical timeframe").classes("w-full")
                mc_mode = ui.select(
                    {"resample": "Bootstrap — with replacement", "shuffle": "Shuffle — without replacement"},
                    value="resample",
                    label="Monte Carlo method",
                ).classes("w-full")

            ui.separator().classes("my-3")
            ui.label("Adjustable account rules").classes("text-xl font-semibold")
            fields = {}
            defaults = {
                "starting_balance": ("Starting balance", 50000, 1000),
                "profit_target": ("Evaluation profit target", 3000, 100),
                "trailing_drawdown": ("Evaluation trailing drawdown", 2000, 100),
                "eval_daily_loss": ("Evaluation daily loss limit", 1000, 100),
                "consistency_pct": ("Best-day limit (%)", 50, 1),
                "consistency_days": ("Minimum evaluation days", 5, 1),
                "funded_max_loss": ("Funded maximum loss", 2000, 100),
                "funded_daily_loss": ("Funded daily loss limit", 1000, 100),
                "reset_fee": ("Evaluation reset fee", 85, 5),
                "payout_days": ("Winning days for payout", 5, 1),
                "winning_day_min": ("Minimum profit per winning day", 150, 25),
                "payout_cap": ("Payout cap", 4000, 100),
            }
            with ui.grid().classes("w-full grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4"):
                for key, (label, value, step) in defaults.items():
                    fields[key] = ui.number(label=label, value=value, min=1, step=step).classes("w-full")

            simulations = ui.select({50: "50 paths", 100: "100 paths", 250: "250 paths"}, value=100, label="Monte Carlo paths").classes("w-64")
            run_button = ui.button("Run historical + Monte Carlo simulation", icon="play_arrow").props("unelevated color=orange")

        results = ui.column().classes("w-full gap-5")

        async def run_model() -> None:
            values = {key: float(field.value) for key, field in fields.items()}
            if values["profit_target"] <= 0 or values["starting_balance"] <= 0:
                ui.notify("Balance and profit target must be positive.", type="negative")
                return
            run_button.disable()
            run_button.props("loading")
            ui.notify("Running the historical replay and randomized paths…", type="info")
            try:
                # The calculation is intentionally compact and completes in a
                # few seconds. Running it in-process avoids worker creation on
                # constrained single-process hosting plans.
                output = _calculate(
                    int(timeframe.value), str(mc_mode.value), int(simulations.value), values
                )
                results.clear()
                with results:
                    ui.label(
                        f"Results: {output['start']} – {output['end']} · "
                        f"{output['eval_count']:,} evaluation exits · {output['funded_count']:,} funded exits"
                    ).classes("text-sm text-slate-300")
                    with ui.card().classes("ps-card w-full p-5"):
                        ui.label("Historical replay").classes("text-2xl font-semibold")
                        hist = output["historical"]
                        with ui.grid().classes("w-full grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3"):
                            for label, value in (
                                ("Net result", _money(hist["net_profit"])),
                                ("Gross payouts", _money(hist["total_revenue"])),
                                ("Reset fees", _money(hist["total_expenses"])),
                                ("Evaluation passes", f"{hist['num_passes']:,}"),
                                ("Payouts", f"{hist['num_payouts']:,}"),
                                ("Evaluation failures", f"{hist['num_fails']:,}"),
                                ("Funded failures", f"{hist['num_blows']:,}"),
                                ("Account stages", f"{hist['num_runs']:,}"),
                            ):
                                with ui.column().classes("ps-stat p-4 gap-1"):
                                    ui.label(label).classes("text-xs uppercase tracking-wide text-slate-400")
                                    ui.label(value).classes("text-xl font-bold")

                    mc = output["monte_carlo"]
                    with ui.card().classes("ps-card w-full p-5"):
                        method = "bootstrap resampling" if mc_mode.value == "resample" else "order shuffling"
                        ui.label(f"Monte Carlo — {method}").classes("text-2xl font-semibold")
                        with ui.grid().classes("w-full grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3"):
                            for label, value in (
                                ("Mean net result", _money(mc["mean"])),
                                ("Median net result", _money(mc["median"])),
                                ("10th percentile", _money(mc["p10"])),
                                ("90th percentile", _money(mc["p90"])),
                                ("Profitable paths", f"{mc['profitable_pct']:.1f}%"),
                                ("Average passes", f"{mc['mean_passes']:.1f}"),
                                ("Average payouts", f"{mc['mean_payouts']:.1f}"),
                                ("Average funded failures", f"{mc['mean_blows']:.1f}"),
                            ):
                                with ui.column().classes("ps-stat p-4 gap-1"):
                                    ui.label(label).classes("text-xs uppercase tracking-wide text-slate-400")
                                    ui.label(value).classes("text-xl font-bold")
                        labels, counts = _histogram(mc["nets"], bins=20)
                        ui.echart({
                            "backgroundColor": "#101e31",
                            "title": {"text": "Distribution of simulated net results", "textStyle": {"color": "#e8eef8"}},
                            "tooltip": {"trigger": "axis"},
                            "grid": {"left": 55, "right": 20, "top": 60, "bottom": 65},
                            "xAxis": {
                                "type": "category",
                                "data": labels,
                                "name": "Net result",
                                "axisLabel": {"color": "#94a3b8", "rotate": 45},
                                "nameTextStyle": {"color": "#94a3b8"},
                            },
                            "yAxis": {
                                "type": "value",
                                "name": "Paths",
                                "axisLabel": {"color": "#94a3b8"},
                                "nameTextStyle": {"color": "#94a3b8"},
                                "splitLine": {"lineStyle": {"color": "#263a55"}},
                            },
                            "series": [{"type": "bar", "data": counts, "itemStyle": {"color": "#f59e0b"}}],
                        }).classes("w-full h-96")
            except Exception as exc:
                results.clear()
                with results:
                    with ui.card().classes("ps-card w-full p-5 border-red-500"):
                        ui.label("Simulation error").classes("text-xl font-semibold text-red-300")
                        ui.label(f"{type(exc).__name__}: {exc}").classes("text-sm text-red-200")
                ui.notify(f"Simulation could not complete: {exc}", type="negative", timeout=0)
            finally:
                run_button.enable()
                run_button.props(remove="loading")

        run_button.on_click(run_model)
