"""Interactive prop-firm historical and Monte Carlo simulator."""

from copy import deepcopy
import asyncio
import csv
import os
import random
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Tuple

from fastapi import Request
from nicegui import app, ui

from .models import Trade, row_to_trade
from .monte_carlo import (
    resample_trades_with_replacement,
    run_monte_carlo_summaries,
    shuffle_trades_no_replacement,
)
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


async def _calculate_cooperatively(
    years: int, mode: str, simulations: int, values: Dict[str, float]
) -> Dict:
    """Calculate in small chunks so the hosted WebSocket remains responsive."""
    eval_trades, funded_trades, start, end = _trade_window(years)
    rules = _rules_from_inputs(values)
    sim_config = {"reset_fee": values["reset_fee"]}
    historical = run_simulation(
        eval_trades, funded_trades, values["starting_balance"], rules, sim_config
    )["financial_summary"]

    transform = (
        shuffle_trades_no_replacement
        if mode == "shuffle"
        else resample_trades_with_replacement
    )
    random.seed(17)
    summaries = []
    for index in range(simulations):
        result = run_simulation(
            transform(eval_trades),
            transform(funded_trades),
            values["starting_balance"],
            rules,
            sim_config,
        )
        summaries.append(result["financial_summary"])
        if index % 2 == 1:
            await asyncio.sleep(0.001)

    nets = [float(item["net_profit"]) for item in summaries]

    def percentile(items: List[float], pct: float) -> float:
        ordered = sorted(items)
        position = (len(ordered) - 1) * pct
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "historical": historical,
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


@app.post("/api/prop-sim")
async def prop_sim_api(request: Request) -> Dict:
    """Run over HTTP so the interactive page WebSocket remains idle."""
    payload = await request.json()
    years = int(payload["years"])
    mode = str(payload["mode"])
    simulations = int(payload["simulations"])
    if years not in {1, 2, 3} or mode not in {"resample", "shuffle"}:
        raise ValueError("Unsupported simulation selection")
    if simulations not in {50, 100, 250}:
        raise ValueError("Unsupported Monte Carlo path count")
    values = {key: float(value) for key, value in payload["values"].items()}
    return await _calculate_cooperatively(years, mode, simulations, values)


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


CLIENT_RUN_SCRIPT = r"""
async () => {
    const placeholder = document.querySelector('#prop-results');
    let results = document.querySelector('#prop-results-output');
    if (!results) {
        results = document.createElement('div');
        results.id = 'prop-results-output';
        results.className = 'w-full max-w-7xl mx-auto px-4 md:px-8 pb-8';
        const pageColumn = placeholder.closest('.nicegui-column');
        pageColumn.parentElement.insertBefore(results, pageColumn.nextSibling);
    }
    const button = document.querySelector('#run-prop-sim');
    const inputValue = (id) => {
        const node = document.querySelector(`#${id}`);
        return (node?.matches('input') ? node.value : node?.querySelector('input')?.value)
            || document.querySelector(`.field-${id} input`)?.value || '';
    };
    const numberValue = (id) => Number(inputValue(id));
    const timeframeText = inputValue('timeframe');
    const methodText = inputValue('mc-mode');
    const pathsText = inputValue('simulations');
    const values = {};
    for (const key of [
        'starting_balance', 'profit_target', 'trailing_drawdown', 'eval_daily_loss',
        'consistency_pct', 'consistency_days', 'funded_max_loss', 'funded_daily_loss',
        'reset_fee', 'payout_days', 'winning_day_min', 'payout_cap'
    ]) values[key] = numberValue(key);

    button.disabled = true;
    results.innerHTML = '<div style="padding:16px;color:#fbbf24">Running the historical replay and randomized paths…</div>';
    try {
        const response = await fetch('/api/prop-sim', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                years: parseInt(timeframeText),
                mode: methodText.startsWith('Bootstrap') ? 'resample' : 'shuffle',
                simulations: parseInt(pathsText),
                values,
            }),
        });
        if (!response.ok) throw new Error(await response.text());
        const output = await response.json();
        const hist = output.historical;
        const mc = output.monte_carlo;
        const money = (value) => `${value < 0 ? '-' : ''}$${Math.abs(value).toLocaleString(undefined, {maximumFractionDigits: 0})}`;
        const stat = (label, value) => `<div class="ps-stat" style="padding:16px"><div style="font-size:12px;text-transform:uppercase;color:#94a3b8">${label}</div><div style="font-size:20px;font-weight:700">${value}</div></div>`;
        const grid = (items) => `<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px">${items.join('')}</div>`;

        const low = Math.min(...mc.nets);
        const high = Math.max(...mc.nets);
        const binCount = 20;
        const width = high === low ? 1 : (high - low) / binCount;
        const bins = Array(binCount).fill(0);
        for (const value of mc.nets) bins[Math.min(Math.floor((value - low) / width), binCount - 1)] += 1;
        const maxBin = Math.max(...bins, 1);
        const bars = bins.map((count) => `<div title="${count} paths" style="flex:1;height:${Math.max(3, count / maxBin * 150)}px;background:#f59e0b;border-radius:3px 3px 0 0"></div>`).join('');

        results.innerHTML = `
            <div style="color:#cbd5e1;margin-bottom:16px">Results: ${output.start} – ${output.end} · ${output.eval_count.toLocaleString()} evaluation exits · ${output.funded_count.toLocaleString()} funded exits</div>
            <section class="ps-card" style="padding:20px;margin-bottom:20px">
                <h2 style="font-size:24px;font-weight:600;margin-bottom:16px">Historical replay</h2>
                ${grid([
                    stat('Net result', money(hist.net_profit)), stat('Gross payouts', money(hist.total_revenue)),
                    stat('Reset fees', money(hist.total_expenses)), stat('Evaluation passes', hist.num_passes.toLocaleString()),
                    stat('Payouts', hist.num_payouts.toLocaleString()), stat('Evaluation failures', hist.num_fails.toLocaleString()),
                    stat('Funded failures', hist.num_blows.toLocaleString()), stat('Account stages', hist.num_runs.toLocaleString())
                ])}
            </section>
            <section class="ps-card" style="padding:20px">
                <h2 style="font-size:24px;font-weight:600;margin-bottom:16px">Monte Carlo — ${methodText.startsWith('Bootstrap') ? 'bootstrap resampling' : 'order shuffling'}</h2>
                ${grid([
                    stat('Mean net result', money(mc.mean)), stat('Median net result', money(mc.median)),
                    stat('10th percentile', money(mc.p10)), stat('90th percentile', money(mc.p90)),
                    stat('Profitable paths', `${mc.profitable_pct.toFixed(1)}%`), stat('Average passes', mc.mean_passes.toFixed(1)),
                    stat('Average payouts', mc.mean_payouts.toFixed(1)), stat('Average funded failures', mc.mean_blows.toFixed(1))
                ])}
                <h3 style="font-size:18px;font-weight:600;margin:24px 0 12px">Distribution of simulated net results</h3>
                <div style="height:170px;display:flex;align-items:flex-end;gap:4px;border-bottom:1px solid #475569">${bars}</div>
                <div style="display:flex;justify-content:space-between;color:#94a3b8;font-size:12px;margin-top:6px"><span>${money(low)}</span><span>${money(high)}</span></div>
            </section>`;
    } catch (error) {
        results.innerHTML = `<section class="ps-card" style="padding:20px;border-color:#ef4444"><h2 style="color:#fca5a5;font-size:20px;font-weight:600">Simulation error</h2><p style="color:#fecaca">${error.message}</p></section>`;
    } finally {
        button.disabled = false;
    }
}
"""


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
        ui.label("Diagnostic build: 8").classes("hidden")
        ui.label(f"Process: {os.getpid()}").classes("hidden")

        with ui.card().classes("ps-card w-full p-5"):
            ui.label("Simulation setup").classes("text-2xl font-semibold")
            with ui.grid().classes("w-full grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4"):
                ui.select(STRATEGY_OPTIONS, value=STRATEGY_OPTIONS[0], label="Evaluation strategy").classes("w-full")
                ui.select(STRATEGY_OPTIONS, value=STRATEGY_OPTIONS[0], label="Funded strategy").classes("w-full")
                timeframe = ui.select({1: "1 year", 2: "2 years", 3: "3 years"}, value=3, label="Historical timeframe").props("id=timeframe").classes("w-full")
                mc_mode = ui.select(
                    {"resample": "Bootstrap — with replacement", "shuffle": "Shuffle — without replacement"},
                    value="resample",
                    label="Monte Carlo method",
                ).props("id=mc-mode").classes("w-full")

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
                    fields[key] = ui.number(label=label, value=value, min=1, step=step).classes(f"field-{key} w-full")

            simulations = ui.select({50: "50 paths", 100: "100 paths", 250: "250 paths"}, value=100, label="Monte Carlo paths").props("id=simulations").classes("w-64")
            run_button = ui.button("Run historical + Monte Carlo simulation", icon="play_arrow").props("id=run-prop-sim unelevated color=orange")

        ui.element("div").props("id=prop-results").classes("w-full")
        run_button.on("click", js_handler=CLIENT_RUN_SCRIPT)
