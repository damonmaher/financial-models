# src/prop_sim/simulator.py
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from .models import Trade, AccountState
from .rules import (
    should_switch_phase,
    check_payout_eligibility,
)

DEFAULT_SIM_CONFIG = {
    "reset_fee": 85.0,
    "allow_resets": True,
    "funded_continue_after_payout": True,
    "payout_withdraws_from_equity": False,
    "restart_offset_seconds": 1,  # when advancing pointer after event, skip trades with same timestamp
}


def _find_next_index_after_timestamp(trades: List[Trade], ts: datetime, start_idx: int = 0) -> int:
    """
    Return the index of the first trade whose timestamp > ts (strictly greater).
    If none, returns len(trades).
    """
    n = len(trades)
    for i in range(start_idx, n):
        if trades[i].timestamp > ts:
            return i
    return n


def _record_event(events: List[Dict], timestamp: datetime, event_type: str, info: Dict, account: AccountState, run_id: int):
    events.append({
        "timestamp": timestamp.isoformat(),
        "event_type": event_type,
        "info": json.dumps(info),
        "equity": float(account.equity),
        "daily_pnl": float(account.daily_pnl),
        "phase": account.phase,
        "run_id": run_id,
    })


def run_simulation(
    eval_trades: List[Trade],
    funded_trades: List[Trade],
    initial_balance: float,
    rules_config: Dict = None,
    sim_config: Dict = None,
) -> Dict:
    """
    Run the historical simulator using two trade streams:
      - eval_trades: trades used during evaluation phase
      - funded_trades: trades used during funded phase

    Returns a dict with keys:
      - events: List[dict]
      - equity_curve: List[dict]
      - summary_runs: List[dict]
      - financial_summary: dict
    """
    rules_cfg = rules_config or {}
    sim_cfg = {**DEFAULT_SIM_CONFIG, **(sim_config or {})}

    # pointers into trade lists
    i_eval = 0
    i_funded = 0
    n_eval = len(eval_trades)
    n_funded = len(funded_trades)

    # logs and aggregates
    events: List[Dict] = []
    equity_curve: List[Dict] = []
    summary_runs: List[Dict] = []
    revenue = 0.0
    expenses = 0.0
    run_id = 0

    # start in eval
    acct = AccountState(starting_balance=initial_balance, phase="eval")

    # helper to append equity point
    def _append_equity_point(ts: datetime):
        equity_curve.append({
            "timestamp": ts.isoformat(),
            "equity": float(acct.equity),
            "phase": acct.phase,
            "run_id": run_id,
        })

    # main loop: continue while there are trades left in the active stream
    while i_eval < n_eval or i_funded < n_funded:
        if acct.phase == "eval":
            # start a new eval run
            run_id += 1
            run_start_ts = None
            run_violations = []
            run_resets = 0
            run_max_drawdown = 0.0
            run_trades_processed = 0
            current_run_trades = []

            while i_eval < n_eval:
                trade = eval_trades[i_eval]
                if run_start_ts is None:
                    run_start_ts = trade.timestamp

                # apply trade
                current_run_trades.append(trade)
                result = acct.apply_trade(trade, trade_history=current_run_trades, config=rules_cfg)
                _record_event(events, trade.timestamp, "trade", {"qty": trade.qty, "pnl": trade.pnl}, acct, run_id)
                _append_equity_point(trade.timestamp)
                run_trades_processed += 1

                # record any violations returned
                for v in result["violations"]:
                    _record_event(events, trade.timestamp, "violation", v, acct, run_id)
                    run_violations.append(v)

                # update run max drawdown
                drawdown = acct.balance - acct.equity
                if drawdown > run_max_drawdown:
                    run_max_drawdown = drawdown

                # immediate failure?
                if result["failed"]:
                    # record fail run
                    summary_runs.append({
                        "run_id": run_id,
                        "phase_start": "eval",
                        "start_time": run_start_ts.isoformat() if run_start_ts else None,
                        "end_time": trade.timestamp.isoformat(),
                        "duration_seconds": (trade.timestamp - run_start_ts).total_seconds() if run_start_ts else None,
                        "final_equity": float(acct.equity),
                        "max_drawdown": float(run_max_drawdown),
                        "resets_in_run": run_resets,
                        "violations_count": len(run_violations),
                        "outcome": "fail",
                        "fail_info": result["info"],
                    })
                    # charge reset fee if allowed
                    if sim_cfg["allow_resets"]:
                        expenses += sim_cfg["reset_fee"]
                        run_resets += 1
                        acct.resets += 1
                        _record_event(events, trade.timestamp, "reset", {"fee": sim_cfg["reset_fee"]}, acct, run_id)

                    # reset account to fresh eval state (starting balance preserved)
                    acct = AccountState(starting_balance=initial_balance, phase="eval")
                    # advance pointer to next trade after this timestamp
                    i_eval = _find_next_index_after_timestamp(eval_trades, trade.timestamp, start_idx=i_eval+1)
                    break  # start a new run
                
                # check if ready to switch phase (eval -> funded)
                reached, info = should_switch_phase(acct, trade_history=current_run_trades, config=rules_cfg)
                if reached:
                    # record pass run
                    summary_runs.append({
                        "run_id": run_id,
                        "phase_start": "eval",
                        "start_time": run_start_ts.isoformat() if run_start_ts else None,
                        "end_time": trade.timestamp.isoformat(),
                        "duration_seconds": (trade.timestamp - run_start_ts).total_seconds() if run_start_ts else None,
                        "final_equity": float(acct.equity),
                        "max_drawdown": float(run_max_drawdown),
                        "resets_in_run": run_resets,
                        "violations_count": len(run_violations),
                        "outcome": "pass",
                        "pass_info": info,
                    })
                    _record_event(events, trade.timestamp, "pass", info, acct, run_id)
                    # A funded account begins with a fresh balance after the
                    # evaluation account passes.
                    acct = AccountState(starting_balance=initial_balance, phase="funded")
                    # set funded pointer to first funded trade with timestamp >= pass timestamp
                    i_funded = _find_next_index_after_timestamp(funded_trades, trade.timestamp, start_idx=0)
                    break  # break to funded processing
                # otherwise continue
                i_eval += 1

            # if we exhausted eval_trades without pass/fail, record incomplete run
            if i_eval >= n_eval and acct.phase == "eval":
                summary_runs.append({
                    "run_id": run_id,
                    "phase_start": "eval",
                    "start_time": run_start_ts.isoformat() if run_start_ts else None,
                    "end_time": None,
                    "duration_seconds": None,
                    "final_equity": float(acct.equity),
                    "max_drawdown": float(run_max_drawdown),
                    "resets_in_run": run_resets,
                    "violations_count": len(run_violations),
                    "outcome": "incomplete",
                })
                break  # no more eval trades; simulation ends

        elif acct.phase == "funded":
            # process funded trades
            run_id += 1
            run_start_ts = None
            run_violations = []
            run_resets = 0
            run_max_drawdown = 0.0
            run_trades_processed = 0
            payouts_in_run = 0
            current_run_trades = []

            while i_funded < n_funded:
                trade = funded_trades[i_funded]
                if run_start_ts is None:
                    run_start_ts = trade.timestamp

                current_run_trades.append(trade)

                result = acct.apply_trade(trade, trade_history=current_run_trades, config=rules_cfg)
                _record_event(events, trade.timestamp, "trade", {"qty": trade.qty, "pnl": trade.pnl}, acct, run_id)
                _append_equity_point(trade.timestamp)
                run_trades_processed += 1

                for v in result["violations"]:
                    _record_event(events, trade.timestamp, "violation", v, acct, run_id)
                    run_violations.append(v)

                drawdown = acct.balance - acct.equity
                if drawdown > run_max_drawdown:
                    run_max_drawdown = drawdown

                # immediate failure in funded (static max loss or daily drawdown)
                if result["failed"]:
                    summary_runs.append({
                        "run_id": run_id,
                        "phase_start": "funded",
                        "start_time": run_start_ts.isoformat() if run_start_ts else None,
                        "end_time": trade.timestamp.isoformat(),
                        "duration_seconds": (trade.timestamp - run_start_ts).total_seconds() if run_start_ts else None,
                        "final_equity": float(acct.equity),
                        "max_drawdown": float(run_max_drawdown),
                        "resets_in_run": run_resets,
                        "violations_count": len(run_violations),
                        "outcome": "blow",
                        "blow_info": result["info"],
                    })
                    _record_event(events, trade.timestamp, "blow", result["info"], acct, run_id)
                    # switch back to eval
                    acct = AccountState(starting_balance=initial_balance, phase="eval")
                    # advance eval pointer to next eval trade after this timestamp
                    i_eval = _find_next_index_after_timestamp(eval_trades, trade.timestamp, start_idx=0)
                    break

                # check funded payout conditions
                reached, info = check_payout_eligibility(acct, trade_history=current_run_trades, config=rules_cfg)
                
                if reached:
                    payout_amount = info.get("payout_amount", 0.0)
                    revenue += float(payout_amount)
                    payouts_in_run += 1
                    _record_event(events, trade.timestamp, "payout", info, acct, run_id)

                    if sim_cfg["payout_withdraws_from_equity"]:
                        acct.equity -= payout_amount

                    summary_runs.append({
                        "run_id": run_id,
                        "phase_start": "funded",
                        "start_time": run_start_ts.isoformat() if run_start_ts else None,
                        "end_time": trade.timestamp.isoformat(),
                        "duration_seconds": (trade.timestamp - run_start_ts).total_seconds() if run_start_ts else None,
                        "final_equity": float(acct.equity),
                        "max_drawdown": float(run_max_drawdown),
                        "resets_in_run": run_resets,
                        "violations_count": len(run_violations),
                        "outcome": "payout",
                        "payout_amount": float(payout_amount),
                        "payout_option": info.get("payout_option")
                    })

                    if not sim_cfg["funded_continue_after_payout"]:
                        acct = AccountState(starting_balance=initial_balance, phase="eval")
                        i_eval = _find_next_index_after_timestamp(eval_trades, trade.timestamp, start_idx=0)
                        
                    i_funded = _find_next_index_after_timestamp(funded_trades, trade.timestamp, start_idx=i_funded+1)
                    break

                # otherwise continue
                i_funded += 1

            # if we exhausted funded_trades without blow/payout, record incomplete funded run
            if i_funded >= n_funded and acct.phase == "funded":
                summary_runs.append({
                    "run_id": run_id,
                    "phase_start": "funded",
                    "start_time": run_start_ts.isoformat() if run_start_ts else None,
                    "end_time": None,
                    "duration_seconds": None,
                    "final_equity": float(acct.equity),
                    "max_drawdown": float(run_max_drawdown),
                    "resets_in_run": run_resets,
                    "violations_count": len(run_violations),
                    "outcome": "incomplete",
                })
                break

        else:
            # unknown phase: reset to eval
            acct = AccountState(starting_balance=initial_balance, phase="eval")

    # final aggregates
    financial_summary = {
        "total_revenue": float(revenue),
        "total_expenses": float(expenses),
        "net_profit": float(revenue - expenses),
        "num_runs": len(summary_runs),
        "num_passes": sum(1 for r in summary_runs if r["outcome"] == "pass"),
        "num_fails": sum(1 for r in summary_runs if r["outcome"] == "fail"),
        "num_payouts": sum(1 for r in summary_runs if r["outcome"] == "payout"),
        "num_blows": sum(1 for r in summary_runs if r["outcome"] == "blow"),
    }

    return {
        "events": events,
        "equity_curve": equity_curve,
        "summary_runs": summary_runs,
        "financial_summary": financial_summary,
    }
