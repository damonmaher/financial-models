# src/prop_sim/rules.py
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from datetime import datetime

# Default rule thresholds (matches RULES_SPEC.md)
DEFAULT_RULES = {
    # Evaluation phase
    "eval": {
        "profit_target": 3000.0,
        "consistency_best_day_pct": 0.50,   # best day <= 50% of total profit
        "consistency_min_days": 5,
        "trailing_drawdown_limit": 2000.0,  # realized + unrealized trailing
        "daily_loss_limit": 1000.0,
        "max_contracts_minis": 5,
        "max_contracts_micros": 50,
        "reset_fee": 85.0,
    },
    # Funded phase
    "funded": {
        "static_max_loss": 2000.0,          # static floor
        "daily_loss_limit": 1000.0,
        "max_contracts_minis": 5,
        "max_contracts_micros": 50,
        "payout_standard": {
            "min_winning_days": 5,
            "min_win_per_day": 150.0,
            "payout_cap": 4000.0,
        },
        "payout_consistency": {
            "min_trading_days": 3,
            "min_trades_per_day": 1,
            "best_day_pct_limit": 0.40,
            "payout_cap": 6000.0,
        },
    },
}


def _ensure_peak_equity(account_state):
    """
    Ensure account_state has a peak_equity attribute used for trailing drawdown.
    If not present, initialize it to the highest seen equity (fallback to starting balance).
    """
    if not hasattr(account_state, "peak_equity"):
        start = getattr(account_state, "balance", getattr(account_state, "equity", 0.0))
        account_state.peak_equity = max(start, getattr(account_state, "equity", 0.0))
    else:
        if getattr(account_state, "equity", 0.0) > account_state.peak_equity:
            account_state.peak_equity = account_state.equity


def check_daily_drawdown(account_state, config: Dict = None) -> Tuple[bool, Optional[Dict]]:
    cfg = (config or DEFAULT_RULES)["eval"]
    limit = cfg["daily_loss_limit"]

    equity = getattr(account_state, "equity", 0.0)
    daily_pnl = getattr(account_state, "daily_pnl", 0.0)
    start_of_day_equity = equity - daily_pnl

    loss = start_of_day_equity - equity 
    violation = loss >= limit

    info = {
        "limit": limit,
        "start_of_day_equity": start_of_day_equity,
        "equity": equity,
        "loss": loss,
    }
    return violation, info if violation else None


def check_trailing_drawdown(account_state, config: Dict = None) -> Tuple[bool, Optional[Dict]]:
    cfg = (config or DEFAULT_RULES)["eval"]
    limit = cfg["trailing_drawdown_limit"]

    _ensure_peak_equity(account_state)
    peak = account_state.peak_equity
    equity = getattr(account_state, "equity", 0.0)

    trailing_dd = peak - equity
    violation = trailing_dd >= limit

    info = {
        "limit": limit,
        "peak_equity": peak,
        "equity": equity,
        "trailing_drawdown": trailing_dd,
    }
    return violation, info if violation else None


def check_max_loss(account_state, config: Dict = None) -> Tuple[bool, Optional[Dict]]:
    cfg = (config or DEFAULT_RULES)["funded"]
    limit = cfg["static_max_loss"]

    starting_balance = getattr(account_state, "balance", None)
    equity = getattr(account_state, "equity", 0.0)

    if starting_balance is None:
        return False, None

    loss = starting_balance - equity
    violation = loss >= limit

    info = {
        "limit": limit,
        "starting_balance": starting_balance,
        "equity": equity,
        "loss": loss,
    }
    return violation, info if violation else None


def check_profit_target(account_state, config: Dict = None) -> Tuple[bool, Optional[Dict]]:
    cfg = config or DEFAULT_RULES
    phase = getattr(account_state, "phase", "eval")
    target = cfg["eval"]["profit_target"] if phase == "eval" else cfg["funded"].get("profit_target", None)

    if target is None:
        return False, None

    starting_balance = getattr(account_state, "balance", 0.0)
    equity = getattr(account_state, "equity", 0.0)
    pnl = equity - starting_balance

    reached = pnl >= target
    info = {
        "phase": phase,
        "target": target,
        "starting_balance": starting_balance,
        "equity": equity,
        "pnl": pnl,
    }
    return reached, info if reached else None


def check_consistency(trade_history: List, account_state=None, config: Dict = None, phase: str = "eval") -> Tuple[bool, Optional[Dict]]:
    cfg = (config or DEFAULT_RULES)
    if phase == "eval":
        best_day_pct_limit = cfg["eval"]["consistency_best_day_pct"]
        min_days = cfg["eval"]["consistency_min_days"]
    else:
        best_day_pct_limit = cfg["funded"]["payout_consistency"]["best_day_pct_limit"]
        min_days = cfg["funded"]["payout_consistency"]["min_trading_days"]

    if not trade_history:
        return True, {"reason": "no_trades", "best_day_pct_limit": best_day_pct_limit, "min_days": min_days}

    day_sums = defaultdict(float)
    trades_per_day = defaultdict(int)
    total_profit = 0.0
    for t in trade_history:
        day = getattr(t, "day_id", None) or getattr(t, "timestamp", None)
        if isinstance(day, datetime):
            day = day.strftime("%Y-%m-%d")
        day_sums[day] += getattr(t, "pnl", 0.0)
        trades_per_day[day] += 1
        total_profit += getattr(t, "pnl", 0.0)

    if total_profit <= 0:
        return True, {"reason": "non_positive_total_profit", "total_profit": total_profit}

    best_day = max(day_sums.values())
    best_day_pct = best_day / total_profit if total_profit != 0 else 1.0
    trading_days = len([d for d, s in day_sums.items() if s != 0.0])

    violation = (best_day_pct > best_day_pct_limit) or (trading_days < min_days)

    info = {
        "best_day": best_day,
        "total_profit": total_profit,
        "best_day_pct": best_day_pct,
        "best_day_pct_limit": best_day_pct_limit,
        "trading_days": trading_days,
        "min_days": min_days,
    }
    return violation, info if violation else None


def should_switch_phase(account_state, trade_history: List = None, config: Dict = None) -> Tuple[bool, Optional[Dict]]:
    cfg = config or DEFAULT_RULES
    phase = getattr(account_state, "phase", "eval")
    if phase != "eval":
        return False, None

    reached, profit_info = check_profit_target(account_state, config=cfg)
    if not reached:
        return False, {"reason": "profit_target_not_reached", "profit_info": profit_info}

    consistency_violation, consistency_info = check_consistency(
        trade_history or [], account_state=account_state, config=cfg, phase="eval"
    )
    if consistency_violation:
        return False, {"reason": "consistency_not_met", "consistency_info": consistency_info}

    return True, {
        "reason": "ready_to_switch",
        "profit_info": profit_info,
        "consistency_info": consistency_info,
    }


def evaluate_rules(account_state, trade: Optional[object] = None, trade_history: Optional[List] = None, config: Dict = None) -> List[Dict]:
    violations = []
    cfg = config or DEFAULT_RULES
    phase = getattr(account_state, "phase", "eval")

    _ensure_peak_equity(account_state)

    dd_violation, dd_info = check_daily_drawdown(account_state, config=cfg)
    if dd_violation:
        violations.append({"rule": "daily_drawdown", "info": dd_info})

    if phase == "eval":
        td_violation, td_info = check_trailing_drawdown(account_state, config=cfg)
        if td_violation:
            violations.append({"rule": "trailing_drawdown", "info": td_info})

    if phase == "funded":
        ml_violation, ml_info = check_max_loss(account_state, config=cfg)
        if ml_violation:
            violations.append({"rule": "static_max_loss", "info": ml_info})

    # The supplied TradingView exports do not identify whether quantity refers
    # to minis or micros, so contract limits cannot be applied reliably here.
    # Evaluation consistency is a pass condition, not an intraday failure; it
    # is checked by should_switch_phase once the profit target is reached.

    if violations:
        if not hasattr(account_state, "violations"):
            account_state.violations = []
        account_state.violations.extend(violations)

    return violations


def check_payout_eligibility(account_state, trade_history: List, config: Dict = None) -> Tuple[bool, Optional[Dict]]:
    """
    Checks if the funded account has met either Option 1 (Standard) or Option 2 (Consistency) for a payout.
    """
    cfg = (config or DEFAULT_RULES)["funded"]
    
    # Aggregate PnL and trade counts by day
    day_sums = defaultdict(float)
    trades_per_day = defaultdict(int)
    
    for t in trade_history:
        day = getattr(t, "day_id", None)
        if day:
            day_sums[day] += getattr(t, "pnl", 0.0)
            trades_per_day[day] += 1
            
    total_profit = sum(day_sums.values())
    
    if total_profit <= 0:
        return False, None

    active_trading_days = [d for d, count in trades_per_day.items() if count >= 1]
    
    # --- Option 1: Standard ---
    std_cfg = cfg["payout_standard"]
    winning_days_count = sum(1 for pnl in day_sums.values() if pnl >= std_cfg["min_win_per_day"])
    
    if winning_days_count >= std_cfg["min_winning_days"]:
        payout_amount = min(total_profit, std_cfg["payout_cap"])
        return True, {
            "payout_option": "standard", 
            "payout_amount": payout_amount,
            "total_profit": total_profit
        }

    # --- Option 2: Consistency ---
    cons_cfg = cfg["payout_consistency"]
    if len(active_trading_days) >= cons_cfg["min_trading_days"]:
        best_day = max(day_sums.values()) if day_sums else 0
        best_day_pct = best_day / total_profit
        
        if best_day_pct <= cons_cfg["best_day_pct_limit"]:
            payout_amount = min(total_profit, cons_cfg["payout_cap"])
            return True, {
                "payout_option": "consistency", 
                "payout_amount": payout_amount,
                "best_day_pct": best_day_pct,
                "total_profit": total_profit
            }

    return False, None
