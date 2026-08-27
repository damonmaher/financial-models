import random
from typing import List, Dict

from .models import Trade
from .simulator import run_simulation

def resample_trades_with_replacement(trades: List[Trade]) -> List[Trade]:
    """Resamples trades with replacement, mapping outcomes to original timestamps."""
    if not trades:
        return []

    sampled_trades = random.choices(trades, k=len(trades))
    resampled_sequence = []
    
    for original_time_trade, sampled_trade in zip(trades, sampled_trades):
        new_trade = Trade(
            timestamp=original_time_trade.timestamp,
            phase=original_time_trade.phase, 
            side=sampled_trade.side,
            qty=sampled_trade.qty,
            entry_price=sampled_trade.entry_price,
            exit_price=sampled_trade.exit_price,
            pnl=sampled_trade.pnl,
            day_id=original_time_trade.day_id
        )
        resampled_sequence.append(new_trade)

    return resampled_sequence

def shuffle_trades_no_replacement(trades: List[Trade]) -> List[Trade]:
    """Shuffles trade outcomes without replacement, mapping to original timestamps."""
    if not trades:
        return []

    # Copy to avoid mutating original list, then shuffle
    shuffled_trades = trades.copy()
    random.shuffle(shuffled_trades)
    
    shuffled_sequence = []
    for original_time_trade, shuffled_trade in zip(trades, shuffled_trades):
        new_trade = Trade(
            timestamp=original_time_trade.timestamp,
            phase=original_time_trade.phase, 
            side=shuffled_trade.side,
            qty=shuffled_trade.qty,
            entry_price=shuffled_trade.entry_price,
            exit_price=shuffled_trade.exit_price,
            pnl=shuffled_trade.pnl,
            day_id=original_time_trade.day_id
        )
        shuffled_sequence.append(new_trade)

    return shuffled_sequence

def run_monte_carlo(
    eval_trades: List[Trade],
    funded_trades: List[Trade],
    initial_balance: float,
    num_simulations: int = 100,
    mode: str = "resample",
    rules_config: Dict = None,
    sim_config: Dict = None
) -> List[Dict]:
    """
    Runs the historical simulator multiple times on randomized trade sequences.
    Mode can be 'resample' (with replacement) or 'shuffle' (no replacement).
    """
    all_results = []
    
    for i in range(num_simulations):
        if mode == "shuffle":
            mc_eval = shuffle_trades_no_replacement(eval_trades)
            mc_funded = shuffle_trades_no_replacement(funded_trades)
        else:
            mc_eval = resample_trades_with_replacement(eval_trades)
            mc_funded = resample_trades_with_replacement(funded_trades)

        result = run_simulation(
            eval_trades=mc_eval,
            funded_trades=mc_funded,
            initial_balance=initial_balance,
            rules_config=rules_config,
            sim_config=sim_config
        )
        all_results.append(result)
        
    return all_results


def run_monte_carlo_summaries(
    eval_trades: List[Trade],
    funded_trades: List[Trade],
    initial_balance: float,
    num_simulations: int = 100,
    mode: str = "resample",
    rules_config: Dict = None,
    sim_config: Dict = None,
    seed: int = 17,
) -> List[Dict]:
    """Run Monte Carlo paths while retaining only compact financial summaries."""
    random.seed(seed)
    summaries = []
    for _ in range(num_simulations):
        transform = shuffle_trades_no_replacement if mode == "shuffle" else resample_trades_with_replacement
        result = run_simulation(
            eval_trades=transform(eval_trades),
            funded_trades=transform(funded_trades),
            initial_balance=initial_balance,
            rules_config=rules_config,
            sim_config=sim_config,
        )
        summaries.append(result["financial_summary"])
    return summaries
