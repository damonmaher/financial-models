from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

@dataclass
class Trade:
    timestamp: datetime
    phase: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    pnl: float
    day_id: str


def row_to_trade(row: Dict[str, Any], phase: str) -> Trade:
    """
    Convert a CSV DictReader row into a Trade object.

    This function is tolerant to common CSV header variants:
    - strips BOM and whitespace from headers
    - lowercases header names for matching
    - accepts multiple common names for timestamp, pnl, qty, etc.
    - tries multiple timestamp formats
    """
    # Normalize keys: strip, remove BOM, lowercase
    _norm_row: Dict[str, Any] = {}
    for k, v in row.items():
        if k is None:
            continue
        key = k.strip()
        # remove BOM if present
        if key.startswith("\ufeff"):
            key = key.lstrip("\ufeff")
        key = key.strip().lower()
        _norm_row[key] = v

    def _pick_value(*candidates: str) -> Optional[str]:
        for c in candidates:
            if c and c in _norm_row and _norm_row[c] not in (None, ""):
                return _norm_row[c]
        return None

    # Timestamp: try several common header names and formats
    ts_str = _pick_value(
        "date and time",
        "date",
        "datetime",
        "timestamp",
        "date_time",
        "time"
    )
    if ts_str is None:
        raise KeyError(f"No timestamp column found. Available headers: {list(_norm_row.keys())}")
    ts_str = str(ts_str).strip()
    ts = None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            ts = datetime.strptime(ts_str, fmt)
            break
        except Exception:
            ts = None
    if ts is None:
        # fallback to fromisoformat for ISO-like strings
        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception as e:
            raise ValueError(f"Unable to parse timestamp '{ts_str}': {e}")

    # Day id
    day_id = ts.strftime("%Y-%m-%d")

    # Side / signal
    side_raw = _pick_value("signal", "type", "side")
    side = str(side_raw).strip().lower() if side_raw is not None else ""

    # Quantity
    qty_raw = _pick_value("size (qty)", "qty", "size_qty", "size", "quantity")
    try:
        qty = float(qty_raw) if qty_raw not in (None, "") else 1.0
    except Exception:
        try:
            qty = float(str(qty_raw).replace(",", ""))
        except Exception:
            qty = 1.0

    # Entry / exit price (best-effort)
    entry_raw = _pick_value("entry price", "entry_price", "price usd", "price", "entry")
    exit_raw = _pick_value("exit price", "exit_price", "exit")
    try:
        entry_price = float(entry_raw) if entry_raw not in (None, "") else 0.0
    except Exception:
        try:
            entry_price = float(str(entry_raw).replace(",", ""))
        except Exception:
            entry_price = 0.0
    try:
        exit_price = float(exit_raw) if exit_raw not in (None, "") else 0.0
    except Exception:
        try:
            exit_price = float(str(exit_raw).replace(",", ""))
        except Exception:
            exit_price = 0.0

    # PnL: try common column names
    pnl_raw = _pick_value(
        "net pnl usd",
        "net pnl",
        "cumulative pnl usd",
        "cumulative pnl",
        "pnl",
        "cumulative pnl %"
    )
    pnl = 0.0
    if pnl_raw not in (None, ""):
        try:
            pnl = float(pnl_raw)
        except Exception:
            s = str(pnl_raw).replace(",", "").replace("%", "")
            try:
                pnl = float(s)
            except Exception:
                pnl = 0.0

    return Trade(
        timestamp=ts,
        phase=phase,
        side=side,
        qty=qty,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl=pnl,
        day_id=day_id
    )


class AccountState:
    def __init__(self, starting_balance: float, phase: str = "eval"):
        # starting_balance is the initial balance for drawdown calculations
        self.balance = float(starting_balance)
        self.equity = float(starting_balance)
        self.max_drawdown = 0.0
        self.daily_pnl = 0.0
        self.current_day: Optional[str] = None
        self.phase = phase
        self.resets = 0
        self.violations: List[Dict] = []
        # peak_equity used by rules.py for trailing drawdown
        self.peak_equity = float(starting_balance)
        # closed flag (optional) to mark account closed on immediate failure
        self.closed = False

    def start_new_day(self, day_id: str):
        self.current_day = day_id
        self.daily_pnl = 0.0

    def apply_trade(self, trade: Trade, trade_history: Optional[List[Trade]] = None, config: Optional[Dict] = None) -> Dict:
        """
        Apply a trade to the account state, run rule checks, and return a result dict.

        Returns:
            {
                "failed": bool,                # True if immediate-failure rule triggered
                "violations": List[dict],      # list of violation records from evaluate_rules
                "action": Optional[str],       # "closed", "reset_required", "none", etc.
                "info": Optional[dict]         # extra info from violation(s)
            }
        """
        # 1) Day rollover
        if getattr(trade, "day_id", None) != self.current_day:
            self.start_new_day(getattr(trade, "day_id", None))

        # 2) Update pnl and equity
        pnl = float(getattr(trade, "pnl", 0.0))
        self.daily_pnl += pnl
        self.equity += pnl

        # 3) Update max drawdown and peak equity
        drawdown = self.balance - self.equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        if self.equity > getattr(self, "peak_equity", self.balance):
            self.peak_equity = self.equity

        # 4) Run rules engine (import locally to avoid circular imports)
        try:
            from .rules import evaluate_rules
            violations = evaluate_rules(self, trade=trade, trade_history=(trade_history or []), config=config)
        except ImportError:
            # Keep this module importable by itself in a notebook, while allowing
            # genuine rule/configuration errors to surface during the live run.
            violations = []

        # 6) Determine immediate-failure rules
        immediate_rules = {"daily_drawdown", "trailing_drawdown", "static_max_loss"}
        immediate_violations = [v for v in violations if v.get("rule") in immediate_rules]

        result = {
            "failed": bool(immediate_violations),
            "violations": violations,
            "action": None,
            "info": None,
        }

        if immediate_violations:
            # Choose action: close account on immediate failure
            result["action"] = "closed"
            result["info"] = {"immediate_violations": immediate_violations}
            self.closed = True
        else:
            result["action"] = "none"

        return result

    def check_rules(self, trade: Optional[Trade] = None, trade_history: Optional[List[Trade]] = None, config: Optional[Dict] = None):
        """
        Lightweight wrapper that calls the rules engine and returns violations.
        """
        from .rules import evaluate_rules
        return evaluate_rules(self, trade=trade, trade_history=trade_history, config=config)

    def switch_phase(self, new_phase: str):
        self.phase = new_phase
