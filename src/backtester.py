"""
Backtesting engine for ES scalping strategies on tick-bar data.

Supports:
  - Fixed and ATR-based stop/target sizing
  - Commission and slippage modeling
  - Trade-by-trade logging with full context
  - Walk-forward and out-of-sample validation
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd


# ES futures constants
ES_TICK_SIZE = 0.25
ES_TICK_VALUE = 12.50  # $12.50 per tick
ES_POINT_VALUE = 50.0  # $50 per point


@dataclass
class TradeConfig:
    """Configuration for trade execution modeling."""

    # Fixed stop/target in points (overridden if use_atr_sizing=True)
    stop_loss_points: float = 2.0
    take_profit_points: float = 2.0

    # ATR-based sizing
    use_atr_sizing: bool = False
    atr_stop_multiplier: float = 1.5
    atr_target_multiplier: float = 2.0

    # Cost model
    commission_per_side: float = 2.50  # $ per contract per side
    slippage_ticks: int = 1  # ticks of slippage per entry/exit

    # Risk
    contracts: int = 1
    max_daily_loss: float = 500.0  # $ max loss per day before stopping

    # Time filter: don't enter in last N minutes before session close
    no_entry_minutes_before_close: int = 15


@dataclass
class Trade:
    """A single completed trade."""

    entry_bar: int
    exit_bar: int
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    stop_price: float
    target_price: float
    exit_reason: str  # "target", "stop", "eod", "signal"
    pnl_points: float = 0.0
    pnl_dollars: float = 0.0
    contracts: int = 1
    signal_name: str = ""
    context: dict = field(default_factory=dict)

    @property
    def is_winner(self) -> bool:
        return self.pnl_points > 0

    @property
    def risk_reward_actual(self) -> float:
        risk = abs(self.entry_price - self.stop_price)
        if risk == 0:
            return 0.0
        return self.pnl_points / risk


# Type alias for signal functions
# Signal function receives bars DataFrame and current bar index,
# returns: "long", "short", or None
SignalFunc = Callable[[pd.DataFrame, int], Optional[str]]


def backtest(
    bars: pd.DataFrame,
    signal_fn: SignalFunc,
    config: TradeConfig = None,
    signal_name: str = "unnamed",
) -> list[Trade]:
    """
    Run a backtest over tick-bar data.

    Parameters
    ----------
    bars : DataFrame from build_chart() with OHLCV + indicators
    signal_fn : Function(bars, i) -> "long" | "short" | None
    config : TradeConfig for execution parameters
    signal_name : Label for this strategy

    Returns
    -------
    List of Trade objects
    """
    if config is None:
        config = TradeConfig()

    trades: list[Trade] = []
    in_trade = False
    slippage = config.slippage_ticks * ES_TICK_SIZE
    cost_per_trade = 2 * config.commission_per_side + 2 * slippage * ES_POINT_VALUE

    # Track daily P&L for max loss cutoff
    current_date = None
    daily_pnl = 0.0
    daily_stopped = False

    for i in range(len(bars)):
        row = bars.iloc[i]
        trade_date = row["timestamp"].date() if hasattr(row["timestamp"], "date") else None

        # Reset daily tracking on new day
        if trade_date != current_date:
            current_date = trade_date
            daily_pnl = 0.0
            daily_stopped = False

        if daily_stopped:
            continue

        if in_trade:
            # Check exit conditions
            exit_reason = None
            exit_price = None

            if direction == "long":
                if row["low"] <= stop_price:
                    exit_price = stop_price - slippage
                    exit_reason = "stop"
                elif row["high"] >= target_price:
                    exit_price = target_price - slippage
                    exit_reason = "target"
            else:  # short
                if row["high"] >= stop_price:
                    exit_price = stop_price + slippage
                    exit_reason = "stop"
                elif row["low"] <= target_price:
                    exit_price = target_price + slippage
                    exit_reason = "target"

            # End-of-day forced exit (check if next bar is a new day or last bar)
            is_last_bar = (i == len(bars) - 1)
            is_new_day_next = False
            if not is_last_bar:
                next_date = bars.iloc[i + 1]["timestamp"].date()
                if next_date != trade_date:
                    is_new_day_next = True

            if exit_reason is None and (is_last_bar or is_new_day_next):
                exit_price = row["close"]
                exit_reason = "eod"

            if exit_reason:
                if direction == "long":
                    pnl_points = exit_price - entry_price
                else:
                    pnl_points = entry_price - exit_price

                pnl_dollars = (pnl_points * ES_POINT_VALUE * config.contracts) - cost_per_trade

                trade = Trade(
                    entry_bar=entry_bar,
                    exit_bar=i,
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_time=entry_time,
                    exit_time=row["timestamp"],
                    stop_price=stop_price,
                    target_price=target_price,
                    exit_reason=exit_reason,
                    pnl_points=pnl_points,
                    pnl_dollars=pnl_dollars,
                    contracts=config.contracts,
                    signal_name=signal_name,
                )
                trades.append(trade)
                in_trade = False

                daily_pnl += pnl_dollars
                if daily_pnl <= -config.max_daily_loss:
                    daily_stopped = True

        if not in_trade:
            # Check for new signal
            signal = signal_fn(bars, i)
            if signal not in ("long", "short"):
                continue

            # Compute stop and target
            if config.use_atr_sizing and "kc_mid" in bars.columns:
                atr_val = bars.iloc[i].get("bar_range", 1.0)
                # Use rolling ATR from keltner if available
                if not pd.isna(bars.iloc[i].get("kc_upper")) and not pd.isna(bars.iloc[i].get("kc_lower")):
                    atr_val = (bars.iloc[i]["kc_upper"] - bars.iloc[i]["kc_lower"]) / 2
                sl_dist = atr_val * config.atr_stop_multiplier
                tp_dist = atr_val * config.atr_target_multiplier
            else:
                sl_dist = config.stop_loss_points
                tp_dist = config.take_profit_points

            if signal == "long":
                entry_price = row["close"] + slippage
                stop_price = entry_price - sl_dist
                target_price = entry_price + tp_dist
            else:
                entry_price = row["close"] - slippage
                stop_price = entry_price + sl_dist
                target_price = entry_price - tp_dist

            direction = signal
            entry_bar = i
            entry_time = row["timestamp"]
            in_trade = True

    return trades


def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    """Convert trade list to a DataFrame for analysis."""
    if not trades:
        return pd.DataFrame()

    records = []
    for t in trades:
        records.append({
            "entry_bar": t.entry_bar,
            "exit_bar": t.exit_bar,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "stop_price": t.stop_price,
            "target_price": t.target_price,
            "exit_reason": t.exit_reason,
            "pnl_points": t.pnl_points,
            "pnl_dollars": t.pnl_dollars,
            "contracts": t.contracts,
            "signal_name": t.signal_name,
            "is_winner": t.is_winner,
            "rr_actual": t.risk_reward_actual,
        })

    df = pd.DataFrame(records)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["trade_date"] = df["entry_time"].dt.date
    df["duration_bars"] = df["exit_bar"] - df["entry_bar"]
    df["cumulative_pnl"] = df["pnl_dollars"].cumsum()
    return df


def walk_forward_split(
    bars: pd.DataFrame,
    n_splits: int = 5,
    train_pct: float = 0.7,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Walk-forward analysis: split bars into sequential train/test windows.

    Each split uses an expanding or rolling training window followed by
    an out-of-sample test window. This prevents look-ahead bias.

    Returns list of (train_bars, test_bars) tuples.
    """
    total = len(bars)
    window = total // n_splits

    splits = []
    for k in range(n_splits):
        test_start = k * window
        test_end = min((k + 1) * window, total)

        if k == 0:
            # First window: use first portion as train, rest as test
            cut = int(window * train_pct)
            train = bars.iloc[:cut].copy()
            test = bars.iloc[cut:test_end].copy()
        else:
            # Use all data up to test window start as training
            train_end = test_start
            train_start = max(0, train_end - int(window / (1 - train_pct) * train_pct))
            train = bars.iloc[train_start:train_end].copy()
            test = bars.iloc[test_start:test_end].copy()

        if len(train) > 0 and len(test) > 0:
            splits.append((train, test))

    return splits
