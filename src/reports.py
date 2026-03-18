"""
Reporting and visualization for backtested strategies.

Generates:
  - Performance summary tables
  - Equity curves
  - Trade distribution charts
  - Time-of-day heatmaps
  - Drawdown analysis
  - Strategy comparison reports
"""
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate


def performance_summary(trades_df: pd.DataFrame) -> dict:
    """
    Compute comprehensive performance metrics from a trades DataFrame.

    Returns a dict with all key metrics for evaluating a strategy.
    """
    if trades_df.empty:
        return {"total_trades": 0, "note": "No trades generated"}

    df = trades_df.copy()
    winners = df[df["is_winner"]]
    losers = df[~df["is_winner"]]

    total_pnl = df["pnl_dollars"].sum()
    total_trades = len(df)
    win_rate = len(winners) / total_trades if total_trades > 0 else 0

    avg_win = winners["pnl_dollars"].mean() if len(winners) > 0 else 0
    avg_loss = losers["pnl_dollars"].mean() if len(losers) > 0 else 0
    profit_factor = (
        abs(winners["pnl_dollars"].sum() / losers["pnl_dollars"].sum())
        if len(losers) > 0 and losers["pnl_dollars"].sum() != 0
        else float("inf") if len(winners) > 0
        else 0
    )

    # Drawdown
    cum = df["cumulative_pnl"]
    peak = cum.cummax()
    drawdown = cum - peak
    max_drawdown = drawdown.min()

    # Consecutive wins/losses
    streaks = df["is_winner"].astype(int)
    max_win_streak = _max_streak(streaks, 1)
    max_loss_streak = _max_streak(streaks, 0)

    # Daily stats
    daily = df.groupby("trade_date")["pnl_dollars"].sum()
    profitable_days = (daily > 0).sum()
    total_days = len(daily)

    # Expectancy
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # Average trade duration in bars
    avg_duration = df["duration_bars"].mean()

    # By direction
    longs = df[df["direction"] == "long"]
    shorts = df[df["direction"] == "short"]

    # By exit reason
    exit_breakdown = df.groupby("exit_reason").agg(
        count=("pnl_dollars", "count"),
        avg_pnl=("pnl_dollars", "mean"),
        total_pnl=("pnl_dollars", "sum"),
    ).to_dict("index")

    return {
        "total_trades": total_trades,
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(win_rate * 100, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "expectancy_per_trade": round(expectancy, 2),
        "max_drawdown": round(max_drawdown, 2),
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "avg_duration_bars": round(avg_duration, 1),
        "profitable_days": profitable_days,
        "total_days": total_days,
        "pct_profitable_days": round(profitable_days / total_days * 100, 1) if total_days > 0 else 0,
        "long_trades": len(longs),
        "long_pnl": round(longs["pnl_dollars"].sum(), 2) if len(longs) > 0 else 0,
        "short_trades": len(shorts),
        "short_pnl": round(shorts["pnl_dollars"].sum(), 2) if len(shorts) > 0 else 0,
        "exit_breakdown": exit_breakdown,
    }


def _max_streak(series: pd.Series, value: int) -> int:
    """Find the maximum consecutive run of `value` in a series."""
    streaks = (series != value).cumsum()
    groups = series.groupby(streaks).sum()
    return int(groups.max()) if len(groups) > 0 else 0


def print_summary(summary: dict, name: str = "") -> str:
    """Format performance summary as a readable text table."""
    if summary.get("total_trades", 0) == 0:
        return f"Strategy '{name}': No trades generated.\n"

    header = f"=== {name} ===" if name else "=== Performance Summary ==="
    lines = [
        header,
        f"{'Total Trades:':<25} {summary['total_trades']}",
        f"{'Total P&L:':<25} ${summary['total_pnl']:,.2f}",
        f"{'Win Rate:':<25} {summary['win_rate']}%",
        f"{'Avg Winner:':<25} ${summary['avg_win']:,.2f}",
        f"{'Avg Loser:':<25} ${summary['avg_loss']:,.2f}",
        f"{'Profit Factor:':<25} {summary['profit_factor']}",
        f"{'Expectancy/Trade:':<25} ${summary['expectancy_per_trade']:,.2f}",
        f"{'Max Drawdown:':<25} ${summary['max_drawdown']:,.2f}",
        f"{'Max Win Streak:':<25} {summary['max_win_streak']}",
        f"{'Max Loss Streak:':<25} {summary['max_loss_streak']}",
        f"{'Avg Duration (bars):':<25} {summary['avg_duration_bars']}",
        f"{'Profitable Days:':<25} {summary['profitable_days']}/{summary['total_days']} ({summary['pct_profitable_days']}%)",
        f"{'Long Trades:':<25} {summary['long_trades']} (P&L: ${summary['long_pnl']:,.2f})",
        f"{'Short Trades:':<25} {summary['short_trades']} (P&L: ${summary['short_pnl']:,.2f})",
        "",
        "Exit Breakdown:",
    ]

    for reason, data in summary.get("exit_breakdown", {}).items():
        lines.append(
            f"  {reason:<12} count={data['count']:<5} avg=${data['avg_pnl']:>8,.2f}  total=${data['total_pnl']:>10,.2f}"
        )

    return "\n".join(lines) + "\n"


def compare_strategies(results: dict[str, dict]) -> str:
    """
    Compare multiple strategies side by side.

    Parameters
    ----------
    results : dict of strategy_name -> performance_summary dict

    Returns formatted comparison table.
    """
    rows = []
    for name, s in results.items():
        if s.get("total_trades", 0) == 0:
            continue
        rows.append({
            "Strategy": name,
            "Trades": s["total_trades"],
            "Win Rate": f"{s['win_rate']}%",
            "P&L": f"${s['total_pnl']:,.2f}",
            "PF": s["profit_factor"],
            "Expect.": f"${s['expectancy_per_trade']:,.2f}",
            "MaxDD": f"${s['max_drawdown']:,.2f}",
            "Avg Dur": s["avg_duration_bars"],
        })

    if not rows:
        return "No strategies produced trades."

    return tabulate(rows, headers="keys", tablefmt="grid")


def plot_equity_curve(
    trades_df: pd.DataFrame,
    title: str = "Equity Curve",
    output_path: Optional[str] = None,
) -> str:
    """Plot cumulative P&L equity curve. Returns path to saved figure."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Equity curve
    ax1 = axes[0]
    ax1.plot(trades_df["exit_time"], trades_df["cumulative_pnl"], linewidth=1.5, color="#2196F3")
    ax1.fill_between(
        trades_df["exit_time"],
        trades_df["cumulative_pnl"],
        alpha=0.15,
        color="#2196F3",
    )
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.set_ylabel("Cumulative P&L ($)")
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    peak = trades_df["cumulative_pnl"].cummax()
    dd = trades_df["cumulative_pnl"] - peak
    ax2.fill_between(trades_df["exit_time"], dd, 0, alpha=0.4, color="#F44336")
    ax2.set_ylabel("Drawdown ($)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_path or "output/equity_curve.png"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_trade_distribution(
    trades_df: pd.DataFrame,
    title: str = "Trade P&L Distribution",
    output_path: Optional[str] = None,
) -> str:
    """Plot histogram of trade P&L. Returns path to saved figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # P&L histogram
    ax1 = axes[0]
    colors = ["#4CAF50" if x > 0 else "#F44336" for x in trades_df["pnl_dollars"]]
    ax1.hist(trades_df["pnl_dollars"], bins=50, color="#2196F3", edgecolor="white", alpha=0.8)
    ax1.axvline(x=0, color="black", linestyle="--")
    ax1.axvline(x=trades_df["pnl_dollars"].mean(), color="#FF9800", linestyle="--", label="Mean")
    ax1.set_title(title)
    ax1.set_xlabel("P&L ($)")
    ax1.set_ylabel("Count")
    ax1.legend()

    # P&L by day of week
    ax2 = axes[1]
    df = trades_df.copy()
    df["weekday"] = pd.to_datetime(df["trade_date"]).dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    daily = df.groupby("weekday")["pnl_dollars"].mean().reindex(day_order)
    colors = ["#4CAF50" if x > 0 else "#F44336" for x in daily.values]
    ax2.bar(daily.index, daily.values, color=colors, edgecolor="white")
    ax2.set_title("Avg P&L by Day of Week")
    ax2.set_ylabel("Avg P&L ($)")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    path = output_path or "output/trade_distribution.png"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_time_of_day_heatmap(
    trades_df: pd.DataFrame,
    title: str = "P&L by Time of Day",
    output_path: Optional[str] = None,
) -> str:
    """Heatmap showing P&L performance by hour and day of week."""
    df = trades_df.copy()
    df["hour"] = df["entry_time"].dt.hour
    df["weekday"] = df["entry_time"].dt.day_name()

    pivot = df.pivot_table(
        values="pnl_dollars",
        index="hour",
        columns="weekday",
        aggfunc="mean",
    )
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    pivot = pivot.reindex(columns=[d for d in day_order if d in pivot.columns])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="RdYlGn", center=0,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Hour of Day")
    ax.set_xlabel("Day of Week")

    plt.tight_layout()
    path = output_path or "output/time_heatmap.png"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
