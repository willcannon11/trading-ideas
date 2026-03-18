"""
CLI entry point for the ES trading validation system.

Usage:
    python -m src.cli backtest --data tick_data.txt --strategy sma_keltner_bounce
    python -m src.cli compare --data tick_data.txt
    python -m src.cli scan --data tick_data.txt
    python -m src.cli discover --data tick_data.txt
"""
import json
import sys

import click
import pandas as pd

from .data_loader import load_ticks, load_multiple
from .tick_aggregator import build_chart
from .backtester import backtest, trades_to_dataframe, walk_forward_split, TradeConfig
from .strategies import STRATEGY_REGISTRY, list_strategies, get_strategy
from .scanner import (
    time_of_day_analysis,
    volatility_regime,
    detect_bar_patterns,
    pattern_outcome_analysis,
    scan_opportunities,
)
from .reports import (
    performance_summary,
    print_summary,
    compare_strategies,
    plot_equity_curve,
    plot_trade_distribution,
    plot_time_of_day_heatmap,
)


@click.group()
def cli():
    """ES Scalping Strategy Validation System

    Validate, backtest, and discover trading ideas for ES futures
    using 610-tick charts with 80 SMA and Keltner Channel.
    """
    pass


def _load_and_build(data, directory, bar_size, sma_period, kc_ema, kc_atr, kc_mult, filter_rth, session_start, session_end):
    """Shared data loading logic."""
    click.echo(f"Loading tick data...")
    if directory:
        ticks = load_multiple(directory, filter_rth=filter_rth, session_start=session_start, session_end=session_end)
    else:
        ticks = load_ticks(data, filter_rth=filter_rth, session_start=session_start, session_end=session_end)

    click.echo(f"  Loaded {len(ticks):,} ticks")
    click.echo(f"  Date range: {ticks['timestamp'].min()} to {ticks['timestamp'].max()}")
    click.echo(f"Building {bar_size}-tick chart with indicators...")

    bars = build_chart(
        ticks,
        bar_size=bar_size,
        sma_period=sma_period,
        kc_ema_period=kc_ema,
        kc_atr_period=kc_atr,
        kc_atr_multiplier=kc_mult,
    )
    click.echo(f"  Generated {len(bars):,} bars")
    return ticks, bars


# Common options
_data_opts = [
    click.option("--data", "-d", type=click.Path(), help="Path to tick data file"),
    click.option("--directory", type=click.Path(), help="Directory of tick data files"),
    click.option("--bar-size", default=610, help="Ticks per bar (default: 610)"),
    click.option("--sma-period", default=80, help="SMA period (default: 80)"),
    click.option("--kc-ema", default=20, help="Keltner EMA period (default: 20)"),
    click.option("--kc-atr", default=10, help="Keltner ATR period (default: 10)"),
    click.option("--kc-mult", default=1.5, help="Keltner ATR multiplier (default: 1.5)"),
    click.option("--filter-rth/--no-filter-rth", default=True, help="Filter to RTH session"),
    click.option("--session-start", default="09:30", help="RTH session start (default: 09:30)"),
    click.option("--session-end", default="16:00", help="RTH session end (default: 16:00)"),
]


def add_data_options(func):
    for opt in reversed(_data_opts):
        func = opt(func)
    return func


@cli.command()
@add_data_options
@click.option("--strategy", "-s", default="sma_keltner_bounce", help=f"Strategy name: {', '.join(list_strategies())}")
@click.option("--stop-loss", default=2.0, help="Stop loss in points (default: 2.0)")
@click.option("--take-profit", default=2.0, help="Take profit in points (default: 2.0)")
@click.option("--atr-sizing/--fixed-sizing", default=False, help="Use ATR-based stop/target sizing")
@click.option("--output-dir", "-o", default="output", help="Output directory for charts")
@click.option("--walk-forward/--no-walk-forward", default=False, help="Run walk-forward validation")
def backtest_cmd(data, directory, bar_size, sma_period, kc_ema, kc_atr, kc_mult,
                 filter_rth, session_start, session_end,
                 strategy, stop_loss, take_profit, atr_sizing, output_dir, walk_forward):
    """Backtest a single strategy on historical tick data."""
    if not data and not directory:
        click.echo("Error: provide --data or --directory", err=True)
        sys.exit(1)

    ticks, bars = _load_and_build(data, directory, bar_size, sma_period, kc_ema, kc_atr, kc_mult, filter_rth, session_start, session_end)

    strat = get_strategy(strategy)
    config = TradeConfig(
        stop_loss_points=stop_loss,
        take_profit_points=take_profit,
        use_atr_sizing=atr_sizing,
    )

    click.echo(f"\nRunning backtest: {strat['label']}...")
    trades = backtest(bars, strat["fn"], config=config, signal_name=strategy)
    trades_df = trades_to_dataframe(trades)

    summary = performance_summary(trades_df)
    click.echo("\n" + print_summary(summary, name=strat["label"]))

    if not trades_df.empty:
        p1 = plot_equity_curve(trades_df, title=f"{strat['label']} - Equity Curve", output_path=f"{output_dir}/{strategy}_equity.png")
        p2 = plot_trade_distribution(trades_df, title=f"{strat['label']} - P&L Distribution", output_path=f"{output_dir}/{strategy}_distribution.png")
        p3 = plot_time_of_day_heatmap(trades_df, title=f"{strat['label']} - Time of Day", output_path=f"{output_dir}/{strategy}_heatmap.png")
        click.echo(f"Charts saved: {p1}, {p2}, {p3}")

        trades_df.to_csv(f"{output_dir}/{strategy}_trades.csv", index=False)
        click.echo(f"Trade log: {output_dir}/{strategy}_trades.csv")

    if walk_forward:
        click.echo("\n--- Walk-Forward Validation ---")
        splits = walk_forward_split(bars, n_splits=5)
        for k, (train, test) in enumerate(splits):
            train_trades = backtest(train, strat["fn"], config=config, signal_name=strategy)
            test_trades = backtest(test, strat["fn"], config=config, signal_name=strategy)
            train_df = trades_to_dataframe(train_trades)
            test_df = trades_to_dataframe(test_trades)
            train_s = performance_summary(train_df)
            test_s = performance_summary(test_df)
            click.echo(f"\nFold {k+1}:")
            click.echo(f"  Train: {train_s.get('total_trades', 0)} trades, WR={train_s.get('win_rate', 0)}%, PF={train_s.get('profit_factor', 0)}")
            click.echo(f"  Test:  {test_s.get('total_trades', 0)} trades, WR={test_s.get('win_rate', 0)}%, PF={test_s.get('profit_factor', 0)}")


@cli.command()
@add_data_options
@click.option("--stop-loss", default=2.0, help="Stop loss in points")
@click.option("--take-profit", default=2.0, help="Take profit in points")
@click.option("--output-dir", "-o", default="output", help="Output directory")
def compare(data, directory, bar_size, sma_period, kc_ema, kc_atr, kc_mult,
            filter_rth, session_start, session_end,
            stop_loss, take_profit, output_dir):
    """Compare all strategies head-to-head on the same data."""
    if not data and not directory:
        click.echo("Error: provide --data or --directory", err=True)
        sys.exit(1)

    ticks, bars = _load_and_build(data, directory, bar_size, sma_period, kc_ema, kc_atr, kc_mult, filter_rth, session_start, session_end)

    config = TradeConfig(stop_loss_points=stop_loss, take_profit_points=take_profit)

    results = {}
    for name, strat in STRATEGY_REGISTRY.items():
        click.echo(f"  Backtesting {strat['label']}...")
        trades = backtest(bars, strat["fn"], config=config, signal_name=name)
        trades_df = trades_to_dataframe(trades)
        summary = performance_summary(trades_df)
        results[strat["label"]] = summary

        if not trades_df.empty:
            plot_equity_curve(trades_df, title=f"{strat['label']}", output_path=f"{output_dir}/{name}_equity.png")

    click.echo("\n" + compare_strategies(results))


@cli.command()
@add_data_options
@click.option("--min-confluence", default=3.0, help="Minimum confluence score (default: 3.0)")
@click.option("--output-dir", "-o", default="output", help="Output directory")
def scan(data, directory, bar_size, sma_period, kc_ema, kc_atr, kc_mult,
         filter_rth, session_start, session_end,
         min_confluence, output_dir):
    """Scan for high-confluence trading opportunities in historical data."""
    if not data and not directory:
        click.echo("Error: provide --data or --directory", err=True)
        sys.exit(1)

    ticks, bars = _load_and_build(data, directory, bar_size, sma_period, kc_ema, kc_atr, kc_mult, filter_rth, session_start, session_end)

    click.echo(f"\nScanning for opportunities (min confluence: {min_confluence})...")
    opps = scan_opportunities(bars, min_confluence=min_confluence)
    click.echo(f"  Found {len(opps)} high-confluence setups")

    if not opps.empty:
        click.echo(f"\nTop 20 setups:")
        top = opps.head(20)[["timestamp", "close", "direction", "confluence_score", "factors"]]
        click.echo(top.to_string(index=False))

        opps.to_csv(f"{output_dir}/opportunities.csv", index=False)
        click.echo(f"\nFull results: {output_dir}/opportunities.csv")


@cli.command()
@add_data_options
@click.option("--forward-bars", default=5, help="Bars forward to measure outcome (default: 5)")
@click.option("--output-dir", "-o", default="output", help="Output directory")
def discover(data, directory, bar_size, sma_period, kc_ema, kc_atr, kc_mult,
             filter_rth, session_start, session_end,
             forward_bars, output_dir):
    """Discover new patterns and statistical edges in the data."""
    if not data and not directory:
        click.echo("Error: provide --data or --directory", err=True)
        sys.exit(1)

    ticks, bars = _load_and_build(data, directory, bar_size, sma_period, kc_ema, kc_atr, kc_mult, filter_rth, session_start, session_end)

    # Time of day analysis
    click.echo("\n--- Time of Day Analysis ---")
    tod = time_of_day_analysis(bars)
    click.echo(tod.to_string(index=False))

    # Volatility regime
    click.echo("\n--- Volatility Regime ---")
    bars["vol_regime"] = volatility_regime(bars)
    regime_counts = bars["vol_regime"].value_counts()
    click.echo(f"  Low:    {regime_counts.get('low', 0)} bars")
    click.echo(f"  Normal: {regime_counts.get('normal', 0)} bars")
    click.echo(f"  High:   {regime_counts.get('high', 0)} bars")

    # Bar patterns
    click.echo(f"\n--- Bar Pattern Discovery (forward={forward_bars} bars) ---")
    patterns = detect_bar_patterns(bars)
    results = pattern_outcome_analysis(bars, patterns, forward_bars=forward_bars)

    for pattern, stats in sorted(results.items(), key=lambda x: x[1]["p_value"]):
        sig = " ***" if stats["significant"] else ""
        click.echo(
            f"  {pattern:<25} n={stats['count']:<5} "
            f"avg_move={stats['avg_move']:>+6.2f}  "
            f"win_rate={stats['win_rate']:.1%}  "
            f"p={stats['p_value']:.4f}{sig}"
        )

    # Save results
    import os
    os.makedirs(output_dir, exist_ok=True)
    tod.to_csv(f"{output_dir}/time_of_day.csv", index=False)
    with open(f"{output_dir}/pattern_stats.json", "w") as f:
        json.dump(results, f, indent=2)
    click.echo(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    cli()
