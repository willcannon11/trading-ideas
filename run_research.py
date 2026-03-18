#!/usr/bin/env python3
"""
Quick-start script for running the research agent.

Usage:
    python run_research.py                          # full report
    python run_research.py --validate               # validate core strategy
    python run_research.py --compare                # compare all strategies
    python run_research.py --discover               # discover patterns
    python run_research.py --scan                   # scan opportunities
    python run_research.py --time                   # time of day analysis
    python run_research.py --regime                 # volatility regime analysis
    python run_research.py --server http://x:8000   # custom server URL
    python run_research.py --local                  # use cached local data
"""
import argparse
import sys

from src.research_agent import ResearchAgent
from src.backtester import TradeConfig


def main():
    parser = argparse.ArgumentParser(description="ES Scalping Research Agent")
    parser.add_argument("--server", default=None, help="Server URL (default: http://10.211.55.3:8000)")
    parser.add_argument("--session", default="rth", choices=["rth", "all"], help="Session filter")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--local", action="store_true", help="Use local cached data only")

    # Analysis modes
    parser.add_argument("--validate", action="store_true", help="Validate core strategy (sma_keltner_bounce)")
    parser.add_argument("--strategy", default="sma_keltner_bounce", help="Strategy to validate")
    parser.add_argument("--compare", action="store_true", help="Compare all strategies")
    parser.add_argument("--discover", action="store_true", help="Discover bar patterns")
    parser.add_argument("--scan", action="store_true", help="Scan for high-confluence setups")
    parser.add_argument("--time", action="store_true", help="Time of day analysis")
    parser.add_argument("--regime", action="store_true", help="Volatility regime analysis")

    # Trade config
    parser.add_argument("--stop", type=float, default=2.0, help="Stop loss points (default: 2.0)")
    parser.add_argument("--target", type=float, default=2.0, help="Take profit points (default: 2.0)")

    args = parser.parse_args()

    agent = ResearchAgent(
        server_url=args.server,
        session=args.session,
    )

    # Test connection
    info = agent.connect()
    if info.get("status") != "ok" and not args.local:
        print("\nCannot reach server. Use --local for cached data, or check the server.")
        sys.exit(1)

    config = TradeConfig(stop_loss_points=args.stop, take_profit_points=args.target)

    # If no specific mode selected, run full report
    run_any = args.validate or args.compare or args.discover or args.scan or args.time or args.regime
    if not run_any:
        agent.full_report(start_date=args.start, end_date=args.end)
        return

    if args.validate:
        agent.validate_strategy(args.strategy, config=config, start_date=args.start, end_date=args.end)
    if args.compare:
        agent.compare_all_strategies(config=config, start_date=args.start, end_date=args.end)
    if args.discover:
        agent.discover_patterns(start_date=args.start, end_date=args.end)
    if args.scan:
        agent.scan(start_date=args.start, end_date=args.end)
    if args.time:
        agent.time_analysis(start_date=args.start, end_date=args.end)
    if args.regime:
        agent.regime_analysis(start_date=args.start, end_date=args.end)


if __name__ == "__main__":
    main()
