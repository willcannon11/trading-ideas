"""
Research agent for ES scalping strategy validation.

This is the main entry point for running research queries against your
historical data. It connects to your Windows server, pulls bar data,
and runs backtests, pattern discovery, and strategy validation.

Usage:
    from src.research_agent import ResearchAgent

    agent = ResearchAgent()           # connects to your Windows server
    agent.validate_strategy("sma_keltner_bounce")
    agent.compare_all_strategies()
    agent.discover_patterns()
    agent.time_analysis()
    agent.full_report()
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .api_client import ESDataClient
from .backtester import backtest, trades_to_dataframe, walk_forward_split, TradeConfig
from .strategies import STRATEGY_REGISTRY, get_strategy
from .scanner import (
    time_of_day_analysis,
    volatility_regime,
    detect_bar_patterns,
    pattern_outcome_analysis,
    scan_opportunities,
    confluence_score,
)
from .reports import (
    performance_summary,
    print_summary,
    compare_strategies,
    plot_equity_curve,
    plot_trade_distribution,
    plot_time_of_day_heatmap,
)
from .tick_aggregator import compute_sma, compute_keltner_channel


OUTPUT_DIR = Path(__file__).parent.parent / "output"


class ResearchAgent:
    """
    Research agent that connects to your data server and runs analysis.

    Manages data fetching, caching, and orchestrates all analysis modules.
    """

    def __init__(
        self,
        server_url: str = None,
        session: str = "rth",
        sma_period: int = 80,
        kc_ema: int = 20,
        kc_atr: int = 10,
        kc_mult: float = 1.5,
        output_dir: str = None,
    ):
        self.client = ESDataClient(base_url=server_url)
        self.session = session
        self.sma_period = sma_period
        self.kc_ema = kc_ema
        self.kc_atr = kc_atr
        self.kc_mult = kc_mult
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cached data
        self._bars: Optional[pd.DataFrame] = None
        self._bars_with_indicators: Optional[pd.DataFrame] = None

    # ---- Connection ----

    def connect(self) -> dict:
        """Test connection to the data server."""
        health = self.client.health_check()
        if health.get("status") != "ok":
            print(f"WARNING: Server not reachable: {health.get('message')}")
            print("Attempting to load from local cache...")
            return health

        info = self.client.check_data()
        print(f"Connected to server")
        print(f"  Bars: {info['rows']:,}")
        print(f"  Columns: {info['columns']}")
        print(f"  Range: {info['first_time']} to {info['last_time']}")
        return info

    # ---- Data Loading ----

    def load_bars(
        self,
        start_date: str = None,
        end_date: str = None,
        force_refresh: bool = False,
        use_local: bool = False,
    ) -> pd.DataFrame:
        """
        Load bar data from server or local cache.

        First tries the server. If unreachable, falls back to cached parquet.
        """
        if self._bars is not None and not force_refresh:
            df = self._bars
            if start_date:
                df = df[df["timestamp"] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df["timestamp"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1)]
            return df

        if use_local:
            return self._load_local()

        try:
            # Try server first — get indicators computed server-side
            print("Fetching data from server...")
            df = self.client.get_indicators(
                start_date=start_date,
                end_date=end_date,
                session=self.session,
                sma_period=self.sma_period,
                kc_ema_period=self.kc_ema,
                kc_atr_period=self.kc_atr,
                kc_atr_multiplier=self.kc_mult,
            )
            if df.empty:
                print("No data returned, trying local cache...")
                return self._load_local()

            print(f"  Loaded {len(df):,} bars from server")
            self._bars = df
            self._bars_with_indicators = df
            return df

        except Exception as e:
            print(f"Server error: {e}")
            print("Falling back to local cache...")
            return self._load_local()

    def _load_local(self) -> pd.DataFrame:
        """Load from local parquet cache, computing indicators locally."""
        try:
            df = self.client.load_local()
        except Exception:
            # Try downloading fresh
            path = self.client.download_parquet()
            df = self.client.load_local(path)

        # Filter to RTH if needed
        if self.session.lower() == "rth" and "timestamp" in df.columns:
            t = df["timestamp"].dt.time
            df = df[
                (t >= pd.Timestamp("09:30:00").time()) &
                (t <= pd.Timestamp("16:00:00").time())
            ].reset_index(drop=True)

        # Compute indicators locally
        df = self._add_indicators(df)
        self._bars = df
        self._bars_with_indicators = df
        print(f"  Loaded {len(df):,} bars from local cache")
        return df

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute SMA and Keltner Channel indicators locally."""
        if "sma" not in df.columns and "close" in df.columns:
            df["sma"] = compute_sma(df["close"], period=self.sma_period)
            kc = compute_keltner_channel(
                df,
                ema_period=self.kc_ema,
                atr_period=self.kc_atr,
                atr_multiplier=self.kc_mult,
            )
            df = pd.concat([df, kc], axis=1)

            df["above_sma"] = df["close"] > df["sma"]
            df["below_sma"] = df["close"] < df["sma"]
            df["above_kc_upper"] = df["close"] > df["kc_upper"]
            df["below_kc_lower"] = df["close"] < df["kc_lower"]
            df["inside_kc"] = (df["close"] <= df["kc_upper"]) & (df["close"] >= df["kc_lower"])
            df["sma_distance"] = df["close"] - df["sma"]
            df["sma_distance_pct"] = df["sma_distance"] / df["sma"] * 100

        if "bar_range" not in df.columns and "high" in df.columns:
            df["bar_range"] = df["high"] - df["low"]
            df["bar_body"] = df["close"] - df["open"]
            df["is_green"] = df["close"] > df["open"]
            df["is_red"] = df["close"] < df["open"]

        return df

    # ---- Strategy Validation ----

    def validate_strategy(
        self,
        strategy_name: str = "sma_keltner_bounce",
        config: TradeConfig = None,
        start_date: str = None,
        end_date: str = None,
        walk_forward: bool = True,
    ) -> dict:
        """
        Full validation of a single strategy.

        Runs backtest, walk-forward analysis, and generates reports.
        Returns performance summary dict.
        """
        bars = self.load_bars(start_date=start_date, end_date=end_date)
        strat = get_strategy(strategy_name)

        if config is None:
            config = TradeConfig()

        print(f"\n{'='*60}")
        print(f"  Validating: {strat['label']}")
        print(f"  Type: {strat['type']}")
        print(f"  Bars: {len(bars):,}")
        print(f"{'='*60}")

        # Full backtest
        trades = backtest(bars, strat["fn"], config=config, signal_name=strategy_name)
        trades_df = trades_to_dataframe(trades)
        summary = performance_summary(trades_df)

        print(print_summary(summary, name=strat["label"]))

        # Generate charts
        if not trades_df.empty:
            p1 = plot_equity_curve(
                trades_df,
                title=f"{strat['label']} — Equity Curve",
                output_path=str(self.output_dir / f"{strategy_name}_equity.png"),
            )
            p2 = plot_trade_distribution(
                trades_df,
                title=f"{strat['label']} — P&L Distribution",
                output_path=str(self.output_dir / f"{strategy_name}_distribution.png"),
            )
            p3 = plot_time_of_day_heatmap(
                trades_df,
                title=f"{strat['label']} — Time of Day",
                output_path=str(self.output_dir / f"{strategy_name}_heatmap.png"),
            )
            print(f"Charts: {p1}, {p2}, {p3}")

            trades_df.to_csv(self.output_dir / f"{strategy_name}_trades.csv", index=False)

        # Walk-forward validation
        if walk_forward and len(bars) > 500:
            print("\n--- Walk-Forward Validation ---")
            wf_results = self._walk_forward(bars, strat["fn"], config, strategy_name)
            summary["walk_forward"] = wf_results

        return summary

    def _walk_forward(self, bars, signal_fn, config, name, n_splits=5) -> list[dict]:
        """Run walk-forward analysis and print results."""
        splits = walk_forward_split(bars, n_splits=n_splits)
        results = []

        for k, (train, test) in enumerate(splits):
            train_trades = backtest(train, signal_fn, config=config, signal_name=name)
            test_trades = backtest(test, signal_fn, config=config, signal_name=name)
            train_s = performance_summary(trades_to_dataframe(train_trades))
            test_s = performance_summary(trades_to_dataframe(test_trades))

            fold = {
                "fold": k + 1,
                "train_trades": train_s.get("total_trades", 0),
                "train_wr": train_s.get("win_rate", 0),
                "train_pf": train_s.get("profit_factor", 0),
                "train_pnl": train_s.get("total_pnl", 0),
                "test_trades": test_s.get("total_trades", 0),
                "test_wr": test_s.get("win_rate", 0),
                "test_pf": test_s.get("profit_factor", 0),
                "test_pnl": test_s.get("total_pnl", 0),
            }
            results.append(fold)

            print(f"  Fold {k+1}: Train WR={fold['train_wr']}% PF={fold['train_pf']} "
                  f"| Test WR={fold['test_wr']}% PF={fold['test_pf']} P&L=${fold['test_pnl']:,.2f}")

        # Overall walk-forward efficiency
        train_total = sum(r["train_pnl"] for r in results)
        test_total = sum(r["test_pnl"] for r in results)
        wfe = (test_total / train_total * 100) if train_total != 0 else 0
        print(f"\n  Walk-Forward Efficiency: {wfe:.1f}% (test P&L / train P&L)")
        print(f"  Combined OOS P&L: ${test_total:,.2f}")

        return results

    # ---- Strategy Comparison ----

    def compare_all_strategies(
        self,
        config: TradeConfig = None,
        start_date: str = None,
        end_date: str = None,
    ) -> dict:
        """Run all strategies on the same data and compare results."""
        bars = self.load_bars(start_date=start_date, end_date=end_date)

        if config is None:
            config = TradeConfig()

        results = {}
        all_trades = {}

        print(f"\nComparing all strategies on {len(bars):,} bars...")
        print("-" * 60)

        for name, strat in STRATEGY_REGISTRY.items():
            trades = backtest(bars, strat["fn"], config=config, signal_name=name)
            trades_df = trades_to_dataframe(trades)
            summary = performance_summary(trades_df)
            results[strat["label"]] = summary
            all_trades[name] = trades_df

            if not trades_df.empty:
                plot_equity_curve(
                    trades_df,
                    title=strat["label"],
                    output_path=str(self.output_dir / f"{name}_equity.png"),
                )

        print("\n" + compare_strategies(results))

        # Save comparison
        with open(self.output_dir / "strategy_comparison.json", "w") as f:
            # Strip non-serializable parts
            clean = {}
            for k, v in results.items():
                clean[k] = {kk: vv for kk, vv in v.items() if kk != "exit_breakdown"}
            json.dump(clean, f, indent=2)

        return results

    # ---- Pattern Discovery ----

    def discover_patterns(
        self,
        forward_bars: int = 5,
        start_date: str = None,
        end_date: str = None,
    ) -> dict:
        """
        Discover statistically significant bar patterns in the data.

        Returns dict of pattern_name -> statistical results.
        """
        bars = self.load_bars(start_date=start_date, end_date=end_date)

        print(f"\n--- Pattern Discovery ({len(bars):,} bars, forward={forward_bars}) ---\n")

        patterns = detect_bar_patterns(bars)
        results = pattern_outcome_analysis(bars, patterns, forward_bars=forward_bars)

        significant = {}
        for pattern, stats in sorted(results.items(), key=lambda x: x[1]["p_value"]):
            sig = " ***" if stats["significant"] else ""
            print(
                f"  {pattern:<25} n={stats['count']:<5} "
                f"avg_move={stats['avg_move']:>+6.2f}  "
                f"win_rate={stats['win_rate']:.1%}  "
                f"p={stats['p_value']:.4f}{sig}"
            )
            if stats["significant"]:
                significant[pattern] = stats

        with open(self.output_dir / "pattern_discovery.json", "w") as f:
            json.dump(results, f, indent=2)

        if significant:
            print(f"\n  {len(significant)} statistically significant patterns found (p < 0.05)")
        else:
            print("\n  No patterns reached statistical significance at p < 0.05")

        return results

    # ---- Time of Day Analysis ----

    def time_analysis(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """Analyze which times of day have the best trading characteristics."""
        bars = self.load_bars(start_date=start_date, end_date=end_date)

        print(f"\n--- Time of Day Analysis ({len(bars):,} bars) ---\n")
        tod = time_of_day_analysis(bars)
        print(tod.to_string(index=False))

        tod.to_csv(self.output_dir / "time_of_day.csv", index=False)

        # Highlight best/worst times
        best = tod.loc[tod["avg_range"].idxmax()]
        calmest = tod.loc[tod["avg_range"].idxmin()]
        most_bull = tod.loc[tod["directional_bias"].idxmax()]
        most_bear = tod.loc[tod["directional_bias"].idxmin()]

        print(f"\n  Most volatile:    {best['hour_min']} (avg range: {best['avg_range']:.2f})")
        print(f"  Calmest:          {calmest['hour_min']} (avg range: {calmest['avg_range']:.2f})")
        print(f"  Most bullish:     {most_bull['hour_min']} (bias: {most_bull['directional_bias']:.3f})")
        print(f"  Most bearish:     {most_bear['hour_min']} (bias: {most_bear['directional_bias']:.3f})")

        return tod

    # ---- Volatility Regime ----

    def regime_analysis(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """Analyze how strategies perform in different volatility regimes."""
        bars = self.load_bars(start_date=start_date, end_date=end_date)
        bars = bars.copy()
        bars["vol_regime"] = volatility_regime(bars)

        print(f"\n--- Volatility Regime Analysis ---\n")
        counts = bars["vol_regime"].value_counts()
        for regime in ["low", "normal", "high"]:
            print(f"  {regime:<8} {counts.get(regime, 0):,} bars ({counts.get(regime, 0)/len(bars)*100:.1f}%)")

        # Backtest each strategy per regime
        config = TradeConfig()
        print(f"\n  Strategy performance by regime:")
        print(f"  {'Strategy':<25} {'Low Vol':>12} {'Normal':>12} {'High Vol':>12}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")

        for name, strat in STRATEGY_REGISTRY.items():
            regime_pnl = {}
            for regime in ["low", "normal", "high"]:
                regime_bars = bars[bars["vol_regime"] == regime].reset_index(drop=True)
                if len(regime_bars) < 100:
                    regime_pnl[regime] = "N/A"
                    continue
                trades = backtest(regime_bars, strat["fn"], config=config, signal_name=name)
                trades_df = trades_to_dataframe(trades)
                s = performance_summary(trades_df)
                wr = s.get("win_rate", 0)
                regime_pnl[regime] = f"WR={wr}%"

            print(f"  {strat['label']:<25} {regime_pnl.get('low', 'N/A'):>12} "
                  f"{regime_pnl.get('normal', 'N/A'):>12} {regime_pnl.get('high', 'N/A'):>12}")

        return bars

    # ---- Opportunity Scanner ----

    def scan(
        self,
        min_confluence: float = 3.0,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """Scan for high-confluence setups in the data."""
        bars = self.load_bars(start_date=start_date, end_date=end_date)

        print(f"\nScanning {len(bars):,} bars (min confluence: {min_confluence})...")
        opps = scan_opportunities(bars, min_confluence=min_confluence)
        print(f"  Found {len(opps)} high-confluence setups")

        if not opps.empty:
            print(f"\nTop 20:")
            top = opps.head(20)[["timestamp", "close", "direction", "confluence_score"]]
            print(top.to_string(index=False))
            opps.to_csv(self.output_dir / "opportunities.csv", index=False)

        return opps

    # ---- Full Report ----

    def full_report(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> dict:
        """
        Run the complete analysis suite and generate all reports.

        This is the one-command way to validate everything.
        """
        print("=" * 60)
        print("  ES SCALPING RESEARCH REPORT")
        print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 60)

        report = {}

        # 1. Data summary
        bars = self.load_bars(start_date=start_date, end_date=end_date)
        report["data"] = {
            "bars": len(bars),
            "start": str(bars["timestamp"].min()) if "timestamp" in bars.columns else "N/A",
            "end": str(bars["timestamp"].max()) if "timestamp" in bars.columns else "N/A",
        }

        # 2. Strategy comparison
        report["strategies"] = self.compare_all_strategies(start_date=start_date, end_date=end_date)

        # 3. Your core strategy deep dive
        report["core_strategy"] = self.validate_strategy(
            "sma_keltner_bounce", start_date=start_date, end_date=end_date
        )

        # 4. Pattern discovery
        report["patterns"] = self.discover_patterns(start_date=start_date, end_date=end_date)

        # 5. Time analysis
        self.time_analysis(start_date=start_date, end_date=end_date)

        # 6. Regime analysis
        self.regime_analysis(start_date=start_date, end_date=end_date)

        print(f"\n{'='*60}")
        print(f"  All reports saved to: {self.output_dir}/")
        print(f"{'='*60}")

        return report
