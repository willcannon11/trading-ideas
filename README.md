# ES Scalping Strategy Validation System

Backtest, validate, and discover trading ideas for ES futures day trading using 610-tick charts with 80 SMA and Keltner Channel indicators.

## Architecture

```
Mac (this repo)                    Windows VM (Parallels)
┌─────────────────────┐            ┌──────────────────────────┐
│  run_research.py    │            │  C:\backtest\server.py   │
│  src/               │   HTTP     │  (FastAPI on port 8000)  │
│    research_agent.py│◄──────────►│                          │
│    api_client.py    │            │  ES_610tick_ALL.parquet   │
│    backtester.py    │            │  (1 year tick data)       │
│    strategies.py    │            └──────────────────────────┘
│    scanner.py       │
│    reports.py       │
│  output/            │
└─────────────────────┘
```

## Setup

**Mac side (this repo):**
```bash
pip install -r requirements.txt
```

**Windows side — upgrade your server:**
1. Copy `server_v2.py` to `C:\backtest\server.py` (replaces the existing one)
2. Restart: `uvicorn server:app --host 0.0.0.0 --port 8000`

The upgraded server adds endpoints to return actual bar data, indicators, and parquet downloads — the original `/run-research` endpoint still works.

## Quick Start — Research Agent

The research agent connects to your Windows server, pulls data, and runs all analysis locally.

```bash
# Full report (all strategies, patterns, time analysis, regimes)
python run_research.py

# Validate your core setup
python run_research.py --validate

# Compare all 5 strategies head-to-head
python run_research.py --compare

# Discover statistically significant patterns
python run_research.py --discover

# Scan for high-confluence setups
python run_research.py --scan

# Time of day analysis
python run_research.py --time

# Volatility regime analysis
python run_research.py --regime

# Filter by date range
python run_research.py --validate --start 2025-06-01 --end 2025-09-30

# Custom stop/target
python run_research.py --compare --stop 1.5 --target 3.0
```

Or use it from Python:

```python
from src.research_agent import ResearchAgent

agent = ResearchAgent()
agent.connect()                              # test server connection
agent.full_report()                          # everything
agent.validate_strategy("sma_keltner_bounce") # deep dive on one strategy
agent.compare_all_strategies()               # head-to-head comparison
agent.discover_patterns()                    # find new edges
```

## CLI (works with local tick data files too)

### Backtest your core strategy

```bash
python -m src backtest --data path/to/ES_ticks.txt --strategy sma_keltner_bounce
```

### Backtest with custom parameters

```bash
python -m src backtest \
  --data ES_ticks.txt \
  --strategy sma_keltner_bounce \
  --bar-size 610 \
  --sma-period 80 \
  --kc-ema 20 \
  --kc-mult 1.5 \
  --stop-loss 2.0 \
  --take-profit 2.0 \
  --walk-forward
```

### Compare all strategies

```bash
python -m src compare --data ES_ticks.txt
```

### Scan for high-confluence setups

```bash
python -m src scan --data ES_ticks.txt --min-confluence 3.5
```

### Discover new patterns

```bash
python -m src discover --data ES_ticks.txt --forward-bars 5
```

## Strategies Included

| Strategy | Type | Description |
|----------|------|-------------|
| `sma_keltner_bounce` | Trend Pullback | **Your core setup** — price pulls back to 80 SMA and bounces within Keltner Channel |
| `keltner_breakout` | Breakout | Volume-confirmed breakout through KC bands |
| `keltner_mean_reversion` | Mean Reversion | Fade overextension outside KC bands, target midline |
| `trend_continuation` | Trend | Enter after 2-3 bar pullback in strong trend |
| `kc_squeeze_breakout` | Squeeze | Enter when KC bands contract then expand with SMA slope |

## Output

All commands save results to the `output/` directory:

- **Equity curves** — cumulative P&L with drawdown overlay
- **Trade distributions** — P&L histograms and day-of-week breakdowns
- **Time-of-day heatmaps** — when your edge is strongest
- **Trade logs** — CSV with every trade for review
- **Pattern stats** — statistically significant bar patterns with p-values

## Key Metrics

The system reports: win rate, profit factor, expectancy per trade, max drawdown, consecutive win/loss streaks, profitable days %, long vs short breakdown, and exit reason analysis (target/stop/EOD).

## Walk-Forward Validation

Use `--walk-forward` to split data into 5 sequential train/test windows. This validates that a strategy's edge holds on unseen data and isn't curve-fitted.

## Adding Custom Strategies

Add a signal function to `src/strategies.py`:

```python
def my_strategy(bars: pd.DataFrame, i: int) -> Optional[str]:
    """Return 'long', 'short', or None."""
    # Your logic using bars.iloc[i] and any indicators
    curr = bars.iloc[i]
    if curr["above_sma"] and curr["close"] > curr["kc_mid"]:
        return "long"
    return None
```

Then register it in `STRATEGY_REGISTRY` at the bottom of the file.

## Project Structure

```
run_research.py          # One-command research agent entry point
server_v2.py             # Upgraded server to deploy on Windows
src/
  research_agent.py      # Research orchestrator (connects to server)
  api_client.py          # HTTP client for Windows data server
  data_loader.py         # NinjaTrader tick data import (local files)
  tick_aggregator.py     # 610-tick bar construction + indicators
  backtester.py          # Trade simulation engine
  strategies.py          # Signal functions (5 built-in)
  scanner.py             # Pattern discovery & opportunity scanner
  reports.py             # Performance stats & charts
  cli.py                 # CLI for local file analysis
```
