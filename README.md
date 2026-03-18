# ES Scalping Strategy Validation System

Backtest, validate, and discover trading ideas for ES futures day trading using 610-tick charts with 80 SMA and Keltner Channel indicators.

## Setup

```bash
pip install -r requirements.txt
```

## Exporting Tick Data from NinjaTrader

1. **Tools > Historical Data** in NinjaTrader
2. Select **ES** instrument, **Tick** data type
3. Select date range (1 year recommended)
4. Export as CSV/TXT

Supported formats: semicolon-separated (`;`), comma-separated with datetime or separate date/time columns, and OHLCV tick exports.

## Usage

### Backtest your core strategy (SMA + Keltner Bounce)

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
src/
  data_loader.py      # NinjaTrader tick data import
  tick_aggregator.py   # 610-tick bar construction + indicators
  backtester.py        # Trade simulation engine
  strategies.py        # Signal functions (5 built-in)
  scanner.py           # Pattern discovery & opportunity scanner
  reports.py           # Performance stats & charts
  cli.py               # Command-line interface
```
