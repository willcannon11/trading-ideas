"""
Generate realistic synthetic ES 610-tick bar data for backtesting.

Produces ~250 trading days of ES futures data with:
- Realistic price distributions (mean-reverting intraday with trends)
- Opening range behavior (9:30-9:45 consolidation then breakout)
- Volume patterns (U-shaped intraday)
- Proper 610-tick bar timing
- RTH session (9:30-16:00 ET)

Output: es_data.parquet with columns StartTime, EndTime, Open, High, Low, Close, Volume
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ES futures parameters
TICK_SIZE = 0.25
BARS_PER_DAY_APPROX = 180  # ~180 610-tick bars per RTH session
TRADING_DAYS = 250


def generate_trading_calendar(n_days: int, start_date: str = "2024-06-01") -> list:
    """Generate list of trading days (weekdays only, skip major holidays)."""
    holidays = {
        datetime(2024, 7, 4).date(), datetime(2024, 9, 2).date(),
        datetime(2024, 11, 28).date(), datetime(2024, 12, 25).date(),
        datetime(2025, 1, 1).date(), datetime(2025, 1, 20).date(),
        datetime(2025, 2, 17).date(),
    }
    days = []
    current = pd.to_datetime(start_date).date()
    while len(days) < n_days:
        if current.weekday() < 5 and current not in holidays:
            days.append(current)
        current += timedelta(days=1)
    return days


def intraday_volume_profile(n_bars: int) -> np.ndarray:
    """U-shaped volume profile: high at open/close, lower midday."""
    x = np.linspace(0, 1, n_bars)
    # U-shape: higher at edges
    profile = 2.0 - 1.5 * np.sin(np.pi * x)
    # Extra spike at open
    profile[:int(n_bars * 0.08)] *= 1.5
    # Extra spike at close
    profile[-int(n_bars * 0.05):] *= 1.3
    profile = profile / profile.sum()
    return profile


def generate_day_bars(date, base_price: float, daily_trend: float,
                      daily_volatility: float) -> tuple:
    """Generate 610-tick bars for one trading day."""
    n_bars = np.random.randint(160, 200)  # vary bar count

    # Volume distribution
    vol_profile = intraday_volume_profile(n_bars)
    total_volume = np.random.randint(800000, 2000000)
    volumes = (vol_profile * total_volume).astype(int)
    volumes = np.maximum(volumes, 610)

    # Time distribution (9:30 to 16:00 = 390 minutes)
    session_start = datetime.combine(date, time(9, 30))
    session_end = datetime.combine(date, time(16, 0))
    total_seconds = (session_end - session_start).total_seconds()

    # Bars are not evenly spaced in time (610-tick bars depend on volume)
    cumulative_vol = np.cumsum(vol_profile)
    bar_times = [session_start + timedelta(seconds=total_seconds * cv)
                 for cv in cumulative_vol]

    # Price generation with realistic microstructure
    # Opening range: first ~15 bars (9:30-9:45) - tighter range
    or_bars = int(n_bars * 0.08)  # ~15 bars in opening range
    post_or_bars = n_bars - or_bars

    # Opening range volatility
    or_volatility = daily_volatility * 0.6
    or_mid = base_price + np.random.normal(0, 1.0)

    # Generate OR price path (mean-reverting around or_mid)
    or_returns = np.random.normal(0, or_volatility, or_bars)
    or_returns[0] = np.random.normal(0, or_volatility * 2)  # bigger opening move

    or_prices = np.zeros(or_bars)
    or_prices[0] = or_mid + or_returns[0]
    for i in range(1, or_bars):
        mean_rev = -0.3 * (or_prices[i-1] - or_mid)
        or_prices[i] = or_prices[i-1] + or_returns[i] + mean_rev

    # Opening range high/low
    or_high = max(or_prices) + np.random.uniform(0.25, 1.5)
    or_low = min(or_prices) - np.random.uniform(0.25, 1.5)
    or_range_size = or_high - or_low

    # Post-OR price path with breakout behavior
    post_returns = np.random.normal(daily_trend / post_or_bars,
                                     daily_volatility, post_or_bars)

    # Add breakout dynamics:
    # - First 20-30 bars after OR: potential breakout move
    # - Middle: trend or chop
    # - Last 30 bars: potential reversal or continuation
    breakout_dir = np.sign(daily_trend) if abs(daily_trend) > 1 else np.random.choice([-1, 1])
    breakout_strength = np.random.exponential(1.5)

    # Breakout phase (first 15% of post-OR)
    breakout_bars = int(post_or_bars * 0.15)
    post_returns[:breakout_bars] += breakout_dir * breakout_strength * daily_volatility * 0.3

    # Pullback phase (next 10%)
    pullback_start = breakout_bars
    pullback_end = pullback_start + int(post_or_bars * 0.10)
    post_returns[pullback_start:pullback_end] -= breakout_dir * breakout_strength * daily_volatility * 0.2

    # Trending phase with occasional mean reversion
    trend_component = daily_trend / post_or_bars
    for i in range(pullback_end, post_or_bars):
        # Add some autocorrelation
        if i > 0:
            post_returns[i] += 0.1 * post_returns[i-1]

    post_prices = np.zeros(post_or_bars)
    post_prices[0] = or_prices[-1]
    for i in range(1, post_or_bars):
        post_prices[i] = post_prices[i-1] + post_returns[i]

    # Combine OR and post-OR
    all_prices = np.concatenate([or_prices, post_prices])

    # Generate OHLC from price path
    bars = []
    for i in range(n_bars):
        close = round(all_prices[i] / TICK_SIZE) * TICK_SIZE

        # Realistic OHLC: open near previous close, H/L bracket the move
        if i == 0:
            open_price = round((base_price + np.random.normal(0, 0.5)) / TICK_SIZE) * TICK_SIZE
        else:
            open_price = bars[-1]["Close"]

        # Bar range depends on volume and volatility
        bar_range = abs(np.random.normal(0, daily_volatility * 1.5)) + 0.25
        bar_range = round(bar_range / TICK_SIZE) * TICK_SIZE
        bar_range = max(bar_range, 0.5)

        if close >= open_price:
            high = max(close, open_price) + round(np.random.exponential(0.3) / TICK_SIZE) * TICK_SIZE
            low = min(close, open_price) - round(np.random.exponential(0.3) / TICK_SIZE) * TICK_SIZE
        else:
            high = max(close, open_price) + round(np.random.exponential(0.3) / TICK_SIZE) * TICK_SIZE
            low = min(close, open_price) - round(np.random.exponential(0.3) / TICK_SIZE) * TICK_SIZE

        high = max(high, max(open_price, close) + 0.25)
        low = min(low, min(open_price, close) - 0.25)

        start_time = bar_times[i] if i < len(bar_times) else bar_times[-1] + timedelta(seconds=30*i)
        end_time = start_time + timedelta(seconds=np.random.randint(5, 120))

        bars.append({
            "StartTime": start_time,
            "EndTime": end_time,
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": int(volumes[i]) if i < len(volumes) else 610,
        })

    final_close = bars[-1]["Close"]
    return bars, final_close


def generate_dataset(n_days: int = TRADING_DAYS) -> pd.DataFrame:
    """Generate full synthetic ES dataset."""
    print(f"Generating {n_days} days of synthetic ES 610-tick bar data...")

    trading_days = generate_trading_calendar(n_days)
    base_price = 5400.0  # Starting ES price

    all_bars = []
    for i, date in enumerate(trading_days):
        # Daily trend: slight upward bias with large variance
        daily_trend = np.random.normal(0.5, 8.0)  # avg +0.5pt/day, high variance

        # Daily volatility varies (VIX-like regime changes)
        if np.random.random() < 0.15:  # 15% high-vol days
            daily_volatility = np.random.uniform(1.2, 2.5)
        elif np.random.random() < 0.30:  # 30% low-vol days
            daily_volatility = np.random.uniform(0.3, 0.7)
        else:
            daily_volatility = np.random.uniform(0.5, 1.2)

        day_bars, close = generate_day_bars(date, base_price, daily_trend, daily_volatility)
        all_bars.extend(day_bars)

        # Next day opens near this close
        base_price = close + np.random.normal(0, 3.0)  # overnight gap

        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{n_days} days ({len(all_bars):,} bars so far)")

    df = pd.DataFrame(all_bars)
    print(f"  Total: {len(df):,} bars across {n_days} days")
    print(f"  Date range: {trading_days[0]} to {trading_days[-1]}")
    print(f"  Price range: {df['Low'].min():.2f} to {df['High'].max():.2f}")

    return df


if __name__ == "__main__":
    df = generate_dataset()
    output_path = "es_data.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path} ({df.memory_usage(deep=True).sum() / 1e6:.1f} MB)")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample data:")
    print(df.head(5).to_string())
