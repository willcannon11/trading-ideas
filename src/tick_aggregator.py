"""
Aggregate raw tick data into N-tick bars (e.g. 610-tick charts)
and compute technical indicators: SMA, Keltner Channel, ATR.
"""
import numpy as np
import pandas as pd


def aggregate_tick_bars(ticks: pd.DataFrame, bar_size: int = 610) -> pd.DataFrame:
    """
    Aggregate raw ticks into N-tick OHLCV bars.

    Parameters
    ----------
    ticks : DataFrame with columns [timestamp, price, volume]
    bar_size : Number of ticks per bar (default 610)

    Returns
    -------
    DataFrame with columns:
        bar_num, timestamp (bar close time), open, high, low, close,
        volume, tick_count, bar_open_time
    """
    n = len(ticks)
    num_bars = n // bar_size

    if num_bars == 0:
        return pd.DataFrame()

    # Trim to exact multiple of bar_size
    trimmed = ticks.iloc[: num_bars * bar_size].copy()
    trimmed["bar_group"] = np.arange(len(trimmed)) // bar_size

    bars = trimmed.groupby("bar_group").agg(
        bar_open_time=("timestamp", "first"),
        timestamp=("timestamp", "last"),
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("volume", "sum"),
        tick_count=("price", "count"),
    ).reset_index(drop=True)

    bars.index.name = "bar_num"
    return bars


def compute_sma(series: pd.Series, period: int = 80) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def compute_atr(bars: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Average True Range over N bars.

    Uses the standard TR = max(H-L, |H-prevC|, |L-prevC|) formula.
    """
    high = bars["high"]
    low = bars["low"]
    prev_close = bars["close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


def compute_keltner_channel(
    bars: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 10,
    atr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Keltner Channel: EMA midline with ATR-based upper/lower bands.

    Parameters
    ----------
    bars : DataFrame with OHLC columns
    ema_period : EMA lookback for the midline (default 20)
    atr_period : ATR lookback (default 10)
    atr_multiplier : Band width in ATR units (default 1.5)

    Returns
    -------
    DataFrame with columns: kc_mid, kc_upper, kc_lower
    """
    mid = bars["close"].ewm(span=ema_period, adjust=False).mean()
    atr = compute_atr(bars, period=atr_period)

    return pd.DataFrame({
        "kc_mid": mid,
        "kc_upper": mid + atr_multiplier * atr,
        "kc_lower": mid - atr_multiplier * atr,
    }, index=bars.index)


def build_chart(
    ticks: pd.DataFrame,
    bar_size: int = 610,
    sma_period: int = 80,
    kc_ema_period: int = 20,
    kc_atr_period: int = 10,
    kc_atr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Full pipeline: ticks -> tick bars -> indicators.

    Returns a single DataFrame with OHLCV bars plus all indicator columns.
    """
    bars = aggregate_tick_bars(ticks, bar_size=bar_size)
    if bars.empty:
        return bars

    # SMA on close
    bars["sma"] = compute_sma(bars["close"], period=sma_period)

    # Keltner Channel
    kc = compute_keltner_channel(
        bars,
        ema_period=kc_ema_period,
        atr_period=kc_atr_period,
        atr_multiplier=kc_atr_multiplier,
    )
    bars = pd.concat([bars, kc], axis=1)

    # Derived signals used by strategies
    bars["above_sma"] = bars["close"] > bars["sma"]
    bars["below_sma"] = bars["close"] < bars["sma"]
    bars["above_kc_upper"] = bars["close"] > bars["kc_upper"]
    bars["below_kc_lower"] = bars["close"] < bars["kc_lower"]
    bars["inside_kc"] = (bars["close"] <= bars["kc_upper"]) & (bars["close"] >= bars["kc_lower"])

    # Price relative to SMA (for trend strength)
    bars["sma_distance"] = bars["close"] - bars["sma"]
    bars["sma_distance_pct"] = bars["sma_distance"] / bars["sma"] * 100

    # Bar direction / momentum
    bars["bar_range"] = bars["high"] - bars["low"]
    bars["bar_body"] = bars["close"] - bars["open"]
    bars["is_green"] = bars["close"] > bars["open"]
    bars["is_red"] = bars["close"] < bars["open"]

    return bars
