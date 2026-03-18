"""
Trading strategies for ES scalping on 610-tick charts.

Each strategy is a signal function: f(bars, i) -> "long" | "short" | None

Strategy catalog:
  1. SMA + Keltner Bounce (your core setup)
  2. Keltner Breakout / Squeeze Release
  3. SMA Crossover Momentum
  4. Mean Reversion at Keltner Extremes
  5. Trend Continuation Pullback
"""
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: ensure we have enough history and indicators are valid
# ---------------------------------------------------------------------------

def _ready(bars: pd.DataFrame, i: int, min_bars: int = 80) -> bool:
    """Check if we have enough bars and indicators are computed."""
    if i < min_bars:
        return False
    row = bars.iloc[i]
    return not (pd.isna(row.get("sma")) or pd.isna(row.get("kc_upper")))


# ---------------------------------------------------------------------------
# Strategy 1: SMA + Keltner Bounce (Core Setup)
# ---------------------------------------------------------------------------
# Long: price pulls back to/below SMA while above KC lower band,
#        then the bar closes above SMA (bounce confirmed).
#        Trend filter: SMA is rising over last N bars.
# Short: mirror image.

def sma_keltner_bounce(bars: pd.DataFrame, i: int) -> Optional[str]:
    """
    Your core setup: price bounces off the 80 SMA while staying within
    the Keltner Channel. This catches trend pullback entries.
    """
    if not _ready(bars, i):
        return None

    curr = bars.iloc[i]
    prev = bars.iloc[i - 1]

    sma_now = curr["sma"]
    sma_prev = bars.iloc[i - 5]["sma"] if i >= 5 else sma_now

    # Trend filter: SMA slope
    sma_rising = sma_now > sma_prev
    sma_falling = sma_now < sma_prev

    # Long: pullback touched/crossed below SMA, bar closes back above it
    if sma_rising:
        pulled_back = prev["low"] <= prev["sma"] or prev["close"] <= prev["sma"]
        bounced = curr["close"] > curr["sma"]
        inside_channel = curr["close"] < curr["kc_upper"]
        if pulled_back and bounced and inside_channel:
            return "long"

    # Short: rally touched/crossed above SMA, bar closes back below it
    if sma_falling:
        pulled_back = prev["high"] >= prev["sma"] or prev["close"] >= prev["sma"]
        bounced = curr["close"] < curr["sma"]
        inside_channel = curr["close"] > curr["kc_lower"]
        if pulled_back and bounced and inside_channel:
            return "short"

    return None


# ---------------------------------------------------------------------------
# Strategy 2: Keltner Channel Breakout
# ---------------------------------------------------------------------------
# Long: close breaks above upper KC band with volume confirmation.
# Short: close breaks below lower KC band with volume confirmation.

def keltner_breakout(bars: pd.DataFrame, i: int) -> Optional[str]:
    """
    Breakout entry when price pushes through KC bands with conviction.
    Volume must be above recent average for confirmation.
    """
    if not _ready(bars, i):
        return None

    curr = bars.iloc[i]
    prev = bars.iloc[i - 1]

    # Volume filter: current bar volume > 1.2x average of last 20 bars
    lookback = min(i, 20)
    avg_vol = bars.iloc[i - lookback:i]["volume"].mean()
    vol_confirm = curr["volume"] > avg_vol * 1.2

    if not vol_confirm:
        return None

    # Breakout long: prev inside channel, current closes above upper
    if prev["close"] <= prev["kc_upper"] and curr["close"] > curr["kc_upper"]:
        if curr["is_green"]:
            return "long"

    # Breakout short: prev inside channel, current closes below lower
    if prev["close"] >= prev["kc_lower"] and curr["close"] < curr["kc_lower"]:
        if curr["is_red"]:
            return "short"

    return None


# ---------------------------------------------------------------------------
# Strategy 3: Keltner Mean Reversion
# ---------------------------------------------------------------------------
# Long: price closes below lower KC band (overextended), then closes
#        back inside on next bar.
# Short: mirror.

def keltner_mean_reversion(bars: pd.DataFrame, i: int) -> Optional[str]:
    """
    Fade the overextension: enter when price snaps back inside KC bands
    after closing outside them. Targets the KC midline.
    """
    if not _ready(bars, i, min_bars=81):
        return None
    if i < 2:
        return None

    curr = bars.iloc[i]
    prev = bars.iloc[i - 1]
    prev2 = bars.iloc[i - 2]

    # Long: prev2 was below lower KC, prev still below, curr closes back inside
    if (prev2["close"] < prev2["kc_lower"]
            and prev["close"] < prev["kc_lower"]
            and curr["close"] > curr["kc_lower"]
            and curr["is_green"]):
        return "long"

    # Short: prev2 was above upper KC, prev still above, curr closes back inside
    if (prev2["close"] > prev2["kc_upper"]
            and prev["close"] > prev["kc_upper"]
            and curr["close"] < curr["kc_upper"]
            and curr["is_red"]):
        return "short"

    return None


# ---------------------------------------------------------------------------
# Strategy 4: Trend Continuation (SMA + Momentum)
# ---------------------------------------------------------------------------
# Long: strong uptrend (price well above SMA), pullback creates 2-3 red
#        bars, then a green bar confirms continuation.
# Short: mirror.

def trend_continuation(bars: pd.DataFrame, i: int) -> Optional[str]:
    """
    Trend continuation after a brief pause: 2-3 counter-trend bars
    followed by a resumption bar in the trend direction.
    """
    if not _ready(bars, i):
        return None
    if i < 5:
        return None

    curr = bars.iloc[i]

    # Count consecutive red/green bars before current
    red_count = 0
    green_count = 0
    for j in range(i - 1, max(i - 5, -1), -1):
        b = bars.iloc[j]
        if b["is_red"]:
            red_count += 1
        else:
            break
    for j in range(i - 1, max(i - 5, -1), -1):
        b = bars.iloc[j]
        if b["is_green"]:
            green_count += 1
        else:
            break

    # Long: uptrend + pullback (2-3 red bars) + green resumption
    if (curr["above_sma"]
            and curr["sma_distance_pct"] > 0.02
            and 2 <= red_count <= 3
            and curr["is_green"]
            and curr["close"] > bars.iloc[i - 1]["high"]):
        return "long"

    # Short: downtrend + rally (2-3 green bars) + red resumption
    if (curr["below_sma"]
            and curr["sma_distance_pct"] < -0.02
            and 2 <= green_count <= 3
            and curr["is_red"]
            and curr["close"] < bars.iloc[i - 1]["low"]):
        return "short"

    return None


# ---------------------------------------------------------------------------
# Strategy 5: SMA Slope + KC Squeeze
# ---------------------------------------------------------------------------
# When KC bands narrow (squeeze), wait for expansion with SMA slope
# confirmation for direction.

def kc_squeeze_breakout(bars: pd.DataFrame, i: int) -> Optional[str]:
    """
    Keltner Channel squeeze: bands contract then expand. Enter in the
    direction of the SMA slope when the squeeze releases.
    """
    if not _ready(bars, i):
        return None
    if i < 20:
        return None

    curr = bars.iloc[i]

    # Measure KC bandwidth
    bw_now = curr["kc_upper"] - curr["kc_lower"]
    bw_lookback = bars.iloc[i - 20:i][["kc_upper", "kc_lower"]].dropna()
    if len(bw_lookback) < 10:
        return None

    bw_series = bw_lookback["kc_upper"] - bw_lookback["kc_lower"]
    bw_min = bw_series.min()
    bw_mean = bw_series.mean()

    # Squeeze: recent bandwidth was near its low, now expanding
    was_squeezed = bw_series.iloc[-5:].min() <= bw_min * 1.1
    expanding = bw_now > bw_mean * 0.9

    if not (was_squeezed and expanding):
        return None

    # Direction from SMA slope
    sma_slope = curr["sma"] - bars.iloc[i - 10]["sma"]

    if sma_slope > 0 and curr["close"] > curr["kc_mid"] and curr["is_green"]:
        return "long"
    elif sma_slope < 0 and curr["close"] < curr["kc_mid"] and curr["is_red"]:
        return "short"

    return None


# ---------------------------------------------------------------------------
# Registry: all strategies
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY = {
    "sma_keltner_bounce": {
        "fn": sma_keltner_bounce,
        "label": "SMA + Keltner Bounce",
        "description": "Core setup: pullback to 80 SMA bounces within Keltner Channel",
        "type": "trend_pullback",
    },
    "keltner_breakout": {
        "fn": keltner_breakout,
        "label": "Keltner Breakout",
        "description": "Volume-confirmed breakout through Keltner Channel bands",
        "type": "breakout",
    },
    "keltner_mean_reversion": {
        "fn": keltner_mean_reversion,
        "label": "Keltner Mean Reversion",
        "description": "Fade overextension outside KC bands, target midline",
        "type": "mean_reversion",
    },
    "trend_continuation": {
        "fn": trend_continuation,
        "label": "Trend Continuation",
        "description": "Enter after 2-3 bar pullback in strong trend",
        "type": "trend_continuation",
    },
    "kc_squeeze_breakout": {
        "fn": kc_squeeze_breakout,
        "label": "KC Squeeze Breakout",
        "description": "Enter when Keltner Channel squeeze releases with SMA slope",
        "type": "squeeze",
    },
}


def list_strategies() -> list[str]:
    """Return names of all registered strategies."""
    return list(STRATEGY_REGISTRY.keys())


def get_strategy(name: str) -> dict:
    """Get strategy info by name."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list_strategies()}")
    return STRATEGY_REGISTRY[name]
