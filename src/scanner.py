"""
Pattern discovery and opportunity scanner.

Scans historical tick-bar data to find statistically significant patterns
and trading opportunities beyond the predefined strategies. Uses:
  - Time-of-day analysis (when do the best setups occur?)
  - Volatility regime detection (trending vs. ranging)
  - Bar pattern statistics (inside bars, engulfing, momentum shifts)
  - Indicator confluence scoring
"""
import numpy as np
import pandas as pd
from scipy import stats


def time_of_day_analysis(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze price behavior by time of day.

    Returns a DataFrame showing average range, directional bias,
    and volatility for each 30-minute bucket.
    """
    df = bars.copy()
    df["hour_min"] = df["timestamp"].dt.floor("30min").dt.strftime("%H:%M")

    grouped = df.groupby("hour_min").agg(
        avg_range=("bar_range", "mean"),
        avg_body=("bar_body", "mean"),
        std_range=("bar_range", "std"),
        pct_green=("is_green", "mean"),
        avg_volume=("volume", "mean"),
        bar_count=("close", "count"),
    ).reset_index()

    grouped["directional_bias"] = grouped["avg_body"] / grouped["avg_range"]
    grouped["volatility_rank"] = grouped["avg_range"].rank(pct=True)

    return grouped.sort_values("hour_min")


def volatility_regime(bars: pd.DataFrame, lookback: int = 50) -> pd.Series:
    """
    Classify each bar into a volatility regime.

    Returns a Series with values: "low", "normal", "high"
    based on rolling ATR percentile.
    """
    rolling_range = bars["bar_range"].rolling(lookback, min_periods=lookback).mean()
    overall_mean = bars["bar_range"].mean()
    overall_std = bars["bar_range"].std()

    regime = pd.Series("normal", index=bars.index)
    regime[rolling_range < overall_mean - 0.5 * overall_std] = "low"
    regime[rolling_range > overall_mean + 0.5 * overall_std] = "high"

    return regime


def detect_bar_patterns(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Detect common bar patterns and return a DataFrame of pattern flags.

    Patterns detected:
      - inside_bar: H < prev H and L > prev L
      - outside_bar: H > prev H and L < prev L
      - bullish_engulf: green bar that engulfs previous red bar
      - bearish_engulf: red bar that engulfs previous green bar
      - momentum_shift: 3+ bars in one direction then reversal
      - narrow_range: bar range < 50th percentile of last 20 bars
      - wide_range: bar range > 90th percentile of last 20 bars
    """
    n = len(bars)
    patterns = pd.DataFrame(index=bars.index)

    patterns["inside_bar"] = False
    patterns["outside_bar"] = False
    patterns["bullish_engulf"] = False
    patterns["bearish_engulf"] = False
    patterns["momentum_shift_bull"] = False
    patterns["momentum_shift_bear"] = False
    patterns["narrow_range"] = False
    patterns["wide_range"] = False

    for i in range(1, n):
        curr = bars.iloc[i]
        prev = bars.iloc[i - 1]

        # Inside bar
        if curr["high"] < prev["high"] and curr["low"] > prev["low"]:
            patterns.iloc[i, patterns.columns.get_loc("inside_bar")] = True

        # Outside bar
        if curr["high"] > prev["high"] and curr["low"] < prev["low"]:
            patterns.iloc[i, patterns.columns.get_loc("outside_bar")] = True

        # Engulfing
        if curr["is_green"] and prev["is_red"]:
            if curr["close"] > prev["open"] and curr["open"] < prev["close"]:
                patterns.iloc[i, patterns.columns.get_loc("bullish_engulf")] = True
        if curr["is_red"] and prev["is_green"]:
            if curr["close"] < prev["open"] and curr["open"] > prev["close"]:
                patterns.iloc[i, patterns.columns.get_loc("bearish_engulf")] = True

        # Momentum shift
        if i >= 4:
            prev_bars = bars.iloc[i - 3:i]
            all_red = prev_bars["is_red"].all()
            all_green = prev_bars["is_green"].all()
            if all_red and curr["is_green"]:
                patterns.iloc[i, patterns.columns.get_loc("momentum_shift_bull")] = True
            if all_green and curr["is_red"]:
                patterns.iloc[i, patterns.columns.get_loc("momentum_shift_bear")] = True

        # Range classification
        if i >= 20:
            recent_ranges = bars.iloc[i - 20:i]["bar_range"]
            p50 = recent_ranges.quantile(0.5)
            p90 = recent_ranges.quantile(0.9)
            if curr["bar_range"] < p50:
                patterns.iloc[i, patterns.columns.get_loc("narrow_range")] = True
            if curr["bar_range"] > p90:
                patterns.iloc[i, patterns.columns.get_loc("wide_range")] = True

    return patterns


def pattern_outcome_analysis(
    bars: pd.DataFrame,
    patterns: pd.DataFrame,
    forward_bars: int = 5,
) -> dict:
    """
    For each detected pattern, measure what happens in the next N bars.

    Returns a dict of pattern_name -> stats dict with:
      - count: how many times the pattern occurred
      - avg_move: average price change over next N bars
      - win_rate: % of times price moved favorably (green direction for bull patterns)
      - avg_range: average range of the next N bars
      - t_stat, p_value: statistical significance of directional move
    """
    results = {}
    n = len(bars)

    for col in patterns.columns:
        mask = patterns[col]
        occurrences = bars.index[mask]

        if len(occurrences) < 5:
            continue

        moves = []
        for idx in occurrences:
            if idx + forward_bars >= n:
                continue
            future_close = bars.iloc[idx + forward_bars]["close"]
            entry_close = bars.iloc[idx]["close"]
            moves.append(future_close - entry_close)

        if len(moves) < 5:
            continue

        moves = np.array(moves)
        is_bull = "bull" in col or "inside" in col  # bias direction for win rate
        if is_bull:
            win_rate = np.mean(moves > 0)
        elif "bear" in col:
            win_rate = np.mean(moves < 0)
        else:
            win_rate = np.mean(np.abs(moves) > 0)  # any move for neutral patterns

        t_stat, p_val = stats.ttest_1samp(moves, 0)

        results[col] = {
            "count": len(moves),
            "avg_move": float(np.mean(moves)),
            "median_move": float(np.median(moves)),
            "std_move": float(np.std(moves)),
            "win_rate": float(win_rate),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "significant": p_val < 0.05,
        }

    return results


def confluence_score(bars: pd.DataFrame, i: int) -> dict:
    """
    Score the setup quality at bar i based on indicator confluence.

    Returns a dict with individual scores and total confluence score.
    Higher score = more factors align for a potential trade.
    """
    if i < 80 or pd.isna(bars.iloc[i].get("sma")):
        return {"total": 0, "direction": "neutral", "factors": {}}

    curr = bars.iloc[i]
    factors = {}
    bull_score = 0
    bear_score = 0

    # 1. Price vs SMA
    if curr["above_sma"]:
        factors["price_above_sma"] = True
        bull_score += 1
    elif curr["below_sma"]:
        factors["price_below_sma"] = True
        bear_score += 1

    # 2. SMA slope (5-bar)
    sma_slope = curr["sma"] - bars.iloc[i - 5]["sma"]
    if sma_slope > 0:
        factors["sma_rising"] = True
        bull_score += 1
    elif sma_slope < 0:
        factors["sma_falling"] = True
        bear_score += 1

    # 3. Position within KC
    if curr["above_kc_upper"]:
        factors["above_kc_upper"] = True
        bull_score += 1  # Strong breakout
    elif curr["below_kc_lower"]:
        factors["below_kc_lower"] = True
        bear_score += 1
    elif curr["close"] > curr["kc_mid"]:
        factors["above_kc_mid"] = True
        bull_score += 0.5
    else:
        factors["below_kc_mid"] = True
        bear_score += 0.5

    # 4. Bar direction
    if curr["is_green"]:
        factors["green_bar"] = True
        bull_score += 0.5
    elif curr["is_red"]:
        factors["red_bar"] = True
        bear_score += 0.5

    # 5. Volume relative to average
    if i >= 20:
        avg_vol = bars.iloc[i - 20:i]["volume"].mean()
        if curr["volume"] > avg_vol * 1.3:
            factors["high_volume"] = True
            # Add to whichever direction the bar moved
            if curr["is_green"]:
                bull_score += 1
            else:
                bear_score += 1

    total = max(bull_score, bear_score)
    direction = "bullish" if bull_score > bear_score else "bearish" if bear_score > bull_score else "neutral"

    return {
        "total": total,
        "bull_score": bull_score,
        "bear_score": bear_score,
        "direction": direction,
        "factors": factors,
    }


def scan_opportunities(bars: pd.DataFrame, min_confluence: float = 3.0) -> pd.DataFrame:
    """
    Scan all bars for high-confluence trading opportunities.

    Returns a DataFrame of bars where confluence score >= min_confluence,
    sorted by score descending.
    """
    records = []
    for i in range(80, len(bars)):
        score = confluence_score(bars, i)
        if score["total"] >= min_confluence:
            row = bars.iloc[i]
            records.append({
                "bar_index": i,
                "timestamp": row["timestamp"],
                "close": row["close"],
                "direction": score["direction"],
                "confluence_score": score["total"],
                "bull_score": score["bull_score"],
                "bear_score": score["bear_score"],
                "factors": str(score["factors"]),
                "sma": row["sma"],
                "kc_upper": row["kc_upper"],
                "kc_lower": row["kc_lower"],
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("confluence_score", ascending=False)
    return df
