"""
Opening Range Midpoint Trade Backtest v2
=========================================

Tests the hypothesis:
  After the 15-min opening range (9:30-9:45 ET) forms,
  if price leaves the range by X+ points, comes back, and reaches
  the midpoint, there's a tradeable reaction AT the midpoint.

Trade Logic:
  - Price breaks ABOVE OR by X+ pts, pulls back to midpoint → LONG at midpoint (support)
  - Price breaks BELOW OR by X+ pts, pulls back to midpoint → SHORT at midpoint (resistance)

  The midpoint acts as a pullback entry in the direction of the original breakout.

Questions answered:
  1. What are the best conditions for a midpoint trade?
     (OR size, excursion distance, time of day, speed of pullback, etc.)
  2. What separates winning midpoint trades from losing ones?
  3. Optimal stop loss and take profit?
  4. Trailing stop vs fixed?
  5. Combined filter optimization

Usage:
  1. Ensure es_data.parquet is in the current directory (or set ES_DATA env var)
  2. Run: python3 backtest_midpoint_v2.py
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PARQUET_PATH = os.environ.get("ES_DATA", "es_data.parquet")
TICK_SIZE = 0.25
POINT = 1.0

OR_START = "09:30"
OR_END = "09:45"
RTH_END = "16:00"

# Excursion thresholds to test
EXCURSION_THRESHOLDS = [3, 5, 7, 9, 11, 13, 15, 20]

# Stop / target levels to test (points from midpoint)
STOP_LEVELS = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
TARGET_LEVELS = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15]

# Trailing stop configs
TRAILING_CONFIGS = [
    (1.0, 0.75),
    (1.0, 1.0),
    (1.5, 1.0),
    (2.0, 1.0),
    (2.0, 1.5),
    (3.0, 1.5),
    (3.0, 2.0),
    (4.0, 2.0),
    (5.0, 2.5),
    (5.0, 3.0),
]


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load ES 610-tick bar data."""
    print(f"Loading data from {path}...")
    df = pd.read_parquet(path)
    df["StartTime"] = pd.to_datetime(df["StartTime"])
    df["EndTime"] = pd.to_datetime(df["EndTime"])
    df["Date"] = df["StartTime"].dt.date
    df["Time"] = df["StartTime"].dt.time
    df["DayOfWeek"] = df["StartTime"].dt.dayofweek  # 0=Mon, 4=Fri
    print(f"  Loaded {len(df):,} bars from {df['Date'].min()} to {df['Date'].max()}")
    print(f"  {df['Date'].nunique()} unique trading days")
    return df


# ---------------------------------------------------------------------------
# OPENING RANGE CONSTRUCTION
# ---------------------------------------------------------------------------
def build_opening_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Build 15-min opening range for each trading day."""
    or_start = pd.to_datetime(OR_START).time()
    or_end = pd.to_datetime(OR_END).time()

    or_bars = df[(df["Time"] >= or_start) & (df["Time"] < or_end)].copy()

    or_ranges = or_bars.groupby("Date").agg(
        OR_High=("High", "max"),
        OR_Low=("Low", "min"),
        OR_Open=("Open", "first"),
        OR_Close=("Close", "last"),
        OR_Volume=("Volume", "sum"),
        OR_BarCount=("Close", "count"),
    ).reset_index()

    or_ranges["OR_Mid"] = (or_ranges["OR_High"] + or_ranges["OR_Low"]) / 2
    or_ranges["OR_Size"] = or_ranges["OR_High"] - or_ranges["OR_Low"]

    # Filter out days with too few bars
    or_ranges = or_ranges[or_ranges["OR_BarCount"] >= 2]

    print(f"  Built opening ranges for {len(or_ranges)} days")
    print(f"  Avg OR size: {or_ranges['OR_Size'].mean():.2f} pts")
    print(f"  Median OR size: {or_ranges['OR_Size'].median():.2f} pts")
    print(f"  Min OR size: {or_ranges['OR_Size'].min():.2f} pts")
    print(f"  Max OR size: {or_ranges['OR_Size'].max():.2f} pts")

    return or_ranges


# ---------------------------------------------------------------------------
# MIDPOINT SETUP DETECTION
# ---------------------------------------------------------------------------
@dataclass
class MidpointSetup:
    """A trade setup where price left OR and pulled back to the midpoint."""
    date: object
    direction: str  # "long" (broke above, buy at mid) or "short" (broke below, sell at mid)

    # Opening range
    or_high: float
    or_low: float
    or_mid: float
    or_size: float
    or_close_location: float  # 0=low, 1=high

    # Excursion info
    excursion_points: float
    excursion_peak_price: float
    excursion_peak_bar_idx: int
    excursion_bar_count: int  # bars with price outside OR
    volume_during_excursion: float

    # Pullback to midpoint
    midpoint_touch_time: object
    midpoint_touch_bar_idx: int
    bars_from_reentry_to_mid: int  # how quickly price went from OR boundary to mid
    bars_from_peak_to_mid: int  # bars from excursion peak to midpoint
    pullback_speed: float  # avg pts/bar during pullback
    volume_during_pullback: float
    momentum_deceleration: float  # ratio of bar sizes: last 3 bars vs excursion avg

    # Context
    day_of_week: int  # 0=Mon, 4=Fri
    time_of_midpoint_touch: object
    excursion_pct_of_or: float  # excursion / OR size

    # Bars from midpoint onward for trade simulation
    bars_after: list = field(default_factory=list)

    # Outcome (filled during analysis)
    max_favorable: float = 0.0  # max move in trade direction from midpoint
    max_adverse: float = 0.0  # max move against from midpoint
    reaction_size: float = 0.0  # how far price bounced from midpoint


def find_midpoint_setups(df: pd.DataFrame, or_ranges: pd.DataFrame,
                          min_excursion: float = 3.0) -> list:
    """
    Find setups where price:
    1. Leaves OR by min_excursion+ points
    2. Comes back and touches the midpoint

    Trade direction = same as original breakout:
      - Broke ABOVE OR → pullback to mid → LONG at mid
      - Broke BELOW OR → pullback to mid → SHORT at mid
    """
    or_end_time = pd.to_datetime(OR_END).time()
    rth_end_time = pd.to_datetime(RTH_END).time()

    setups = []

    for _, or_row in or_ranges.iterrows():
        date = or_row["Date"]
        or_high = or_row["OR_High"]
        or_low = or_row["OR_Low"]
        or_mid = or_row["OR_Mid"]
        or_size = or_row["OR_Size"]
        or_close_loc = (or_row["OR_Close"] - or_low) / or_size if or_size > 0 else 0.5

        # Get post-OR bars for this day
        day_bars = df[(df["Date"] == date) &
                      (df["Time"] >= or_end_time) &
                      (df["Time"] <= rth_end_time)].copy()

        if len(day_bars) < 5:
            continue

        day_bars = day_bars.sort_values("StartTime").reset_index(drop=True)

        # Get day of week
        dow = day_bars.iloc[0]["DayOfWeek"]

        for direction in ["long", "short"]:
            # LONG: price broke ABOVE OR, pulled back to mid → buy
            # SHORT: price broke BELOW OR, pulled back to mid → sell

            max_excursion = 0.0
            excursion_started = False
            excursion_bar_count = 0
            excursion_volume = 0
            excursion_move_sum = 0
            excursion_peak_price = 0.0
            excursion_peak_bar_idx = 0
            reentry_bar_idx = None
            already_triggered = False

            # Phase tracking
            phase = "looking_for_excursion"
            # Phases: looking_for_excursion → excursion → pullback_to_mid → done

            pullback_volume = 0
            pullback_bar_count = 0
            pullback_move_sum = 0
            last_3_bar_sizes = []

            for i in range(len(day_bars)):
                if already_triggered:
                    break

                bar = day_bars.iloc[i]

                if direction == "long":
                    # Looking for price ABOVE OR high (bullish excursion)
                    if phase == "looking_for_excursion" or phase == "excursion":
                        if bar["High"] > or_high:
                            excursion = bar["High"] - or_high
                            if excursion > max_excursion:
                                max_excursion = excursion
                                excursion_peak_price = bar["High"]
                                excursion_peak_bar_idx = i
                            if not excursion_started:
                                excursion_started = True
                            excursion_bar_count += 1
                            excursion_volume += bar.get("Volume", 0)
                            excursion_move_sum += abs(bar["Close"] - bar["Open"])
                            phase = "excursion"

                        # Check if excursion threshold met AND price pulling back
                        if phase == "excursion" and max_excursion >= min_excursion:
                            # Has price started pulling back toward midpoint?
                            if bar["Low"] <= or_high:
                                # Price is back at or below OR high - pullback started
                                phase = "pullback_to_mid"
                                reentry_bar_idx = i

                    if phase == "pullback_to_mid":
                        pullback_bar_count += 1
                        pullback_volume += bar.get("Volume", 0)
                        pullback_move_sum += abs(bar["Close"] - bar["Open"])
                        last_3_bar_sizes.append(abs(bar["Close"] - bar["Open"]))
                        if len(last_3_bar_sizes) > 3:
                            last_3_bar_sizes.pop(0)

                        # Did price reach midpoint?
                        if bar["Low"] <= or_mid:
                            # ENTRY TRIGGERED at midpoint
                            entry_price = or_mid

                            # Compute context metrics
                            avg_exc_bar_size = excursion_move_sum / max(excursion_bar_count, 1)
                            avg_last3 = np.mean(last_3_bar_sizes) if last_3_bar_sizes else avg_exc_bar_size
                            momentum_decel = avg_last3 / avg_exc_bar_size if avg_exc_bar_size > 0 else 1.0
                            pullback_speed_val = (excursion_peak_price - or_mid) / max(pullback_bar_count, 1)

                            # Collect bars from this point onward
                            bars_after_data = []
                            for j in range(i, min(i + 300, len(day_bars))):
                                rbar = day_bars.iloc[j]
                                bars_after_data.append({
                                    "Open": rbar["Open"], "High": rbar["High"],
                                    "Low": rbar["Low"], "Close": rbar["Close"],
                                    "Volume": rbar.get("Volume", 0),
                                    "StartTime": rbar["StartTime"],
                                })

                            setup = MidpointSetup(
                                date=date,
                                direction="long",
                                or_high=or_high, or_low=or_low,
                                or_mid=or_mid, or_size=or_size,
                                or_close_location=or_close_loc,
                                excursion_points=max_excursion,
                                excursion_peak_price=excursion_peak_price,
                                excursion_peak_bar_idx=excursion_peak_bar_idx,
                                excursion_bar_count=excursion_bar_count,
                                volume_during_excursion=excursion_volume,
                                midpoint_touch_time=bar["StartTime"],
                                midpoint_touch_bar_idx=i,
                                bars_from_reentry_to_mid=pullback_bar_count,
                                bars_from_peak_to_mid=i - excursion_peak_bar_idx,
                                pullback_speed=pullback_speed_val,
                                volume_during_pullback=pullback_volume,
                                momentum_deceleration=momentum_decel,
                                day_of_week=dow,
                                time_of_midpoint_touch=bar["StartTime"].time(),
                                excursion_pct_of_or=max_excursion / or_size * 100 if or_size > 0 else 0,
                                bars_after=bars_after_data,
                            )
                            _analyze_outcome(setup)
                            setups.append(setup)
                            already_triggered = True

                else:  # direction == "short"
                    # Looking for price BELOW OR low (bearish excursion)
                    if phase == "looking_for_excursion" or phase == "excursion":
                        if bar["Low"] < or_low:
                            excursion = or_low - bar["Low"]
                            if excursion > max_excursion:
                                max_excursion = excursion
                                excursion_peak_price = bar["Low"]
                                excursion_peak_bar_idx = i
                            if not excursion_started:
                                excursion_started = True
                            excursion_bar_count += 1
                            excursion_volume += bar.get("Volume", 0)
                            excursion_move_sum += abs(bar["Close"] - bar["Open"])
                            phase = "excursion"

                        if phase == "excursion" and max_excursion >= min_excursion:
                            if bar["High"] >= or_low:
                                phase = "pullback_to_mid"
                                reentry_bar_idx = i

                    if phase == "pullback_to_mid":
                        pullback_bar_count += 1
                        pullback_volume += bar.get("Volume", 0)
                        pullback_move_sum += abs(bar["Close"] - bar["Open"])
                        last_3_bar_sizes.append(abs(bar["Close"] - bar["Open"]))
                        if len(last_3_bar_sizes) > 3:
                            last_3_bar_sizes.pop(0)

                        if bar["High"] >= or_mid:
                            entry_price = or_mid

                            avg_exc_bar_size = excursion_move_sum / max(excursion_bar_count, 1)
                            avg_last3 = np.mean(last_3_bar_sizes) if last_3_bar_sizes else avg_exc_bar_size
                            momentum_decel = avg_last3 / avg_exc_bar_size if avg_exc_bar_size > 0 else 1.0
                            pullback_speed_val = (or_mid - excursion_peak_price) / max(pullback_bar_count, 1)

                            bars_after_data = []
                            for j in range(i, min(i + 300, len(day_bars))):
                                rbar = day_bars.iloc[j]
                                bars_after_data.append({
                                    "Open": rbar["Open"], "High": rbar["High"],
                                    "Low": rbar["Low"], "Close": rbar["Close"],
                                    "Volume": rbar.get("Volume", 0),
                                    "StartTime": rbar["StartTime"],
                                })

                            setup = MidpointSetup(
                                date=date,
                                direction="short",
                                or_high=or_high, or_low=or_low,
                                or_mid=or_mid, or_size=or_size,
                                or_close_location=or_close_loc,
                                excursion_points=max_excursion,
                                excursion_peak_price=excursion_peak_price,
                                excursion_peak_bar_idx=excursion_peak_bar_idx,
                                excursion_bar_count=excursion_bar_count,
                                volume_during_excursion=excursion_volume,
                                midpoint_touch_time=bar["StartTime"],
                                midpoint_touch_bar_idx=i,
                                bars_from_reentry_to_mid=pullback_bar_count,
                                bars_from_peak_to_mid=i - excursion_peak_bar_idx,
                                pullback_speed=pullback_speed_val,
                                volume_during_pullback=pullback_volume,
                                momentum_deceleration=momentum_decel,
                                day_of_week=dow,
                                time_of_midpoint_touch=bar["StartTime"].time(),
                                excursion_pct_of_or=max_excursion / or_size * 100 if or_size > 0 else 0,
                                bars_after=bars_after_data,
                            )
                            _analyze_outcome(setup)
                            setups.append(setup)
                            already_triggered = True

    return setups


def _analyze_outcome(setup: MidpointSetup):
    """Measure MFE and MAE from the midpoint entry."""
    if not setup.bars_after:
        return

    entry = setup.or_mid
    max_favorable = 0.0
    max_adverse = 0.0

    for bar in setup.bars_after:
        if setup.direction == "long":
            # Long at midpoint: favorable = up, adverse = down
            favorable = bar["High"] - entry
            adverse = entry - bar["Low"]
        else:
            # Short at midpoint: favorable = down, adverse = up
            favorable = entry - bar["Low"]
            adverse = bar["High"] - entry

        max_favorable = max(max_favorable, favorable)
        max_adverse = max(max_adverse, adverse)

    setup.max_favorable = max_favorable
    setup.max_adverse = max_adverse

    # Measure the reaction: how far did price move in our direction
    # within the first N bars after entry
    _measure_reaction(setup)


def _measure_reaction(setup: MidpointSetup):
    """Measure how much price bounced from the midpoint in the trade direction."""
    entry = setup.or_mid
    max_reaction = 0.0

    # Look at first 50 bars for the reaction
    for bar in setup.bars_after[:50]:
        if setup.direction == "long":
            reaction = bar["High"] - entry
        else:
            reaction = entry - bar["Low"]
        max_reaction = max(max_reaction, reaction)

    setup.reaction_size = max_reaction


# ---------------------------------------------------------------------------
# TRADE SIMULATION
# ---------------------------------------------------------------------------
def simulate_fixed_sl_tp(setup: MidpointSetup, stop_pts: float, target_pts: float) -> dict:
    """Simulate a trade entered at midpoint with fixed SL/TP."""
    entry = setup.or_mid

    if setup.direction == "long":
        stop = entry - stop_pts
        target = entry + target_pts
    else:
        stop = entry + stop_pts
        target = entry - target_pts

    for bar in setup.bars_after:
        if setup.direction == "long":
            if bar["Low"] <= stop:
                return {"result": "stop", "pnl": -stop_pts}
            if bar["High"] >= target:
                return {"result": "target", "pnl": target_pts}
        else:
            if bar["High"] >= stop:
                return {"result": "stop", "pnl": -stop_pts}
            if bar["Low"] <= target:
                return {"result": "target", "pnl": target_pts}

    # EOD
    last_close = setup.bars_after[-1]["Close"] if setup.bars_after else entry
    pnl = (last_close - entry) if setup.direction == "long" else (entry - last_close)
    return {"result": "eod", "pnl": pnl}


def simulate_trailing_stop(setup: MidpointSetup, initial_stop_pts: float,
                            trail_activate: float, trail_distance: float) -> dict:
    """Simulate with initial fixed stop + trailing stop after activation."""
    entry = setup.or_mid

    if setup.direction == "long":
        stop = entry - initial_stop_pts
        best_price = entry
    else:
        stop = entry + initial_stop_pts
        best_price = entry

    trailing_active = False

    for bar in setup.bars_after:
        if setup.direction == "long":
            if bar["High"] > best_price:
                best_price = bar["High"]
            if not trailing_active and (best_price - entry) >= trail_activate:
                trailing_active = True
            if trailing_active:
                trail_stop = best_price - trail_distance
                stop = max(stop, trail_stop)
            if bar["Low"] <= stop:
                pnl = stop - entry
                return {"result": "trail_stop" if trailing_active else "stop",
                        "pnl": pnl}
        else:
            if bar["Low"] < best_price:
                best_price = bar["Low"]
            if not trailing_active and (entry - best_price) >= trail_activate:
                trailing_active = True
            if trailing_active:
                trail_stop = best_price + trail_distance
                stop = min(stop, trail_stop)
            if bar["High"] >= stop:
                pnl = entry - stop
                return {"result": "trail_stop" if trailing_active else "stop",
                        "pnl": pnl}

    last_close = setup.bars_after[-1]["Close"] if setup.bars_after else entry
    pnl = (last_close - entry) if setup.direction == "long" else (entry - last_close)
    return {"result": "eod", "pnl": pnl}


# ---------------------------------------------------------------------------
# HELPER: compute stats for a group of setups with a given SL/TP
# ---------------------------------------------------------------------------
def _group_stats(setups_group, sl, tp, label=""):
    """Compute win rate, avg PnL, PF for a group of setups."""
    if not setups_group:
        return None
    results = [simulate_fixed_sl_tp(s, sl, tp) for s in setups_group]
    wins = sum(1 for r in results if r["result"] == "target")
    total_pnl = sum(r["pnl"] for r in results)
    avg_pnl = total_pnl / len(results)
    win_rate = wins / len(results) * 100
    gross_profit = sum(r["pnl"] for r in results if r["pnl"] > 0)
    gross_loss = abs(sum(r["pnl"] for r in results if r["pnl"] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.9
    return {
        "n": len(results), "wins": wins, "win_rate": win_rate,
        "avg_pnl": avg_pnl, "total_pnl": total_pnl, "pf": pf,
        "label": label,
    }


# ---------------------------------------------------------------------------
# ANALYSIS & REPORTING
# ---------------------------------------------------------------------------
def analyze_setups(setups: list) -> None:
    """Comprehensive analysis focused on what makes midpoint trades work."""
    if not setups:
        print("No setups found!")
        return

    n_long = sum(1 for s in setups if s.direction == "long")
    n_short = sum(1 for s in setups if s.direction == "short")

    print(f"\n{'='*80}")
    print("OPENING RANGE MIDPOINT TRADE BACKTEST v2")
    print("Entry: AT the midpoint | Direction: same as original breakout")
    print(f"{'='*80}")
    print(f"Total setups (min 3pt excursion, price reached midpoint): {len(setups)}")
    print(f"  Long (broke above OR, buy at mid):  {n_long}")
    print(f"  Short (broke below OR, sell at mid): {n_short}")

    # ==========================================
    # SECTION 1: MFE/MAE DISTRIBUTIONS
    # ==========================================
    print(f"\n{'='*80}")
    print("SECTION 1: HOW DOES PRICE BEHAVE AFTER TOUCHING MIDPOINT?")
    print(f"{'='*80}")

    mfes = [s.max_favorable for s in setups]
    maes = [s.max_adverse for s in setups]
    reactions = [s.reaction_size for s in setups]

    print(f"\n  Max Favorable Excursion (how far price goes IN your direction from mid):")
    for pct in [25, 50, 75, 90]:
        print(f"    {pct}th percentile: {np.percentile(mfes, pct):.1f} pts")
    print(f"    Mean: {np.mean(mfes):.1f} pts")

    print(f"\n  Max Adverse Excursion (how far price goes AGAINST you from mid):")
    for pct in [25, 50, 75, 90]:
        print(f"    {pct}th percentile: {np.percentile(maes, pct):.1f} pts")
    print(f"    Mean: {np.mean(maes):.1f} pts")

    print(f"\n  Reaction size (bounce within first 50 bars):")
    for pct in [25, 50, 75, 90]:
        print(f"    {pct}th percentile: {np.percentile(reactions, pct):.1f} pts")
    print(f"    Mean: {np.mean(reactions):.1f} pts")

    # How often does price bounce at least X pts?
    print(f"\n  How often does price bounce at least X pts from midpoint?")
    for threshold in [1, 2, 3, 4, 5, 6, 8, 10, 15]:
        hit = sum(1 for s in setups if s.reaction_size >= threshold)
        print(f"    {threshold:>3}pt+ bounce: {hit:>4} ({hit/len(setups)*100:.1f}%)")

    # By direction
    for direction in ["long", "short"]:
        dir_setups = [s for s in setups if s.direction == direction]
        if not dir_setups:
            continue
        dir_label = "LONG (broke above, buy at mid)" if direction == "long" else "SHORT (broke below, sell at mid)"
        print(f"\n  --- {dir_label} ({len(dir_setups)} setups) ---")
        dir_mfes = [s.max_favorable for s in dir_setups]
        dir_maes = [s.max_adverse for s in dir_setups]
        print(f"    Avg MFE: {np.mean(dir_mfes):.1f} pts | Avg MAE: {np.mean(dir_maes):.1f} pts")
        print(f"    Median MFE: {np.median(dir_mfes):.1f} pts | Median MAE: {np.median(dir_maes):.1f} pts")

    # ==========================================
    # SECTION 2: FACTOR ANALYSIS - WHAT MAKES WINNERS
    # ==========================================
    print(f"\n{'='*80}")
    print("SECTION 2: WHAT FACTORS MAKE A MIDPOINT TRADE A WINNER?")
    print("(Using SL=3pt, TP=3pt as baseline to classify wins/losses)")
    print(f"{'='*80}")

    # Use a reasonable SL/TP as baseline for factor analysis
    baseline_sl = 3.0
    baseline_tp = 3.0

    for s in setups:
        r = simulate_fixed_sl_tp(s, baseline_sl, baseline_tp)
        s._is_winner = r["result"] == "target"
        s._pnl = r["pnl"]

    total_winners = sum(1 for s in setups if s._is_winner)
    total_wr = total_winners / len(setups) * 100
    print(f"\n  Baseline: SL={baseline_sl}pt, TP={baseline_tp}pt → {total_wr:.1f}% win rate ({total_winners}/{len(setups)})")

    # --- Factor A: OR Size ---
    print(f"\n  A) Opening Range Size")
    or_size_buckets = [
        ("< 3 pts", 0, 3),
        ("3-5 pts", 3, 5),
        ("5-8 pts", 5, 8),
        ("8-12 pts", 8, 12),
        ("12-20 pts", 12, 20),
        ("20-40 pts", 20, 40),
        ("40+ pts", 40, 999),
    ]
    print(f"     {'OR Size':>15} {'Setups':>8} {'Winners':>8} {'Win Rate':>10} {'Avg PnL':>10} {'Avg MFE':>10} {'Avg MAE':>10}")
    print("     " + "-" * 75)
    for label, lo, hi in or_size_buckets:
        bucket = [s for s in setups if lo <= s.or_size < hi]
        if not bucket:
            continue
        wins = sum(1 for s in bucket if s._is_winner)
        wr = wins / len(bucket) * 100
        avg_pnl = np.mean([s._pnl for s in bucket])
        avg_mfe = np.mean([s.max_favorable for s in bucket])
        avg_mae = np.mean([s.max_adverse for s in bucket])
        print(f"     {label:>15} {len(bucket):>8} {wins:>8} {wr:>9.1f}% {avg_pnl:>+9.2f} {avg_mfe:>9.1f} {avg_mae:>9.1f}")

    # --- Factor B: Excursion Distance ---
    print(f"\n  B) Excursion Distance (how far price left OR)")
    print(f"     {'Excursion':>15} {'Setups':>8} {'Winners':>8} {'Win Rate':>10} {'Avg PnL':>10} {'Avg MFE':>10} {'Avg MAE':>10}")
    print("     " + "-" * 75)
    for thresh in EXCURSION_THRESHOLDS:
        bucket = [s for s in setups if s.excursion_points >= thresh]
        if not bucket:
            continue
        wins = sum(1 for s in bucket if s._is_winner)
        wr = wins / len(bucket) * 100
        avg_pnl = np.mean([s._pnl for s in bucket])
        avg_mfe = np.mean([s.max_favorable for s in bucket])
        avg_mae = np.mean([s.max_adverse for s in bucket])
        print(f"     {thresh:>13.0f}pt+ {len(bucket):>8} {wins:>8} {wr:>9.1f}% {avg_pnl:>+9.2f} {avg_mfe:>9.1f} {avg_mae:>9.1f}")

    # --- Factor C: Excursion as % of OR ---
    print(f"\n  C) Excursion as % of OR Size")
    exc_pct_buckets = [
        ("50-100%", 50, 100),
        ("100-150%", 100, 150),
        ("150-200%", 150, 200),
        ("200-300%", 200, 300),
        ("300-500%", 300, 500),
        ("500%+", 500, 9999),
    ]
    print(f"     {'Exc/OR':>15} {'Setups':>8} {'Winners':>8} {'Win Rate':>10} {'Avg PnL':>10}")
    print("     " + "-" * 55)
    for label, lo, hi in exc_pct_buckets:
        bucket = [s for s in setups if lo <= s.excursion_pct_of_or < hi]
        if not bucket:
            continue
        wins = sum(1 for s in bucket if s._is_winner)
        wr = wins / len(bucket) * 100
        avg_pnl = np.mean([s._pnl for s in bucket])
        print(f"     {label:>15} {len(bucket):>8} {wins:>8} {wr:>9.1f}% {avg_pnl:>+9.2f}")

    # --- Factor D: Time of Midpoint Touch ---
    print(f"\n  D) Time of Midpoint Touch")
    time_buckets = [
        ("9:45-10:30", datetime.time(9, 45), datetime.time(10, 30)),
        ("10:30-11:30", datetime.time(10, 30), datetime.time(11, 30)),
        ("11:30-13:00", datetime.time(11, 30), datetime.time(13, 0)),
        ("13:00-14:30", datetime.time(13, 0), datetime.time(14, 30)),
        ("14:30-16:00", datetime.time(14, 30), datetime.time(16, 0)),
    ]
    print(f"     {'Time':>15} {'Setups':>8} {'Winners':>8} {'Win Rate':>10} {'Avg PnL':>10} {'Avg Reaction':>14}")
    print("     " + "-" * 70)
    for label, t_start, t_end in time_buckets:
        bucket = [s for s in setups if t_start <= s.time_of_midpoint_touch < t_end]
        if not bucket:
            continue
        wins = sum(1 for s in bucket if s._is_winner)
        wr = wins / len(bucket) * 100
        avg_pnl = np.mean([s._pnl for s in bucket])
        avg_react = np.mean([s.reaction_size for s in bucket])
        print(f"     {label:>15} {len(bucket):>8} {wins:>8} {wr:>9.1f}% {avg_pnl:>+9.2f} {avg_react:>13.1f}")

    # --- Factor E: Speed of Pullback to Midpoint ---
    print(f"\n  E) Speed of Pullback (bars from excursion peak to midpoint)")
    pb_bars = [s.bars_from_peak_to_mid for s in setups]
    pb_quartiles = [np.percentile(pb_bars, p) for p in [25, 50, 75]]

    speed_buckets = [
        (f"Fast (≤{pb_quartiles[0]:.0f} bars)", 0, pb_quartiles[0] + 1),
        (f"Medium ({pb_quartiles[0]:.0f}-{pb_quartiles[1]:.0f})", pb_quartiles[0] + 1, pb_quartiles[1] + 1),
        (f"Slow ({pb_quartiles[1]:.0f}-{pb_quartiles[2]:.0f})", pb_quartiles[1] + 1, pb_quartiles[2] + 1),
        (f"Very slow (>{pb_quartiles[2]:.0f})", pb_quartiles[2] + 1, 9999),
    ]
    print(f"     {'Speed':>25} {'Setups':>8} {'Winners':>8} {'Win Rate':>10} {'Avg PnL':>10}")
    print("     " + "-" * 65)
    for label, lo, hi in speed_buckets:
        bucket = [s for s in setups if lo <= s.bars_from_peak_to_mid < hi]
        if not bucket:
            continue
        wins = sum(1 for s in bucket if s._is_winner)
        wr = wins / len(bucket) * 100
        avg_pnl = np.mean([s._pnl for s in bucket])
        print(f"     {label:>25} {len(bucket):>8} {wins:>8} {wr:>9.1f}% {avg_pnl:>+9.2f}")

    # --- Factor F: Momentum Deceleration ---
    print(f"\n  F) Momentum Deceleration at Midpoint Approach")
    print(f"     (Ratio of last 3 bar sizes vs excursion avg bar size)")
    print(f"     < 1.0 = decelerating (bars getting smaller), > 1.0 = accelerating")
    decel_buckets = [
        ("Strong decel (<0.5)", 0, 0.5),
        ("Mild decel (0.5-0.8)", 0.5, 0.8),
        ("Neutral (0.8-1.2)", 0.8, 1.2),
        ("Accelerating (>1.2)", 1.2, 99),
    ]
    print(f"     {'Momentum':>25} {'Setups':>8} {'Winners':>8} {'Win Rate':>10} {'Avg PnL':>10}")
    print("     " + "-" * 65)
    for label, lo, hi in decel_buckets:
        bucket = [s for s in setups if lo <= s.momentum_deceleration < hi]
        if not bucket:
            continue
        wins = sum(1 for s in bucket if s._is_winner)
        wr = wins / len(bucket) * 100
        avg_pnl = np.mean([s._pnl for s in bucket])
        print(f"     {label:>25} {len(bucket):>8} {wins:>8} {wr:>9.1f}% {avg_pnl:>+9.2f}")

    # --- Factor G: Day of Week ---
    print(f"\n  G) Day of Week")
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    print(f"     {'Day':>15} {'Setups':>8} {'Winners':>8} {'Win Rate':>10} {'Avg PnL':>10}")
    print("     " + "-" * 55)
    for dow in range(5):
        bucket = [s for s in setups if s.day_of_week == dow]
        if not bucket:
            continue
        wins = sum(1 for s in bucket if s._is_winner)
        wr = wins / len(bucket) * 100
        avg_pnl = np.mean([s._pnl for s in bucket])
        print(f"     {day_names[dow]:>15} {len(bucket):>8} {wins:>8} {wr:>9.1f}% {avg_pnl:>+9.2f}")

    # --- Factor H: OR Close Location ---
    print(f"\n  H) OR Close Location (0=closed at low, 1=closed at high)")
    close_buckets = [
        ("Near Low (<0.33)", 0, 0.33),
        ("Near Mid (0.33-0.67)", 0.33, 0.67),
        ("Near High (>0.67)", 0.67, 1.01),
    ]
    print(f"     {'Close Loc':>25} {'Setups':>8} {'Winners':>8} {'Win Rate':>10} {'Avg PnL':>10}")
    print("     " + "-" * 65)
    for label, lo, hi in close_buckets:
        bucket = [s for s in setups if lo <= s.or_close_location < hi]
        if not bucket:
            continue
        wins = sum(1 for s in bucket if s._is_winner)
        wr = wins / len(bucket) * 100
        avg_pnl = np.mean([s._pnl for s in bucket])
        print(f"     {label:>25} {len(bucket):>8} {wins:>8} {wr:>9.1f}% {avg_pnl:>+9.2f}")

    # --- Factor I: Direction ---
    print(f"\n  I) Direction")
    print(f"     {'Direction':>15} {'Setups':>8} {'Winners':>8} {'Win Rate':>10} {'Avg PnL':>10} {'Avg MFE':>10} {'Avg MAE':>10}")
    print("     " + "-" * 75)
    for direction in ["long", "short"]:
        bucket = [s for s in setups if s.direction == direction]
        if not bucket:
            continue
        wins = sum(1 for s in bucket if s._is_winner)
        wr = wins / len(bucket) * 100
        avg_pnl = np.mean([s._pnl for s in bucket])
        avg_mfe = np.mean([s.max_favorable for s in bucket])
        avg_mae = np.mean([s.max_adverse for s in bucket])
        print(f"     {direction:>15} {len(bucket):>8} {wins:>8} {wr:>9.1f}% {avg_pnl:>+9.2f} {avg_mfe:>9.1f} {avg_mae:>9.1f}")

    # ==========================================
    # SECTION 3: OPTIMAL SL/TP
    # ==========================================
    print(f"\n{'='*80}")
    print("SECTION 3: OPTIMAL STOP LOSS & TAKE PROFIT (entry at midpoint)")
    print(f"{'='*80}")

    print(f"\n  Fixed SL/TP Grid: Win Rate / Avg PnL / Profit Factor")
    print(f"  {'':>8}", end="")
    for tp in TARGET_LEVELS:
        print(f"  TP={tp:>4.1f}pt", end="")
    print()

    best_combo = None
    best_pf = 0
    all_combos = []

    for sl in STOP_LEVELS:
        print(f"  SL={sl:>4.1f}", end="")
        for tp in TARGET_LEVELS:
            results = [simulate_fixed_sl_tp(s, sl, tp) for s in setups]
            wins = sum(1 for r in results if r["result"] == "target")
            total_pnl = sum(r["pnl"] for r in results)
            avg_pnl = total_pnl / len(results)
            win_rate = wins / len(results) * 100
            gross_profit = sum(r["pnl"] for r in results if r["pnl"] > 0)
            gross_loss = abs(sum(r["pnl"] for r in results if r["pnl"] < 0))
            pf = gross_profit / gross_loss if gross_loss > 0 else 99.9

            combo = (sl, tp, win_rate, avg_pnl, total_pnl, pf, len(results))
            all_combos.append(combo)

            # Track best by profit factor (must have positive PnL)
            if avg_pnl > 0 and pf > best_pf:
                best_pf = pf
                best_combo = combo
            elif best_combo is None and avg_pnl > (best_combo[3] if best_combo else -999):
                best_combo = combo

            print(f"  {win_rate:>3.0f}%/{avg_pnl:>+.1f}", end="")
        print()

    # Find best by avg PnL if no profitable combo found
    if best_combo is None or best_combo[3] <= 0:
        best_combo = max(all_combos, key=lambda x: x[3])

    print(f"\n  BEST COMBO (by {'profit factor' if best_pf > 1 else 'avg PnL'}):")
    print(f"    SL={best_combo[0]:.1f}pt, TP={best_combo[1]:.1f}pt")
    print(f"    Win Rate: {best_combo[2]:.1f}%")
    print(f"    Avg PnL/trade: {best_combo[3]:+.2f}pt (${best_combo[3]*50:+.0f} per contract)")
    print(f"    Total PnL: {best_combo[4]:+.1f}pt (${best_combo[4]*50:+.0f} per contract)")
    print(f"    Profit Factor: {best_combo[5]:.2f}")
    print(f"    Trades: {best_combo[6]}")

    # Top 10 combos by avg PnL
    print(f"\n  Top 10 SL/TP Combos by Avg PnL:")
    print(f"  {'SL':>6} {'TP':>6} {'Win%':>7} {'Avg PnL':>10} {'Total PnL':>12} {'PF':>8} {'Trades':>8}")
    print("  " + "-" * 60)
    top10 = sorted(all_combos, key=lambda x: x[3], reverse=True)[:10]
    for c in top10:
        print(f"  {c[0]:>5.1f} {c[1]:>5.1f} {c[2]:>6.1f}% {c[3]:>+9.2f} {c[4]:>+11.1f} {c[5]:>7.2f} {c[6]:>8}")

    # ==========================================
    # SECTION 4: TRAILING STOP
    # ==========================================
    print(f"\n{'='*80}")
    print("SECTION 4: TRAILING STOP PERFORMANCE")
    print(f"{'='*80}")

    best_fixed_sl = best_combo[0]
    print(f"\n  Using initial SL = {best_fixed_sl:.1f}pt (best from fixed analysis)")
    print(f"  {'Activate':>10} {'Trail':>8} {'Win%':>8} {'Avg PnL':>10} {'Total PnL':>12} {'PF':>8}")
    print("  " + "-" * 60)

    best_trail = None
    best_trail_pnl = -999

    for activate, trail in TRAILING_CONFIGS:
        results = [simulate_trailing_stop(s, best_fixed_sl, activate, trail) for s in setups]
        wins = sum(1 for r in results if r["pnl"] > 0)
        total_pnl = sum(r["pnl"] for r in results)
        avg_pnl = total_pnl / len(results)
        win_rate = wins / len(results) * 100
        gross_profit = sum(r["pnl"] for r in results if r["pnl"] > 0)
        gross_loss = abs(sum(r["pnl"] for r in results if r["pnl"] < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else 99.9

        if avg_pnl > best_trail_pnl:
            best_trail_pnl = avg_pnl
            best_trail = (activate, trail, win_rate, avg_pnl, total_pnl, pf)

        print(f"  {activate:>8.1f}pt {trail:>7.1f}pt {win_rate:>7.1f}% {avg_pnl:>+9.2f}pt {total_pnl:>+11.1f}pt {pf:>7.2f}")

    if best_trail:
        print(f"\n  BEST TRAILING: Activate at +{best_trail[0]:.1f}pt, Trail {best_trail[1]:.1f}pt")
        print(f"    Win Rate: {best_trail[2]:.1f}%, Avg PnL: {best_trail[3]:+.2f}pt, PF: {best_trail[5]:.2f}")

    print(f"\n  --- COMPARISON ---")
    print(f"  Best Fixed  (SL={best_combo[0]:.1f}, TP={best_combo[1]:.1f}): avg {best_combo[3]:+.2f}pt/trade, PF {best_combo[5]:.2f}")
    if best_trail:
        print(f"  Best Trail  (Act={best_trail[0]:.1f}, Tr={best_trail[1]:.1f}): avg {best_trail[3]:+.2f}pt/trade, PF {best_trail[5]:.2f}")

    # ==========================================
    # SECTION 5: COMBINED FILTER OPTIMIZATION
    # ==========================================
    print(f"\n{'='*80}")
    print("SECTION 5: COMBINED FILTER OPTIMIZATION")
    print("Testing combinations of the most promising factors")
    print(f"{'='*80}")

    # Use best SL/TP from section 3
    opt_sl = best_combo[0]
    opt_tp = best_combo[1]
    print(f"\n  Using SL={opt_sl:.1f}pt, TP={opt_tp:.1f}pt")

    # Define filter functions
    filters = {
        "excursion_5pt+": lambda s: s.excursion_points >= 5,
        "excursion_7pt+": lambda s: s.excursion_points >= 7,
        "excursion_9pt+": lambda s: s.excursion_points >= 9,
        "OR_size_3-8pt": lambda s: 3 <= s.or_size < 8,
        "OR_size_5-15pt": lambda s: 5 <= s.or_size < 15,
        "OR_size_8-25pt": lambda s: 8 <= s.or_size < 25,
        "morning_entry": lambda s: s.time_of_midpoint_touch < datetime.time(12, 0),
        "afternoon_entry": lambda s: s.time_of_midpoint_touch >= datetime.time(12, 0),
        "10:30-14:30": lambda s: datetime.time(10, 30) <= s.time_of_midpoint_touch < datetime.time(14, 30),
        "long_only": lambda s: s.direction == "long",
        "short_only": lambda s: s.direction == "short",
        "decel_pullback": lambda s: s.momentum_deceleration < 0.8,
        "fast_pullback": lambda s: s.bars_from_peak_to_mid <= np.median([x.bars_from_peak_to_mid for x in setups]),
        "slow_pullback": lambda s: s.bars_from_peak_to_mid > np.median([x.bars_from_peak_to_mid for x in setups]),
    }

    print(f"\n  Single Filters:")
    print(f"  {'Filter':>30} {'Setups':>8} {'Win%':>8} {'Avg PnL':>10} {'Total PnL':>12} {'PF':>8}")
    print("  " + "-" * 80)

    for fname, ffunc in filters.items():
        filtered = [s for s in setups if ffunc(s)]
        if len(filtered) < 5:
            continue
        stats = _group_stats(filtered, opt_sl, opt_tp)
        if stats:
            print(f"  {fname:>30} {stats['n']:>8} {stats['win_rate']:>7.1f}% {stats['avg_pnl']:>+9.2f} {stats['total_pnl']:>+11.1f} {stats['pf']:>7.2f}")

    # Promising combinations
    print(f"\n  Promising Combinations:")
    print(f"  {'Combo':>50} {'Setups':>8} {'Win%':>8} {'Avg PnL':>10} {'Total PnL':>12} {'PF':>8}")
    print("  " + "-" * 100)

    combos = [
        ("exc_7pt + OR_3-8", lambda s: s.excursion_points >= 7 and 3 <= s.or_size < 8),
        ("exc_7pt + OR_5-15", lambda s: s.excursion_points >= 7 and 5 <= s.or_size < 15),
        ("exc_7pt + OR_8-25", lambda s: s.excursion_points >= 7 and 8 <= s.or_size < 25),
        ("exc_5pt + morning", lambda s: s.excursion_points >= 5 and s.time_of_midpoint_touch < datetime.time(12, 0)),
        ("exc_5pt + afternoon", lambda s: s.excursion_points >= 5 and s.time_of_midpoint_touch >= datetime.time(12, 0)),
        ("exc_7pt + 10:30-14:30", lambda s: s.excursion_points >= 7 and datetime.time(10, 30) <= s.time_of_midpoint_touch < datetime.time(14, 30)),
        ("exc_7pt + long_only", lambda s: s.excursion_points >= 7 and s.direction == "long"),
        ("exc_7pt + short_only", lambda s: s.excursion_points >= 7 and s.direction == "short"),
        ("exc_7pt + decel", lambda s: s.excursion_points >= 7 and s.momentum_deceleration < 0.8),
        ("exc_5pt + OR_5-15 + 10:30-14:30", lambda s: s.excursion_points >= 5 and 5 <= s.or_size < 15 and datetime.time(10, 30) <= s.time_of_midpoint_touch < datetime.time(14, 30)),
        ("exc_7pt + OR_5-15 + long", lambda s: s.excursion_points >= 7 and 5 <= s.or_size < 15 and s.direction == "long"),
        ("exc_7pt + OR_5-15 + short", lambda s: s.excursion_points >= 7 and 5 <= s.or_size < 15 and s.direction == "short"),
        ("exc_9pt + OR_5-15", lambda s: s.excursion_points >= 9 and 5 <= s.or_size < 15),
        ("exc_5pt + slow_pullback", lambda s: s.excursion_points >= 5 and s.bars_from_peak_to_mid > np.median([x.bars_from_peak_to_mid for x in setups])),
        ("exc_5pt + fast_pullback", lambda s: s.excursion_points >= 5 and s.bars_from_peak_to_mid <= np.median([x.bars_from_peak_to_mid for x in setups])),
        ("exc_7pt + decel + OR_5-15", lambda s: s.excursion_points >= 7 and s.momentum_deceleration < 0.8 and 5 <= s.or_size < 15),
    ]

    best_combo_filter = None
    best_combo_filter_pf = 0

    for cname, cfunc in combos:
        filtered = [s for s in setups if cfunc(s)]
        if len(filtered) < 5:
            continue
        stats = _group_stats(filtered, opt_sl, opt_tp)
        if stats:
            print(f"  {cname:>50} {stats['n']:>8} {stats['win_rate']:>7.1f}% {stats['avg_pnl']:>+9.2f} {stats['total_pnl']:>+11.1f} {stats['pf']:>7.2f}")
            if stats['avg_pnl'] > 0 and stats['pf'] > best_combo_filter_pf and stats['n'] >= 10:
                best_combo_filter_pf = stats['pf']
                best_combo_filter = (cname, stats)

    if best_combo_filter:
        print(f"\n  BEST FILTER COMBO: {best_combo_filter[0]}")
        s = best_combo_filter[1]
        print(f"    {s['n']} trades, {s['win_rate']:.1f}% WR, avg PnL {s['avg_pnl']:+.2f}pt, PF {s['pf']:.2f}")

    # ==========================================
    # SECTION 6: RE-OPTIMIZE SL/TP FOR BEST FILTER
    # ==========================================
    if best_combo_filter:
        print(f"\n{'='*80}")
        print(f"SECTION 6: RE-OPTIMIZE SL/TP FOR BEST FILTER")
        print(f"Filter: {best_combo_filter[0]}")
        print(f"{'='*80}")

        # Get the filter function
        best_filter_name = best_combo_filter[0]
        best_filter_func = None
        for cname, cfunc in combos:
            if cname == best_filter_name:
                best_filter_func = cfunc
                break

        if best_filter_func:
            filtered_setups = [s for s in setups if best_filter_func(s)]
            print(f"\n  Setups after filter: {len(filtered_setups)}")

            print(f"\n  Top 15 SL/TP Combos:")
            print(f"  {'SL':>6} {'TP':>6} {'Win%':>7} {'Avg PnL':>10} {'Total PnL':>12} {'PF':>8} {'Trades':>8}")
            print("  " + "-" * 60)

            filtered_combos = []
            for sl in STOP_LEVELS:
                for tp in TARGET_LEVELS:
                    results = [simulate_fixed_sl_tp(s, sl, tp) for s in filtered_setups]
                    wins = sum(1 for r in results if r["result"] == "target")
                    total_pnl = sum(r["pnl"] for r in results)
                    avg_pnl = total_pnl / len(results)
                    win_rate = wins / len(results) * 100
                    gross_profit = sum(r["pnl"] for r in results if r["pnl"] > 0)
                    gross_loss = abs(sum(r["pnl"] for r in results if r["pnl"] < 0))
                    pf = gross_profit / gross_loss if gross_loss > 0 else 99.9
                    filtered_combos.append((sl, tp, win_rate, avg_pnl, total_pnl, pf, len(results)))

            top15 = sorted(filtered_combos, key=lambda x: x[3], reverse=True)[:15]
            for c in top15:
                print(f"  {c[0]:>5.1f} {c[1]:>5.1f} {c[2]:>6.1f}% {c[3]:>+9.2f} {c[4]:>+11.1f} {c[5]:>7.2f} {c[6]:>8}")

            # Also test trailing stops on filtered setups
            best_filt_sl = top15[0][0]
            print(f"\n  Trailing Stop on Filtered (initial SL={best_filt_sl:.1f}pt):")
            print(f"  {'Activate':>10} {'Trail':>8} {'Win%':>8} {'Avg PnL':>10} {'Total PnL':>12} {'PF':>8}")
            print("  " + "-" * 60)

            for activate, trail in TRAILING_CONFIGS:
                results = [simulate_trailing_stop(s, best_filt_sl, activate, trail)
                           for s in filtered_setups]
                wins = sum(1 for r in results if r["pnl"] > 0)
                total_pnl = sum(r["pnl"] for r in results)
                avg_pnl = total_pnl / len(results)
                win_rate = wins / len(results) * 100
                gross_profit = sum(r["pnl"] for r in results if r["pnl"] > 0)
                gross_loss = abs(sum(r["pnl"] for r in results if r["pnl"] < 0))
                pf = gross_profit / gross_loss if gross_loss > 0 else 99.9
                print(f"  {activate:>8.1f}pt {trail:>7.1f}pt {win_rate:>7.1f}% {avg_pnl:>+9.2f}pt {total_pnl:>+11.1f}pt {pf:>7.2f}")

    # ==========================================
    # SECTION 7: DAILY BREAKDOWN
    # ==========================================
    print(f"\n{'='*80}")
    print(f"SECTION 7: DAILY P&L (SL={best_combo[0]:.1f}, TP={best_combo[1]:.1f})")
    print(f"{'='*80}")

    sl, tp = best_combo[0], best_combo[1]
    daily_results = {}
    for s in setups:
        r = simulate_fixed_sl_tp(s, sl, tp)
        d = str(s.date)
        if d not in daily_results:
            daily_results[d] = {"trades": 0, "pnl": 0, "wins": 0}
        daily_results[d]["trades"] += 1
        daily_results[d]["pnl"] += r["pnl"]
        if r["pnl"] > 0:
            daily_results[d]["wins"] += 1

    sorted_days = sorted(daily_results.keys())
    cumulative = 0
    print(f"\n  {'Date':>12} {'Trades':>8} {'PnL':>10} {'Cumulative':>12}")
    print("  " + "-" * 45)

    for d in sorted_days:
        dr = daily_results[d]
        cumulative += dr["pnl"]
        print(f"  {d:>12} {dr['trades']:>8} {dr['pnl']:>+9.2f}pt {cumulative:>+11.2f}pt")

    win_days = sum(1 for d in daily_results.values() if d["pnl"] > 0)
    lose_days = sum(1 for d in daily_results.values() if d["pnl"] < 0)
    flat_days = sum(1 for d in daily_results.values() if d["pnl"] == 0)

    print(f"\n  Total days with trades: {len(daily_results)}")
    print(f"  Winning days: {win_days} | Losing days: {lose_days} | Flat: {flat_days}")
    print(f"  Max daily gain: {max(d['pnl'] for d in daily_results.values()):+.2f}pt")
    print(f"  Max daily loss: {min(d['pnl'] for d in daily_results.values()):+.2f}pt")
    print(f"  Final cumulative PnL: {cumulative:+.2f}pt (${cumulative*50:+.0f} per contract)")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data(PARQUET_PATH)
    or_ranges = build_opening_ranges(df)

    print(f"\n{'='*80}")
    print("SCANNING FOR MIDPOINT SETUPS...")
    print("(Price must leave OR by 3+ pts AND reach midpoint on pullback)")
    print(f"{'='*80}")

    setups = find_midpoint_setups(df, or_ranges, min_excursion=3.0)
    print(f"Found {len(setups)} midpoint setups")
    print(f"  Long (broke above, buy at mid):  {sum(1 for s in setups if s.direction == 'long')}")
    print(f"  Short (broke below, sell at mid): {sum(1 for s in setups if s.direction == 'short')}")

    analyze_setups(setups)

    print(f"\n{'='*80}")
    print("BACKTEST COMPLETE")
    print(f"{'='*80}")
