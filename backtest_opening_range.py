"""
Opening Range Midpoint Reaction Backtest
=========================================

Tests the hypothesis:
  After the 15-min opening range (9:30-9:45 ET) forms,
  if price leaves the range by X+ points and re-enters,
  does it react at the midpoint?

Questions answered:
  1. How far must price leave the OR to improve midpoint reaction probability?
  2. What other factors predict a strong midpoint reaction?
  3. Optimal stop loss and take profit levels?
  4. Trailing stop vs fixed TP performance?
  5. Does price react 6 ticks before the midpoint or at it?

Usage:
  1. Download the parquet first:
     curl -o es_data.parquet http://10.211.55.3:8000/download
  2. Run:
     python3 backtest_opening_range.py
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PARQUET_PATH = os.environ.get("ES_DATA", "es_data.parquet")
TICK_SIZE = 0.25            # ES tick size
POINT = 1.0                 # 1 point = 4 ticks for ES

OR_START = "09:30"
OR_END   = "09:45"
RTH_END  = "16:00"

# Excursion thresholds to test (points beyond OR)
EXCURSION_THRESHOLDS = [3, 5, 7, 9, 11, 13, 15]

# Stop / target levels to test (points)
STOP_LEVELS   = [1, 1.5, 2, 2.5, 3, 4, 5, 6]
TARGET_LEVELS = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]

# Trailing stop configs (activation, trail distance) in points
TRAILING_CONFIGS = [
    (1.0, 1.0),    # activate at +1, trail 1 pt
    (1.5, 1.0),    # activate at +1.5, trail 1 pt
    (2.0, 1.0),    # activate at +2, trail 1 pt
    (2.0, 1.5),    # activate at +2, trail 1.5 pt
    (3.0, 1.5),    # activate at +3, trail 1.5 pt
    (3.0, 2.0),    # activate at +3, trail 2 pt
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
    print(f"  Loaded {len(df):,} bars from {df['Date'].min()} to {df['Date'].max()}")
    print(f"  {df['Date'].nunique()} unique trading days")
    return df


# ---------------------------------------------------------------------------
# OPENING RANGE CONSTRUCTION
# ---------------------------------------------------------------------------
def build_opening_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each trading day, compute the 15-min opening range (9:30-9:45).
    Returns DataFrame with columns: Date, OR_High, OR_Low, OR_Mid, OR_Size
    """
    or_start = pd.to_datetime(OR_START).time()
    or_end   = pd.to_datetime(OR_END).time()

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

    # Filter out days with too few bars in OR (data gaps)
    or_ranges = or_ranges[or_ranges["OR_BarCount"] >= 2]

    print(f"  Built opening ranges for {len(or_ranges)} days")
    print(f"  Avg OR size: {or_ranges['OR_Size'].mean():.2f} pts")
    print(f"  Median OR size: {or_ranges['OR_Size'].median():.2f} pts")

    return or_ranges


# ---------------------------------------------------------------------------
# EXCURSION & RE-ENTRY DETECTION
# ---------------------------------------------------------------------------
@dataclass
class TradeSetup:
    """A single trade setup where price leaves OR and re-enters toward midpoint."""
    date: object
    direction: str              # "long" (price went below OR, re-entering up) or "short" (above OR, re-entering down)
    or_high: float
    or_low: float
    or_mid: float
    or_size: float
    excursion_points: float     # how far price went beyond OR
    excursion_bar_count: int    # how many bars price spent outside OR
    reentry_price: float        # price when re-entering OR
    reentry_time: object        # timestamp of re-entry
    reentry_bar_idx: int        # index in the day's bars

    # Context variables
    or_close_location: float    # where OR closed relative to range (0=low, 1=high)
    pre_excursion_momentum: float  # avg bar size during excursion
    volume_during_excursion: float
    time_of_reentry: object     # time of day

    # Outcome tracking (filled during simulation)
    bars_after: list = field(default_factory=list)  # subsequent bar data
    max_favorable: float = 0.0   # max move toward midpoint after entry
    max_adverse: float = 0.0     # max move against (beyond reentry point away from mid)
    hit_midpoint: bool = False
    hit_6ticks_before_mid: bool = False  # reaction 6 ticks (1.5 pts) before midpoint
    midpoint_reaction_size: float = 0.0  # how much price bounced at/near midpoint


def find_setups(df: pd.DataFrame, or_ranges: pd.DataFrame, min_excursion: float = 3.0) -> list:
    """
    Find all trade setups where:
    1. Price leaves the opening range
    2. Goes at least min_excursion points beyond
    3. Re-enters the opening range
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

        # Get post-OR bars for this day (9:45 onwards, RTH only)
        day_bars = df[(df["Date"] == date) &
                      (df["Time"] >= or_end_time) &
                      (df["Time"] <= rth_end_time)].copy()

        if len(day_bars) < 5:
            continue

        day_bars = day_bars.sort_values("StartTime").reset_index(drop=True)

        # Track state: looking for excursion then re-entry
        # Can have multiple setups per day (upside and downside)

        for direction in ["long", "short"]:
            # "long" setup: price breaks BELOW OR, then re-enters upward toward mid
            # "short" setup: price breaks ABOVE OR, then re-enters downward toward mid

            max_excursion = 0.0
            excursion_started = False
            excursion_bar_count = 0
            excursion_volume = 0
            excursion_move_sum = 0
            first_excursion_bar = None
            already_triggered = False  # only take first re-entry per direction per day

            for i, bar in day_bars.iterrows():
                if already_triggered:
                    break

                if direction == "long":
                    # Looking for price below OR low
                    if bar["Low"] < or_low:
                        excursion = or_low - bar["Low"]
                        if excursion > max_excursion:
                            max_excursion = excursion
                        if not excursion_started:
                            excursion_started = True
                            first_excursion_bar = i
                        excursion_bar_count += 1
                        excursion_volume += bar.get("Volume", 0)
                        excursion_move_sum += abs(bar["Close"] - bar["Open"])

                    # Check for re-entry: price comes back into OR
                    if excursion_started and max_excursion >= min_excursion:
                        if bar["High"] >= or_low:
                            # Re-entry! Price is coming back into the range
                            reentry_price = or_low  # approximate entry at OR low
                            avg_bar_size = excursion_move_sum / max(excursion_bar_count, 1)

                            # Collect subsequent bars for outcome analysis
                            bar_position = day_bars.index.get_loc(i)
                            remaining_bars = day_bars.iloc[bar_position:]
                            bars_after_data = []
                            for _, rbar in remaining_bars.head(200).iterrows():
                                bars_after_data.append({
                                    "Open": rbar["Open"], "High": rbar["High"],
                                    "Low": rbar["Low"], "Close": rbar["Close"],
                                    "Volume": rbar.get("Volume", 0),
                                    "StartTime": rbar["StartTime"],
                                })

                            setup = TradeSetup(
                                date=date,
                                direction="long",
                                or_high=or_high,
                                or_low=or_low,
                                or_mid=or_mid,
                                or_size=or_size,
                                excursion_points=max_excursion,
                                excursion_bar_count=excursion_bar_count,
                                reentry_price=reentry_price,
                                reentry_time=bar["StartTime"],
                                reentry_bar_idx=bar_position,
                                or_close_location=or_close_loc,
                                pre_excursion_momentum=avg_bar_size,
                                volume_during_excursion=excursion_volume,
                                time_of_reentry=bar["StartTime"].time(),
                                bars_after=bars_after_data,
                            )

                            # Analyze outcome
                            _analyze_outcome(setup)
                            setups.append(setup)
                            already_triggered = True

                else:  # direction == "short"
                    # Looking for price above OR high
                    if bar["High"] > or_high:
                        excursion = bar["High"] - or_high
                        if excursion > max_excursion:
                            max_excursion = excursion
                        if not excursion_started:
                            excursion_started = True
                            first_excursion_bar = i
                        excursion_bar_count += 1
                        excursion_volume += bar.get("Volume", 0)
                        excursion_move_sum += abs(bar["Close"] - bar["Open"])

                    # Check for re-entry
                    if excursion_started and max_excursion >= min_excursion:
                        if bar["Low"] <= or_high:
                            reentry_price = or_high
                            avg_bar_size = excursion_move_sum / max(excursion_bar_count, 1)

                            bar_position = day_bars.index.get_loc(i)
                            remaining_bars = day_bars.iloc[bar_position:]
                            bars_after_data = []
                            for _, rbar in remaining_bars.head(200).iterrows():
                                bars_after_data.append({
                                    "Open": rbar["Open"], "High": rbar["High"],
                                    "Low": rbar["Low"], "Close": rbar["Close"],
                                    "Volume": rbar.get("Volume", 0),
                                    "StartTime": rbar["StartTime"],
                                })

                            setup = TradeSetup(
                                date=date,
                                direction="short",
                                or_high=or_high,
                                or_low=or_low,
                                or_mid=or_mid,
                                or_size=or_size,
                                excursion_points=max_excursion,
                                excursion_bar_count=excursion_bar_count,
                                reentry_price=reentry_price,
                                reentry_time=bar["StartTime"],
                                reentry_bar_idx=bar_position,
                                or_close_location=or_close_loc,
                                pre_excursion_momentum=avg_bar_size,
                                volume_during_excursion=excursion_volume,
                                time_of_reentry=bar["StartTime"].time(),
                                bars_after=bars_after_data,
                            )

                            _analyze_outcome(setup)
                            setups.append(setup)
                            already_triggered = True

    return setups


def _analyze_outcome(setup: TradeSetup):
    """Analyze what happened after re-entry: did price reach midpoint? How far did it go?"""
    if not setup.bars_after:
        return

    mid = setup.or_mid
    entry = setup.reentry_price
    six_ticks = 6 * TICK_SIZE  # 1.5 points

    max_favorable = 0.0
    max_adverse = 0.0

    for bar in setup.bars_after:
        if setup.direction == "long":
            # Favorable = price moving UP toward/past midpoint
            favorable = bar["High"] - entry
            adverse = entry - bar["Low"]

            # Check if price reached 6 ticks before midpoint
            if bar["High"] >= (mid - six_ticks):
                setup.hit_6ticks_before_mid = True
            # Check if price reached midpoint
            if bar["High"] >= mid:
                setup.hit_midpoint = True

        else:  # short
            # Favorable = price moving DOWN toward/past midpoint
            favorable = entry - bar["Low"]
            adverse = bar["High"] - entry

            if bar["Low"] <= (mid + six_ticks):
                setup.hit_6ticks_before_mid = True
            if bar["Low"] <= mid:
                setup.hit_midpoint = True

        max_favorable = max(max_favorable, favorable)
        max_adverse = max(max_adverse, adverse)

    setup.max_favorable = max_favorable
    setup.max_adverse = max_adverse

    # Measure reaction at midpoint: how much did price bounce after touching mid area
    _measure_midpoint_reaction(setup)


def _measure_midpoint_reaction(setup: TradeSetup):
    """Measure the reaction (bounce) when price reaches the midpoint area."""
    mid = setup.or_mid
    six_ticks = 6 * TICK_SIZE

    reached_mid_area = False
    reaction_start_price = None
    max_reaction = 0.0

    for bar in setup.bars_after:
        if setup.direction == "long":
            if not reached_mid_area and bar["High"] >= (mid - six_ticks):
                reached_mid_area = True
                reaction_start_price = bar["High"]
            if reached_mid_area and reaction_start_price:
                # Reaction = how much price dropped after reaching mid area
                reaction = reaction_start_price - bar["Low"]
                max_reaction = max(max_reaction, reaction)
        else:
            if not reached_mid_area and bar["Low"] <= (mid + six_ticks):
                reached_mid_area = True
                reaction_start_price = bar["Low"]
            if reached_mid_area and reaction_start_price:
                reaction = bar["High"] - reaction_start_price
                max_reaction = max(max_reaction, reaction)

    setup.midpoint_reaction_size = max_reaction


# ---------------------------------------------------------------------------
# TRADE SIMULATION
# ---------------------------------------------------------------------------
def simulate_fixed_sl_tp(setup: TradeSetup, stop_pts: float, target_pts: float) -> dict:
    """Simulate a trade with fixed stop loss and take profit."""
    entry = setup.reentry_price

    if setup.direction == "long":
        stop = entry - stop_pts
        target = entry + target_pts
    else:
        stop = entry + stop_pts
        target = entry - target_pts

    for bar in setup.bars_after:
        if setup.direction == "long":
            # Check stop first (conservative)
            if bar["Low"] <= stop:
                return {"result": "loss", "pnl": -stop_pts}
            if bar["High"] >= target:
                return {"result": "win", "pnl": target_pts}
        else:
            if bar["High"] >= stop:
                return {"result": "loss", "pnl": -stop_pts}
            if bar["Low"] <= target:
                return {"result": "win", "pnl": target_pts}

    # End of day - close at last bar's close
    last_close = setup.bars_after[-1]["Close"] if setup.bars_after else entry
    if setup.direction == "long":
        pnl = last_close - entry
    else:
        pnl = entry - last_close

    return {"result": "eod", "pnl": pnl}


def simulate_trailing_stop(setup: TradeSetup, stop_pts: float,
                           trail_activate: float, trail_distance: float) -> dict:
    """Simulate with initial fixed stop + trailing stop after activation."""
    entry = setup.reentry_price

    if setup.direction == "long":
        stop = entry - stop_pts
        best_price = entry
    else:
        stop = entry + stop_pts
        best_price = entry

    trailing_active = False

    for bar in setup.bars_after:
        if setup.direction == "long":
            # Update best price
            if bar["High"] > best_price:
                best_price = bar["High"]

            # Activate trailing stop
            if not trailing_active and (best_price - entry) >= trail_activate:
                trailing_active = True

            # Update trailing stop
            if trailing_active:
                trail_stop = best_price - trail_distance
                stop = max(stop, trail_stop)

            # Check stop
            if bar["Low"] <= stop:
                pnl = stop - entry
                return {"result": "trail_stop" if trailing_active else "loss", "pnl": pnl}

        else:  # short
            if bar["Low"] < best_price:
                best_price = bar["Low"]

            if not trailing_active and (entry - best_price) >= trail_activate:
                trailing_active = True

            if trailing_active:
                trail_stop = best_price + trail_distance
                stop = min(stop, trail_stop)

            if bar["High"] >= stop:
                pnl = entry - stop
                return {"result": "trail_stop" if trailing_active else "loss", "pnl": pnl}

    # EOD
    last_close = setup.bars_after[-1]["Close"] if setup.bars_after else entry
    pnl = (last_close - entry) if setup.direction == "long" else (entry - last_close)
    return {"result": "eod", "pnl": pnl}


# ---------------------------------------------------------------------------
# ANALYSIS & REPORTING
# ---------------------------------------------------------------------------
def analyze_setups(setups: list) -> None:
    """Comprehensive analysis of all setups."""
    if not setups:
        print("No setups found!")
        return

    print(f"\n{'='*80}")
    print(f"OPENING RANGE MIDPOINT REACTION BACKTEST RESULTS")
    print(f"{'='*80}")
    print(f"Total setups found (min 3pt excursion): {len(setups)}")
    print(f"  Long setups (price broke below OR, re-entered up): {sum(1 for s in setups if s.direction == 'long')}")
    print(f"  Short setups (price broke above OR, re-entered down): {sum(1 for s in setups if s.direction == 'short')}")

    # ==========================================
    # QUESTION 1: Excursion threshold analysis
    # ==========================================
    print(f"\n{'='*80}")
    print("QUESTION 1: HOW FAR MUST PRICE LEAVE THE OR?")
    print("(Hit rate = % of setups where price reached the midpoint)")
    print(f"{'='*80}")

    print(f"\n{'Excursion':>12} {'Setups':>8} {'Hit Mid':>10} {'Hit Rate':>10} {'Hit 6T Before':>15} {'6T Rate':>10} {'Avg MFE':>10} {'Avg MAE':>10} {'Avg Reaction':>14}")
    print("-" * 110)

    for thresh in EXCURSION_THRESHOLDS:
        filtered = [s for s in setups if s.excursion_points >= thresh]
        if not filtered:
            continue
        n = len(filtered)
        hit_mid = sum(1 for s in filtered if s.hit_midpoint)
        hit_6t = sum(1 for s in filtered if s.hit_6ticks_before_mid)
        avg_mfe = np.mean([s.max_favorable for s in filtered])
        avg_mae = np.mean([s.max_adverse for s in filtered])
        avg_react = np.mean([s.midpoint_reaction_size for s in filtered])

        print(f"{thresh:>10.0f}pt {n:>8} {hit_mid:>10} {hit_mid/n*100:>9.1f}% {hit_6t:>15} {hit_6t/n*100:>9.1f}% {avg_mfe:>9.2f} {avg_mae:>9.2f} {avg_react:>13.2f}")

    # By direction
    for direction in ["long", "short"]:
        dir_label = "LONG (broke below, re-enter up)" if direction == "long" else "SHORT (broke above, re-enter down)"
        print(f"\n  --- {dir_label} ---")
        print(f"  {'Excursion':>12} {'Setups':>8} {'Hit Mid':>10} {'Hit Rate':>10}")
        print("  " + "-" * 50)

        for thresh in EXCURSION_THRESHOLDS:
            filtered = [s for s in setups if s.excursion_points >= thresh and s.direction == direction]
            if not filtered:
                continue
            n = len(filtered)
            hit_mid = sum(1 for s in filtered if s.hit_midpoint)
            print(f"  {thresh:>10.0f}pt {n:>8} {hit_mid:>10} {hit_mid/n*100:>9.1f}%")

    # ==========================================
    # QUESTION 2: OTHER FACTORS
    # ==========================================
    print(f"\n{'='*80}")
    print("QUESTION 2: WHAT OTHER FACTORS INFLUENCE MIDPOINT REACTION?")
    print(f"{'='*80}")

    # Use 7pt threshold as baseline
    base_setups = [s for s in setups if s.excursion_points >= 7]
    if not base_setups:
        base_setups = setups

    # Factor: OR Size
    print("\n  A) Opening Range Size")
    or_sizes = [s.or_size for s in base_setups]
    median_or = np.median(or_sizes)
    small_or = [s for s in base_setups if s.or_size <= median_or]
    large_or = [s for s in base_setups if s.or_size > median_or]

    if small_or:
        sr_small = sum(1 for s in small_or if s.hit_midpoint) / len(small_or) * 100
        print(f"     Small OR (≤{median_or:.1f}pt): {len(small_or)} setups, {sr_small:.1f}% hit midpoint")
    if large_or:
        sr_large = sum(1 for s in large_or if s.hit_midpoint) / len(large_or) * 100
        print(f"     Large OR (>{median_or:.1f}pt): {len(large_or)} setups, {sr_large:.1f}% hit midpoint")

    # Factor: OR Close Location
    print("\n  B) Where OR Closed Within Range (0=low, 1=high)")
    closed_low = [s for s in base_setups if s.or_close_location < 0.33]
    closed_mid = [s for s in base_setups if 0.33 <= s.or_close_location <= 0.67]
    closed_high = [s for s in base_setups if s.or_close_location > 0.67]

    for label, group in [("Closed near Low", closed_low), ("Closed near Mid", closed_mid), ("Closed near High", closed_high)]:
        if group:
            sr = sum(1 for s in group if s.hit_midpoint) / len(group) * 100
            print(f"     {label}: {len(group)} setups, {sr:.1f}% hit midpoint")

    # Factor: Time of re-entry
    print("\n  C) Time of Re-entry")
    import datetime
    time_buckets = [
        ("9:45-10:30", datetime.time(9, 45), datetime.time(10, 30)),
        ("10:30-11:30", datetime.time(10, 30), datetime.time(11, 30)),
        ("11:30-13:00", datetime.time(11, 30), datetime.time(13, 0)),
        ("13:00-14:30", datetime.time(13, 0), datetime.time(14, 30)),
        ("14:30-16:00", datetime.time(14, 30), datetime.time(16, 0)),
    ]
    for label, t_start, t_end in time_buckets:
        bucket = [s for s in base_setups if t_start <= s.time_of_reentry < t_end]
        if bucket:
            sr = sum(1 for s in bucket if s.hit_midpoint) / len(bucket) * 100
            avg_react = np.mean([s.midpoint_reaction_size for s in bucket])
            print(f"     {label}: {len(bucket)} setups, {sr:.1f}% hit mid, avg reaction {avg_react:.2f}pt")

    # Factor: Bars spent outside OR (speed of re-entry)
    print("\n  D) Speed of Re-entry (bars spent outside OR)")
    bar_counts = [s.excursion_bar_count for s in base_setups]
    median_bars = np.median(bar_counts)
    fast_reentry = [s for s in base_setups if s.excursion_bar_count <= median_bars]
    slow_reentry = [s for s in base_setups if s.excursion_bar_count > median_bars]

    if fast_reentry:
        sr = sum(1 for s in fast_reentry if s.hit_midpoint) / len(fast_reentry) * 100
        print(f"     Fast (≤{median_bars:.0f} bars outside): {len(fast_reentry)} setups, {sr:.1f}% hit midpoint")
    if slow_reentry:
        sr = sum(1 for s in slow_reentry if s.hit_midpoint) / len(slow_reentry) * 100
        print(f"     Slow (>{median_bars:.0f} bars outside): {len(slow_reentry)} setups, {sr:.1f}% hit midpoint")

    # Factor: Excursion relative to OR size
    print("\n  E) Excursion as % of OR Size")
    for s in base_setups:
        s._exc_pct = s.excursion_points / s.or_size * 100 if s.or_size > 0 else 0

    pct_buckets = [(50, 100), (100, 150), (150, 200), (200, 500)]
    for pct_low, pct_high in pct_buckets:
        bucket = [s for s in base_setups if pct_low <= s._exc_pct < pct_high]
        if bucket:
            sr = sum(1 for s in bucket if s.hit_midpoint) / len(bucket) * 100
            print(f"     {pct_low}-{pct_high}% of OR: {len(bucket)} setups, {sr:.1f}% hit midpoint")

    # ==========================================
    # QUESTION 3: OPTIMAL STOP LOSS & TAKE PROFIT
    # ==========================================
    print(f"\n{'='*80}")
    print("QUESTION 3: OPTIMAL STOP LOSS & TAKE PROFIT (7pt+ excursion)")
    print(f"{'='*80}")

    if not base_setups:
        print("  No setups with sufficient excursion")
        return

    print(f"\n  Fixed SL/TP Grid (showing: win%, avg PnL per trade, total PnL, profit factor)")
    print(f"  {'':>8}", end="")
    for tp in TARGET_LEVELS:
        print(f"  TP={tp:>4.1f}pt", end="")
    print()

    best_combo = None
    best_avg_pnl = -999

    for sl in STOP_LEVELS:
        print(f"  SL={sl:>4.1f}", end="")
        for tp in TARGET_LEVELS:
            results = [simulate_fixed_sl_tp(s, sl, tp) for s in base_setups]
            wins = sum(1 for r in results if r["result"] == "win")
            total_pnl = sum(r["pnl"] for r in results)
            avg_pnl = total_pnl / len(results)
            win_rate = wins / len(results) * 100

            gross_profit = sum(r["pnl"] for r in results if r["pnl"] > 0)
            gross_loss = abs(sum(r["pnl"] for r in results if r["pnl"] < 0))
            pf = gross_profit / gross_loss if gross_loss > 0 else 99.9

            if avg_pnl > best_avg_pnl:
                best_avg_pnl = avg_pnl
                best_combo = (sl, tp, win_rate, avg_pnl, total_pnl, pf, len(results))

            # Color code: show win rate
            print(f"  {win_rate:>5.0f}%/{avg_pnl:>+.1f}", end="")
        print()

    print(f"\n  BEST COMBO: SL={best_combo[0]:.1f}pt, TP={best_combo[1]:.1f}pt")
    print(f"    Win Rate: {best_combo[2]:.1f}%")
    print(f"    Avg PnL/trade: {best_combo[3]:+.2f}pt (${best_combo[3]*50:+.0f} per contract)")
    print(f"    Total PnL: {best_combo[4]:+.1f}pt (${best_combo[4]*50:+.0f} per contract)")
    print(f"    Profit Factor: {best_combo[5]:.2f}")
    print(f"    Trades: {best_combo[6]}")

    # ==========================================
    # QUESTION 4: TRAILING STOP ANALYSIS
    # ==========================================
    print(f"\n{'='*80}")
    print("QUESTION 4: TRAILING STOP PERFORMANCE (7pt+ excursion)")
    print(f"{'='*80}")

    # Use the best fixed SL as initial stop
    best_sl = best_combo[0] if best_combo else 2.0

    print(f"\n  Using initial SL = {best_sl:.1f}pt (best from fixed analysis)")
    print(f"  {'Activate':>10} {'Trail':>8} {'Win%':>8} {'Avg PnL':>10} {'Total PnL':>12} {'PF':>8}")
    print("  " + "-" * 60)

    best_trail = None
    best_trail_pnl = -999

    for activate, trail in TRAILING_CONFIGS:
        results = [simulate_trailing_stop(s, best_sl, activate, trail) for s in base_setups]
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

    print(f"\n  BEST TRAILING: Activate at +{best_trail[0]:.1f}pt, Trail {best_trail[1]:.1f}pt")
    print(f"    Win Rate: {best_trail[2]:.1f}%, Avg PnL: {best_trail[3]:+.2f}pt, PF: {best_trail[5]:.2f}")

    # Compare best fixed vs best trailing
    print(f"\n  --- COMPARISON ---")
    print(f"  Best Fixed  (SL={best_combo[0]:.1f}, TP={best_combo[1]:.1f}): avg {best_combo[3]:+.2f}pt/trade, PF {best_combo[5]:.2f}")
    print(f"  Best Trail  (Act={best_trail[0]:.1f}, Tr={best_trail[1]:.1f}): avg {best_trail[3]:+.2f}pt/trade, PF {best_trail[5]:.2f}")

    # ==========================================
    # QUESTION 5: 6-TICK EARLY ENTRY ANALYSIS
    # ==========================================
    print(f"\n{'='*80}")
    print("QUESTION 5: ENTRY AT MIDPOINT vs 6 TICKS (1.5pt) BEFORE MIDPOINT")
    print(f"{'='*80}")

    # Compare: entering at OR boundary (current) vs trying to enter near midpoint
    # This answers whether the reaction happens AT midpoint or 6 ticks before

    hit_6t_only = sum(1 for s in base_setups if s.hit_6ticks_before_mid and not s.hit_midpoint)
    hit_both = sum(1 for s in base_setups if s.hit_6ticks_before_mid and s.hit_midpoint)
    hit_mid_only = sum(1 for s in base_setups if s.hit_midpoint and not s.hit_6ticks_before_mid)
    hit_neither = sum(1 for s in base_setups if not s.hit_6ticks_before_mid and not s.hit_midpoint)

    print(f"\n  Of {len(base_setups)} setups (7pt+ excursion):")
    print(f"    Reached 6T before mid but NOT mid: {hit_6t_only} ({hit_6t_only/len(base_setups)*100:.1f}%)")
    print(f"    Reached both 6T zone AND mid:      {hit_both} ({hit_both/len(base_setups)*100:.1f}%)")
    print(f"    Reached mid (should include 6T):    {hit_mid_only} ({hit_mid_only/len(base_setups)*100:.1f}%)")
    print(f"    Reached neither:                    {hit_neither} ({hit_neither/len(base_setups)*100:.1f}%)")

    # Reaction size comparison
    setups_at_6t = [s for s in base_setups if s.hit_6ticks_before_mid]
    setups_at_mid = [s for s in base_setups if s.hit_midpoint]

    if setups_at_6t:
        avg_reaction_6t = np.mean([s.midpoint_reaction_size for s in setups_at_6t])
        print(f"\n  Avg reaction size when reaching 6T zone: {avg_reaction_6t:.2f}pt")
    if setups_at_mid:
        avg_reaction_mid = np.mean([s.midpoint_reaction_size for s in setups_at_mid])
        print(f"  Avg reaction size when reaching midpoint: {avg_reaction_mid:.2f}pt")

    # ==========================================
    # MULTI-LEVEL TP ANALYSIS
    # ==========================================
    print(f"\n{'='*80}")
    print("BONUS: SCALED EXIT ANALYSIS (Split targets)")
    print(f"{'='*80}")

    best_sl_val = best_combo[0]

    # Simulate: 50% at TP1, 50% at TP2 (with trailing)
    split_configs = [
        (1.5, 3.0, "TP1=1.5pt, TP2=3.0pt"),
        (2.0, 4.0, "TP1=2.0pt, TP2=4.0pt"),
        (1.5, 4.0, "TP1=1.5pt, TP2=4.0pt"),
        (2.0, 5.0, "TP1=2.0pt, TP2=5.0pt"),
        (1.0, 3.0, "TP1=1.0pt, TP2=3.0pt"),
        (2.0, 6.0, "TP1=2.0pt, TP2=6.0pt (trail 2pt after +3)"),
    ]

    print(f"\n  Using SL={best_sl_val:.1f}pt, 50% at TP1, 50% at TP2")
    print(f"  (TP2 uses trailing stop after activation)")
    print(f"  {'Config':>35} {'Avg PnL':>10} {'Total PnL':>12} {'PF':>8}")
    print("  " + "-" * 70)

    for tp1, tp2, label in split_configs:
        total_pnl = 0
        gross_profit = 0
        gross_loss = 0

        for s in base_setups:
            # Half at fixed TP1
            r1 = simulate_fixed_sl_tp(s, best_sl_val, tp1)
            # Half with trailing toward TP2
            r2 = simulate_trailing_stop(s, best_sl_val, tp1, best_sl_val)

            trade_pnl = (r1["pnl"] + r2["pnl"]) / 2  # average of two halves
            total_pnl += trade_pnl
            if trade_pnl > 0:
                gross_profit += trade_pnl
            else:
                gross_loss += abs(trade_pnl)

        avg_pnl = total_pnl / len(base_setups)
        pf = gross_profit / gross_loss if gross_loss > 0 else 99.9
        print(f"  {label:>35} {avg_pnl:>+9.2f}pt {total_pnl:>+11.1f}pt {pf:>7.2f}")

    # ==========================================
    # SUMMARY TABLE BY DAY
    # ==========================================
    print(f"\n{'='*80}")
    print("DAILY BREAKDOWN (7pt+ excursion, best fixed SL/TP)")
    print(f"{'='*80}")

    sl, tp = best_combo[0], best_combo[1]
    daily_results = {}
    for s in base_setups:
        r = simulate_fixed_sl_tp(s, sl, tp)
        d = str(s.date)
        if d not in daily_results:
            daily_results[d] = {"trades": 0, "pnl": 0, "wins": 0}
        daily_results[d]["trades"] += 1
        daily_results[d]["pnl"] += r["pnl"]
        if r["pnl"] > 0:
            daily_results[d]["wins"] += 1

    # Show first and last 10 days
    sorted_days = sorted(daily_results.keys())
    cumulative = 0
    print(f"\n  {'Date':>12} {'Trades':>8} {'PnL':>10} {'Cumulative':>12}")
    print("  " + "-" * 45)

    for d in sorted_days:
        dr = daily_results[d]
        cumulative += dr["pnl"]
        print(f"  {d:>12} {dr['trades']:>8} {dr['pnl']:>+9.2f}pt {cumulative:>+11.2f}pt")

    print(f"\n  Total days with trades: {len(daily_results)}")
    print(f"  Winning days: {sum(1 for d in daily_results.values() if d['pnl'] > 0)}")
    print(f"  Losing days: {sum(1 for d in daily_results.values() if d['pnl'] < 0)}")
    print(f"  Max daily gain: {max(d['pnl'] for d in daily_results.values()):+.2f}pt")
    print(f"  Max daily loss: {min(d['pnl'] for d in daily_results.values()):+.2f}pt")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data(PARQUET_PATH)
    or_ranges = build_opening_ranges(df)

    print(f"\n{'='*80}")
    print("SCANNING FOR SETUPS...")
    print(f"{'='*80}")

    # Find setups with minimum 3pt excursion (we'll filter higher later)
    setups = find_setups(df, or_ranges, min_excursion=3.0)
    print(f"Found {len(setups)} total setups (3pt+ excursion)")

    # Run full analysis
    analyze_setups(setups)

    print(f"\n{'='*80}")
    print("BACKTEST COMPLETE")
    print(f"{'='*80}")
