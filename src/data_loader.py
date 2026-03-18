"""
Load and normalize NinjaTrader tick data exports for ES futures.

Supports NinjaTrader's standard export formats:
  - .txt/.csv tick exports (timestamp, price, volume)
  - Market Replay tick data
"""
import pathlib
from typing import Optional

import numpy as np
import pandas as pd


# NinjaTrader tick export column layouts
_COLUMN_MAPS = {
    # Standard NT export: Date;Time;Price;Volume
    "semicolon_4col": {
        "sep": ";",
        "names": ["date", "time", "price", "volume"],
        "parse": "_parse_semicolon_4col",
    },
    # CSV export: Timestamp,Price,Volume
    "csv_3col": {
        "sep": ",",
        "names": ["timestamp", "price", "volume"],
        "parse": "_parse_csv_3col",
    },
    # Full CSV: Date,Time,Open,High,Low,Close,Volume  (tick = OHLC identical)
    "csv_ohlcv": {
        "sep": ",",
        "names": ["date", "time", "open", "high", "low", "close", "volume"],
        "parse": "_parse_csv_ohlcv",
    },
    # Single datetime column CSV: DateTime,Price,Volume
    "csv_datetime_price_vol": {
        "sep": ",",
        "names": ["datetime", "price", "volume"],
        "parse": "_parse_csv_datetime_price_vol",
    },
}


def _sniff_format(path: pathlib.Path) -> dict:
    """Auto-detect the tick data format by reading the first few lines."""
    with open(path, "r") as f:
        lines = [f.readline() for _ in range(5)]

    first_data = lines[0].strip()

    # Check for header row
    has_header = False
    lower = first_data.lower()
    if any(kw in lower for kw in ["date", "time", "price", "open", "close", "volume"]):
        has_header = True
        first_data = lines[1].strip() if len(lines) > 1 else first_data

    if ";" in first_data:
        parts = first_data.split(";")
        if len(parts) == 4:
            return {**_COLUMN_MAPS["semicolon_4col"], "has_header": has_header}

    parts = first_data.split(",")
    if len(parts) == 7:
        return {**_COLUMN_MAPS["csv_ohlcv"], "has_header": has_header}
    elif len(parts) == 3:
        # Distinguish: is first field a full datetime or just a date?
        field = parts[0].strip()
        if " " in field or "T" in field:
            return {**_COLUMN_MAPS["csv_datetime_price_vol"], "has_header": has_header}
        else:
            return {**_COLUMN_MAPS["csv_3col"], "has_header": has_header}

    # Fallback: try comma-separated with timestamp,price,volume
    return {**_COLUMN_MAPS["csv_3col"], "has_header": has_header}


def _parse_semicolon_4col(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df[["timestamp", "price", "volume"]].copy()
    return df


def _parse_csv_3col(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[["timestamp", "price", "volume"]].copy()
    return df


def _parse_csv_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
    df["price"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(int)
    df = df[["timestamp", "price", "volume"]].copy()
    return df


def _parse_csv_datetime_price_vol(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["datetime"])
    df = df[["timestamp", "price", "volume"]].copy()
    return df


_PARSERS = {
    "_parse_semicolon_4col": _parse_semicolon_4col,
    "_parse_csv_3col": _parse_csv_3col,
    "_parse_csv_ohlcv": _parse_csv_ohlcv,
    "_parse_csv_datetime_price_vol": _parse_csv_datetime_price_vol,
}


def load_ticks(
    path: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    session_start: str = "09:30",
    session_end: str = "16:00",
    filter_rth: bool = True,
) -> pd.DataFrame:
    """
    Load tick data from a NinjaTrader export file.

    Returns a DataFrame with columns: timestamp, price, volume
    sorted by timestamp, optionally filtered to RTH session.

    Parameters
    ----------
    path : str
        Path to tick data file (.csv or .txt)
    start, end : str, optional
        Date strings to filter the data range (e.g. '2025-01-01')
    session_start, session_end : str
        RTH session boundaries (HH:MM format, Eastern time)
    filter_rth : bool
        If True, only keep ticks within the RTH session window.
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Tick data file not found: {path}")

    fmt = _sniff_format(p)
    header_row = 0 if fmt["has_header"] else None

    df = pd.read_csv(
        p,
        sep=fmt["sep"],
        header=header_row,
        names=fmt["names"] if not fmt["has_header"] else None,
        low_memory=False,
    )

    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # If header was present but names differ, try to map them
    if fmt["has_header"]:
        col_map = {}
        for col in df.columns:
            cl = col.lower()
            if "date" in cl and "time" in cl:
                col_map[col] = "datetime"
            elif "date" in cl:
                col_map[col] = "date"
            elif "time" in cl:
                col_map[col] = "time"
            elif "price" in cl or "last" in cl or "close" in cl:
                col_map[col] = "price"
            elif "vol" in cl:
                col_map[col] = "volume"
            elif "open" in cl:
                col_map[col] = "open"
            elif "high" in cl:
                col_map[col] = "high"
            elif "low" in cl:
                col_map[col] = "low"
        df = df.rename(columns=col_map)

    parser = _PARSERS[fmt["parse"]]
    df = parser(df)

    df["price"] = df["price"].astype(float)
    df["volume"] = df["volume"].astype(int)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Date range filter
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end)]

    # RTH session filter
    if filter_rth:
        t_start = pd.Timestamp(session_start).time()
        t_end = pd.Timestamp(session_end).time()
        mask = (df["timestamp"].dt.time >= t_start) & (df["timestamp"].dt.time <= t_end)
        df = df[mask].reset_index(drop=True)

    return df


def load_multiple(
    directory: str,
    pattern: str = "*.txt",
    **kwargs,
) -> pd.DataFrame:
    """Load and concatenate all tick files in a directory."""
    p = pathlib.Path(directory)
    files = sorted(p.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {directory}")

    frames = [load_ticks(str(f), **kwargs) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
