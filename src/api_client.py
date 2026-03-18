"""
API client for the ES Backtest Data Server running on the Windows VM.

Handles all communication with the FastAPI server and converts responses
into pandas DataFrames ready for analysis.

Usage:
    from src.api_client import ESDataClient

    client = ESDataClient("http://10.211.55.3:8000")
    bars = client.get_bars(start_date="2025-06-01", session="rth")
    bars_with_indicators = client.get_indicators(sma_period=80)
"""
import json
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


DEFAULT_SERVER = "http://10.211.55.3:8000"
CACHE_DIR = Path(__file__).parent.parent / "data_cache"


class ESDataClient:
    """Client for the ES Backtest Data Server."""

    def __init__(self, base_url: str = None, timeout: int = 120):
        self.base_url = (base_url or os.environ.get("ES_SERVER_URL", DEFAULT_SERVER)).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ---- Connection ----

    def health_check(self) -> dict:
        """Check if the server is reachable."""
        try:
            r = self._session.get(f"{self.base_url}/", timeout=5)
            r.raise_for_status()
            return r.json()
        except requests.ConnectionError:
            return {"status": "error", "message": f"Cannot reach {self.base_url}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def check_data(self) -> dict:
        """Get data summary from the server."""
        r = self._session.get(f"{self.base_url}/check-data", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # ---- Bar Data ----

    def get_bars(
        self,
        start_date: str = None,
        end_date: str = None,
        session: str = "all",
        limit: int = None,
    ) -> pd.DataFrame:
        """
        Fetch 610-tick bar data from the server.

        Returns a DataFrame with columns matching your parquet:
        StartTime, EndTime, Open, High, Low, Close, Volume, etc.
        """
        payload = {
            "start_date": start_date,
            "end_date": end_date,
            "session": session,
        }
        if limit:
            payload["limit"] = limit

        r = self._session.post(
            f"{self.base_url}/bars",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()

        if data.get("rows", 0) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(data["data"])
        df = _normalize_columns(df)
        return df

    def get_indicators(
        self,
        start_date: str = None,
        end_date: str = None,
        session: str = "rth",
        sma_period: int = 80,
        kc_ema_period: int = 20,
        kc_atr_period: int = 10,
        kc_atr_multiplier: float = 1.5,
    ) -> pd.DataFrame:
        """
        Fetch bars with pre-computed SMA and Keltner Channel from the server.

        Returns DataFrame with: OHLCV + SMA, KC_Mid, KC_Upper, KC_Lower, ATR, signals.
        """
        payload = {
            "start_date": start_date,
            "end_date": end_date,
            "session": session,
            "sma_period": sma_period,
            "kc_ema_period": kc_ema_period,
            "kc_atr_period": kc_atr_period,
            "kc_atr_multiplier": kc_atr_multiplier,
        }

        r = self._session.post(
            f"{self.base_url}/indicators",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()

        if data.get("rows", 0) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(data["data"])
        df = _normalize_columns(df)
        return df

    def query(
        self,
        start_date: str = None,
        end_date: str = None,
        session: str = "all",
        resample: str = None,
        columns: list[str] = None,
    ) -> pd.DataFrame:
        """Flexible query with optional resampling and column selection."""
        payload = {
            "start_date": start_date,
            "end_date": end_date,
            "session": session,
            "resample": resample,
            "columns": columns,
        }

        r = self._session.post(
            f"{self.base_url}/query",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()

        if data.get("rows", 0) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(data["data"])
        df = _normalize_columns(df)
        return df

    # ---- Download & Cache ----

    def download_parquet(self, save_path: str = None) -> str:
        """
        Download the full parquet file for local analysis.
        Caches locally so subsequent runs don't re-download.
        """
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = Path(save_path) if save_path else CACHE_DIR / "ES_610tick_ALL.parquet"

        if path.exists():
            age_hours = (time.time() - path.stat().st_mtime) / 3600
            if age_hours < 24:
                return str(path)

        r = self._session.get(
            f"{self.base_url}/download",
            timeout=self.timeout,
            stream=True,
        )
        r.raise_for_status()

        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        return str(path)

    def load_local(self, path: str = None) -> pd.DataFrame:
        """Load from cached parquet file (downloads if needed)."""
        parquet_path = path or self.download_parquet()
        df = pd.read_parquet(parquet_path)
        df = _normalize_columns(df)
        return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names from server format to our analysis format.

    Server uses: StartTime, EndTime, Open, High, Low, Close, Volume
    Our system uses: timestamp, open, high, low, close, volume
    This function adds lowercase aliases so both conventions work.
    """
    df = df.copy()

    # Parse datetime columns
    for col in ["StartTime", "starttime", "start_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df["timestamp"] = df[col]
            break

    for col in ["EndTime", "endtime", "end_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df["bar_open_time"] = df[col]
            break

    # Create lowercase aliases for OHLCV
    rename_map = {}
    for src, dst in [("Open", "open"), ("High", "high"), ("Low", "low"),
                     ("Close", "close"), ("Volume", "volume")]:
        if src in df.columns and dst not in df.columns:
            rename_map[src] = dst

    if rename_map:
        # Keep originals and add lowercase versions
        for src, dst in rename_map.items():
            df[dst] = df[src]

    # Indicator columns (from /indicators endpoint)
    indicator_map = {
        "SMA": "sma", "KC_Mid": "kc_mid", "KC_Upper": "kc_upper",
        "KC_Lower": "kc_lower", "ATR": "atr",
        "Above_SMA": "above_sma", "Above_KC_Upper": "above_kc_upper",
        "Below_KC_Lower": "below_kc_lower",
    }
    for src, dst in indicator_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    # Compute derived fields if we have OHLC
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        if "bar_range" not in df.columns:
            df["bar_range"] = df["high"] - df["low"]
        if "bar_body" not in df.columns:
            df["bar_body"] = df["close"] - df["open"]
        if "is_green" not in df.columns:
            df["is_green"] = df["close"] > df["open"]
        if "is_red" not in df.columns:
            df["is_red"] = df["close"] < df["open"]

    # SMA derived fields
    if "sma" in df.columns and "close" in df.columns:
        if "above_sma" not in df.columns:
            df["above_sma"] = df["close"] > df["sma"]
        if "below_sma" not in df.columns:
            df["below_sma"] = df["close"] < df["sma"]
        if "sma_distance" not in df.columns:
            df["sma_distance"] = df["close"] - df["sma"]
        if "sma_distance_pct" not in df.columns:
            df["sma_distance_pct"] = df["sma_distance"] / df["sma"] * 100

    # KC derived fields
    if all(c in df.columns for c in ["kc_upper", "kc_lower", "close"]):
        if "inside_kc" not in df.columns:
            df["inside_kc"] = (df["close"] <= df["kc_upper"]) & (df["close"] >= df["kc_lower"])

    return df
