"""
Upgraded FastAPI server for ES 610-tick bar data.

DEPLOY THIS ON YOUR WINDOWS MACHINE — replace C:\\backtest\\server.py

Endpoints:
  GET  /                 Health check
  GET  /check-data       Data summary (row count, columns, date range)
  POST /bars             Return actual bar data as JSON (with date/session filters)
  POST /bars/csv         Return bar data as downloadable CSV
  GET  /download         Download the full parquet file
  POST /run-research     Original summary endpoint (kept for compatibility)
  POST /indicators       Return bars with computed indicators (SMA, Keltner)
  POST /query            Flexible query: filter + aggregate in one call

Start with:
  pip install fastapi uvicorn pandas pyarrow
  cd C:\\backtest
  uvicorn server:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
import json

app = FastAPI(title="ES Backtest Data Server", version="2.0")

DATA_PATH = r"C:\backtest\ES_610tick_ALL.parquet"


# ---- Request Models ----

class DateFilter(BaseModel):
    start_date: str | None = None   # "2025-03-01"
    end_date: str | None = None     # "2025-09-30"
    session: str = "all"            # "all" or "rth"
    limit: int | None = None        # max rows to return


class IndicatorRequest(DateFilter):
    sma_period: int = 80
    kc_ema_period: int = 20
    kc_atr_period: int = 10
    kc_atr_multiplier: float = 1.5


class QueryRequest(DateFilter):
    # Aggregation
    resample: str | None = None     # e.g. "1h", "1D" for time-based resampling
    columns: list[str] | None = None  # subset of columns to return


class ResearchRequest(BaseModel):
    """Original model kept for compatibility."""
    start_date: str | None = None
    end_date: str | None = None
    session: str = "all"


# ---- Helpers ----

def _load(req: DateFilter | None = None) -> pd.DataFrame:
    """Load parquet and apply date/session filters."""
    df = pd.read_parquet(DATA_PATH).copy()
    df["StartTime"] = pd.to_datetime(df["StartTime"])
    df["EndTime"] = pd.to_datetime(df["EndTime"])

    if req:
        if req.start_date:
            df = df[df["StartTime"] >= pd.to_datetime(req.start_date)]
        if req.end_date:
            df = df[df["StartTime"] <= pd.to_datetime(req.end_date) + pd.Timedelta(days=1)]
        if req.session.lower() == "rth":
            t = df["StartTime"].dt.time
            df = df[(t >= pd.to_datetime("09:30:00").time()) &
                    (t <= pd.to_datetime("16:00:00").time())]
        if req.limit:
            df = df.head(req.limit)

    return df


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to JSON-serializable records."""
    df = df.copy()
    for col in df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        df[col] = df[col].astype(str)
    return df.to_dict(orient="records")


def _compute_indicators(df: pd.DataFrame, sma_period=80, kc_ema=20, kc_atr=10, kc_mult=1.5) -> pd.DataFrame:
    """Add SMA and Keltner Channel columns to bar data."""
    df = df.copy()

    # SMA on Close
    df["SMA"] = df["Close"].rolling(window=sma_period, min_periods=sma_period).mean()

    # ATR
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=kc_atr, min_periods=kc_atr).mean()

    # Keltner Channel
    df["KC_Mid"] = df["Close"].ewm(span=kc_ema, adjust=False).mean()
    df["KC_Upper"] = df["KC_Mid"] + kc_mult * atr
    df["KC_Lower"] = df["KC_Mid"] - kc_mult * atr
    df["ATR"] = atr

    # Derived signals
    df["Above_SMA"] = df["Close"] > df["SMA"]
    df["Above_KC_Upper"] = df["Close"] > df["KC_Upper"]
    df["Below_KC_Lower"] = df["Close"] < df["KC_Lower"]

    return df


# ---- Endpoints ----

@app.get("/")
def home():
    return {"status": "ok", "message": "ES Backtest Data Server v2.0"}


@app.get("/check-data")
def check_data():
    df = pd.read_parquet(DATA_PATH)
    return {
        "status": "ok",
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "first_time": str(df["StartTime"].iloc[0]),
        "last_time": str(df["StartTime"].iloc[-1]),
        "size_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1),
    }


@app.post("/bars")
def get_bars(req: DateFilter):
    """Return filtered bar data as JSON records."""
    df = _load(req)
    if len(df) == 0:
        return {"status": "ok", "rows": 0, "data": []}
    return {
        "status": "ok",
        "rows": len(df),
        "data": _df_to_records(df),
    }


@app.post("/bars/csv")
def get_bars_csv(req: DateFilter):
    """Return filtered bar data as downloadable CSV."""
    df = _load(req)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=es_bars.csv"},
    )


@app.get("/download")
def download_parquet():
    """Download the full parquet file."""
    return FileResponse(
        DATA_PATH,
        media_type="application/octet-stream",
        filename="ES_610tick_ALL.parquet",
    )


@app.post("/indicators")
def get_indicators(req: IndicatorRequest):
    """Return bars with computed SMA and Keltner Channel indicators."""
    df = _load(req)
    if len(df) == 0:
        return {"status": "ok", "rows": 0, "data": []}

    df = _compute_indicators(
        df,
        sma_period=req.sma_period,
        kc_ema=req.kc_ema_period,
        kc_atr=req.kc_atr_period,
        kc_mult=req.kc_atr_multiplier,
    )

    return {
        "status": "ok",
        "rows": len(df),
        "data": _df_to_records(df),
    }


@app.post("/query")
def flexible_query(req: QueryRequest):
    """Flexible query with optional column selection and time resampling."""
    df = _load(req)
    if len(df) == 0:
        return {"status": "ok", "rows": 0, "data": []}

    # Optional time-based resampling (e.g., "1h", "1D")
    if req.resample:
        df = df.set_index("StartTime")
        resampled = df.resample(req.resample).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna().reset_index()
        resampled["EndTime"] = resampled["StartTime"]  # placeholder
        df = resampled

    # Column subset
    if req.columns:
        valid = [c for c in req.columns if c in df.columns]
        if valid:
            df = df[valid]

    return {
        "status": "ok",
        "rows": len(df),
        "data": _df_to_records(df),
    }


@app.post("/run-research")
def run_research(req: ResearchRequest):
    """Original endpoint kept for backward compatibility."""
    df = _load(DateFilter(
        start_date=req.start_date,
        end_date=req.end_date,
        session=req.session,
    ))
    if len(df) == 0:
        return {"status": "ok", "rows": 0, "message": "No data matched your filters"}

    return {
        "status": "ok",
        "rows": int(len(df)),
        "start_time": str(df["StartTime"].iloc[0]),
        "end_time": str(df["StartTime"].iloc[-1]),
        "open_mean": float(df["Open"].mean()),
        "close_mean": float(df["Close"].mean()),
        "high_max": float(df["High"].max()),
        "low_min": float(df["Low"].min()),
        "volume_sum": int(df["Volume"].sum()),
    }
