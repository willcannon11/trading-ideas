"""
Trade Calendar Web App
======================

Interactive calendar showing daily P&L with clickable days to view
individual trades and apply tags.

Start with:
  pip install fastapi uvicorn pandas pyarrow jinja2
  cd ~/trading-ideas
  python calendar_app.py

Then open http://localhost:8050 in your browser.
"""
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI(title="Trade Calendar")
templates = Jinja2Templates(directory="templates")

# --- Data paths ---
DATA_DIR = Path(__file__).parent
PARQUET_PATH = DATA_DIR / "es_data.parquet"
TAGS_PATH = DATA_DIR / "trade_tags.json"

# --- In-memory stores ---
TRADES_DF: pd.DataFrame = pd.DataFrame()
TAGS: dict[str, list[str]] = {}  # trade_id -> [tag1, tag2, ...]


def _load_tags():
    """Load tags from disk."""
    global TAGS
    if TAGS_PATH.exists():
        TAGS = json.loads(TAGS_PATH.read_text())


def _save_tags():
    """Persist tags to disk."""
    TAGS_PATH.write_text(json.dumps(TAGS, indent=2))


def _generate_trades_from_data() -> pd.DataFrame:
    """
    Load trade data from CSV exports in the output/ directory, or fall back
    to synthetic data for demo purposes.

    To use real data, run a backtest first and place the *_trades.csv file
    in the output/ directory.
    """
    output_dir = DATA_DIR / "output"
    if output_dir.exists():
        csvs = list(output_dir.glob("*_trades.csv"))
        if csvs:
            frames = []
            for csv_path in csvs:
                try:
                    df = pd.read_csv(csv_path, parse_dates=["entry_time", "exit_time"])
                    if "trade_date" not in df.columns:
                        df["trade_date"] = df["entry_time"].dt.date
                    frames.append(df)
                except Exception as e:
                    print(f"  Skipping {csv_path.name}: {e}")
            if frames:
                combined = pd.concat(frames, ignore_index=True)
                print(f"  Loaded {len(combined)} trades from {len(frames)} CSV files")
                return combined

    # No CSV exports found — generate synthetic trades for demo
    return _generate_synthetic_trades()


def _generate_synthetic_trades() -> pd.DataFrame:
    """Generate synthetic trade data for demonstration."""
    print("  Generating synthetic trade data for calendar demo...")
    np.random.seed(42)

    records = []
    # Generate trades for the current month and previous months
    today = date.today()
    start_date = date(today.year, today.month, 1) - timedelta(days=90)

    trade_id = 0
    current = start_date
    while current <= today:
        # Skip weekends
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        # Random number of trades per day (0-15)
        n_trades = np.random.choice([0, 2, 3, 4, 5, 6, 7, 8, 10, 11], p=[0.05, 0.05, 0.1, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05])

        for j in range(n_trades):
            direction = np.random.choice(["long", "short"])
            entry_hour = np.random.randint(9, 16)
            entry_min = np.random.randint(0, 60)
            entry_time = datetime(current.year, current.month, current.day, entry_hour, entry_min)
            exit_time = entry_time + timedelta(minutes=np.random.randint(2, 120))

            entry_price = 5800 + np.random.uniform(-100, 100)
            pnl_points = np.random.choice(
                [-5, -4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4, 5, 7],
                p=[0.02, 0.03, 0.05, 0.1, 0.08, 0.07, 0.05, 0.05, 0.07, 0.08, 0.1, 0.1, 0.08, 0.07, 0.05],
            )

            if direction == "long":
                exit_price = entry_price + pnl_points
            else:
                exit_price = entry_price - pnl_points

            pnl_dollars = pnl_points * 50.0 * 2 - 10  # 2 contracts, $10 RT cost

            exit_reason = "target" if pnl_points > 0 else "stop"
            signal = np.random.choice([
                "sma_keltner_bounce", "keltner_breakout",
                "keltner_mean_reversion", "trend_continuation",
                "kc_squeeze_breakout",
            ])

            records.append({
                "entry_bar": trade_id * 10,
                "exit_bar": trade_id * 10 + np.random.randint(1, 20),
                "direction": direction,
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "entry_time": entry_time,
                "exit_time": exit_time,
                "stop_price": round(entry_price - 2 if direction == "long" else entry_price + 2, 2),
                "target_price": round(entry_price + 2 if direction == "long" else entry_price - 2, 2),
                "exit_reason": exit_reason,
                "pnl_points": pnl_points,
                "pnl_dollars": round(pnl_dollars, 2),
                "contracts": 2,
                "signal_name": signal,
                "is_winner": pnl_points > 0,
                "rr_actual": round(abs(pnl_points) / 2.0, 2),
            })
            trade_id += 1

        current += timedelta(days=1)

    df = pd.DataFrame(records)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["trade_date"] = df["entry_time"].dt.date
    df["duration_bars"] = df["exit_bar"] - df["entry_bar"]
    df["cumulative_pnl"] = df["pnl_dollars"].cumsum()
    print(f"  Generated {len(df)} synthetic trades")
    return df


@app.on_event("startup")
def startup():
    global TRADES_DF
    print("Loading trade data...")
    TRADES_DF = _generate_trades_from_data()
    _load_tags()
    print(f"Ready: {len(TRADES_DF)} trades loaded, {len(TAGS)} tagged trades")


# --- Pages ---

@app.get("/", response_class=HTMLResponse)
async def calendar_page(request: Request):
    return templates.TemplateResponse("calendar.html", {"request": request})


# --- API ---

@app.get("/api/calendar/{year}/{month}")
async def get_month_data(year: int, month: int, mode: str = "gross"):
    """Return daily P&L summary for a given month."""
    df = TRADES_DF.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # Filter to requested month
    mask = (df["trade_date"].dt.year == year) & (df["trade_date"].dt.month == month)
    month_df = df[mask]

    pnl_col = "pnl_dollars"  # gross and net use same column in backtest data

    daily = {}
    if not month_df.empty:
        grouped = month_df.groupby(month_df["trade_date"].dt.day)
        for day, group in grouped:
            daily[int(day)] = {
                "pnl": round(float(group[pnl_col].sum()), 2),
                "trades": int(len(group)),
            }

    total_pnl = round(float(month_df[pnl_col].sum()), 2) if not month_df.empty else 0
    total_trades = int(len(month_df))

    return {
        "year": year,
        "month": month,
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "days": daily,
    }


@app.get("/api/trades/{year}/{month}/{day}")
async def get_day_trades(year: int, month: int, day: int):
    """Return individual trades for a specific day."""
    target = date(year, month, day)
    df = TRADES_DF.copy()
    df["trade_date_obj"] = pd.to_datetime(df["trade_date"]).dt.date
    day_df = df[df["trade_date_obj"] == target]

    trades = []
    for idx, row in day_df.iterrows():
        trade_id = f"{row['entry_time'].isoformat()}_{row['direction']}_{row['entry_price']}"
        trades.append({
            "id": trade_id,
            "direction": row["direction"],
            "entry_time": row["entry_time"].strftime("%H:%M:%S"),
            "exit_time": row["exit_time"].strftime("%H:%M:%S"),
            "entry_price": round(float(row["entry_price"]), 2),
            "exit_price": round(float(row["exit_price"]), 2),
            "pnl_dollars": round(float(row["pnl_dollars"]), 2),
            "pnl_points": round(float(row["pnl_points"]), 2),
            "exit_reason": row["exit_reason"],
            "signal_name": row["signal_name"],
            "contracts": int(row["contracts"]),
            "is_winner": bool(row["is_winner"]),
            "tags": TAGS.get(trade_id, []),
        })

    total_pnl = round(float(day_df["pnl_dollars"].sum()), 2) if not day_df.empty else 0

    return {
        "date": str(target),
        "total_pnl": total_pnl,
        "trades": trades,
    }


@app.post("/api/trades/{trade_id}/tags")
async def update_tags(trade_id: str, request: Request):
    """Add or remove tags for a trade."""
    body = await request.json()
    action = body.get("action", "add")
    tag = body.get("tag", "").strip()

    if not tag:
        return JSONResponse({"error": "Tag is required"}, status_code=400)

    if trade_id not in TAGS:
        TAGS[trade_id] = []

    if action == "add":
        if tag not in TAGS[trade_id]:
            TAGS[trade_id].append(tag)
    elif action == "remove":
        if tag in TAGS[trade_id]:
            TAGS[trade_id].remove(tag)
    elif action == "set":
        TAGS[trade_id] = body.get("tags", [])

    _save_tags()
    return {"trade_id": trade_id, "tags": TAGS[trade_id]}


@app.get("/api/tags")
async def get_all_tags():
    """Return all unique tags used across all trades."""
    all_tags = set()
    for tags in TAGS.values():
        all_tags.update(tags)
    return {"tags": sorted(all_tags)}


if __name__ == "__main__":
    uvicorn.run("calendar_app:app", host="0.0.0.0", port=8050, reload=True)
