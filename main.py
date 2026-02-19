"""
ParmStockAdvisor — FastAPI Backend
Uses Financial Modeling Prep STABLE API endpoints.

Run locally:
    pip install fastapi uvicorn pandas numpy requests
    uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import requests

# ─── Financial Modeling Prep Config ──────────────────────────────────────────
FMP_API_KEY = "BDUFyoYbfR6jCYehbHXlT53Y7D8PIfur"
FMP_BASE = "https://financialmodelingprep.com/stable"

app = FastAPI(title="ParmStockAdvisor API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Data Fetchers ───────────────────────────────────────────────────────────

def fetch_daily(ticker: str) -> pd.DataFrame:
    url = f"{FMP_BASE}/historical-price-eod/full"
    params = {
        "symbol": ticker,
        "apikey": FMP_API_KEY,
    }

    r = requests.get(url, params=params, timeout=15)
    data = r.json()

    if "historical" not in data:
        raise ValueError(f"FMP error for '{ticker}': {data}")

    rows = []
    for item in data["historical"][:150]:
        rows.append({
            "Date":   pd.to_datetime(item["date"]),
            "Open":   float(item["open"]),
            "High":   float(item["high"]),
            "Low":    float(item["low"]),
            "Close":  float(item["close"]),
            "Volume": float(item["volume"]),
        })

    df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    return df


def fetch_quote(ticker: str) -> dict:
    url = f"{FMP_BASE}/quote"
    params = {
        "symbol": ticker,
        "apikey": FMP_API_KEY,
    }

    r = requests.get(url, params=params, timeout=15)
    data = r.json()
    return data[0] if isinstance(data, list) and data else {}


def fetch_overview(ticker: str) -> dict:
    url = f"{FMP_BASE}/profile"
    params = {
        "symbol": ticker,
        "apikey": FMP_API_KEY,
    }

    r = requests.get(url, params=params, timeout=15)
    data = r.json()
    return data[0] if isinstance(data, list) and data else {}


# ─── Technical Indicators ────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)


def compute_macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return (
        round(float(macd.iloc[-1]), 4),
        round(float(signal.iloc[-1]), 4),
        round(float(hist.iloc[-1]), 4),
    )


def compute_trend(series: pd.Series, days: int = 10) -> str:
    recent = series.tail(days)
    pct = (recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0] * 100
    if pct > 0.5: return "Uptrend"
    if pct < -0.5: return "Downtrend"
    return "Sideways"


def compute_bollinger(series: pd.Series, period: int = 20):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return {
        "upper":  round(float(upper.iloc[-1]), 2),
        "middle": round(float(sma.iloc[-1]), 2),
        "lower":  round(float(lower.iloc[-1]), 2),
    }


def score_to_signal(score: int) -> str:
    if score >= 3: return "STRONG BUY"
    if score == 2: return "BUY"
    if score == 1: return "WEAK BUY"
    if score == 0: return "HOLD"
    if score == -1: return "WEAK SELL"
    if score == -2: return "SELL"
    return "STRONG SELL"


# ─── Main Analyzer ───────────────────────────────────────────────────────────

def analyze_ticker(ticker: str) -> dict:
    df = fetch_daily(ticker)

    if df.empty:
        raise ValueError(f"No data found for '{ticker}'")

    close = df["Close"]
    volume = df["Volume"]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    price = round(float(close.iloc[-1]), 2)
    prev = round(float(close.iloc[-2]), 2)
    chg = round((price - prev) / prev * 100, 2)

    s20 = round(float(sma20.iloc[-1]), 2)
    s50 = round(float(sma50.iloc[-1]), 2) if not pd.isna(sma50.iloc[-1]) else s20

    rsi = compute_rsi(close)
    macd, macd_sig, macd_hist = compute_macd(close)
    trend = compute_trend(close)
    bb = compute_bollinger(close)

    avg_vol = float(volume.tail(20).mean())
    last_vol = float(volume.iloc[-1])
    vol_ratio = round(last_vol / avg_vol, 2) if avg_vol > 0 else 1.0

    score = 0
    reasons = []

    if price > s20 > s50:
        score += 1; reasons.append("Bullish alignment")
    elif price < s20 < s50:
        score -= 1; reasons.append("Bearish alignment")

    if rsi < 30:
        score += 2; reasons.append("RSI Oversold")
    elif rsi > 70:
        score -= 2; reasons.append("RSI Overbought")

    if macd > macd_sig:
        score += 1; reasons.append("MACD Bullish")
    elif macd < macd_sig:
        score -= 1; reasons.append("MACD Bearish")

    if trend == "Uptrend": score += 1
    if trend == "Downtrend": score -= 1

    if price < bb["lower"]: score += 1
    if price > bb["upper"]: score -= 1

    overview = fetch_overview(ticker)

    return {
        "ticker": ticker.upper(),
        "companyName": overview.get("companyName", ticker.upper()),
        "sector": overview.get("sector", ""),
        "industry": overview.get("industry", ""),
        "price": price,
        "changePct": chg,
        "sma20": s20,
        "sma50": s50,
        "rsi": rsi,
        "macd": macd,
        "trend": trend,
        "bb": bb,
        "volRatio": vol_ratio,
        "marketCap": overview.get("mktCap"),
        "peRatio": overview.get("pe"),
        "score": score,
        "signal": score_to_signal(score),
        "reasons": reasons,
    }


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "ParmStockAdvisor API v3.0", "docs": "/docs"}


@app.get("/analyze/{ticker}")
def analyze(ticker: str):
    try:
        return analyze_ticker(ticker.upper().strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quote/{ticker}")
def quick_quote(ticker: str):
    try:
        quote = fetch_quote(ticker.upper())

        if not quote or "price" not in quote:
            raise ValueError("No quote data")

        price = float(quote["price"])
        prev = float(quote.get("previousClose", price))

        return {
            "ticker": ticker.upper(),
            "price": round(price, 2),
            "change": round(price - prev, 2),
            "changePct": round((price - prev) / prev * 100, 2) if prev else 0,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
