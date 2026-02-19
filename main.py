"""
ParmStockAdvisor — FastAPI Backend
Fetches real-time data from Yahoo Finance using yfinance + pandas/numpy.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Deploy to Railway/Render:
    Just push this folder — they auto-detect the requirements.txt
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import yfinance as yf
import pandas as pd
import numpy as np
import json
import time
import requests

# Session with browser-like headers to avoid Yahoo Finance blocks
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
})

app = FastAPI(title="ParmStockAdvisor API", version="2.0.0")

# Allow all origins so React Native (on any IP) can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Technical Analysis ───────────────────────────────────────────────────────

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
    if pct > 0.5:  return "Uptrend"
    if pct < -0.5: return "Downtrend"
    return "Sideways"

def compute_bollinger(series: pd.Series, period: int = 20):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return {
        "upper": round(float(upper.iloc[-1]), 2),
        "middle": round(float(sma.iloc[-1]), 2),
        "lower": round(float(lower.iloc[-1]), 2),
    }

def score_to_signal(score: int) -> str:
    if score >= 3:  return "STRONG BUY"
    if score == 2:  return "BUY"
    if score == 1:  return "WEAK BUY"
    if score == 0:  return "HOLD"
    if score == -1: return "WEAK SELL"
    if score == -2: return "SELL"
    return "STRONG SELL"

def analyze_ticker(ticker: str) -> dict:
    time.sleep(0.3)  # small delay to avoid rate limiting
    stock = yf.Ticker(ticker, session=session)
    df = stock.history(period="6mo")

    if df is None or df.empty:
        # Try shorter period as fallback
        df = stock.history(period="1mo")

    if df is None or df.empty:
        raise ValueError(f"No data found for '{ticker}'")

    close  = df["Close"]
    volume = df["Volume"]

    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()

    price  = round(float(close.iloc[-1]), 2)
    prev   = round(float(close.iloc[-2]), 2)
    chg    = round((price - prev) / prev * 100, 2)
    s20    = round(float(sma20.iloc[-1]), 2)
    s50    = round(float(sma50.iloc[-1]), 2) if not pd.isna(sma50.iloc[-1]) else s20

    rsi              = compute_rsi(close)
    macd, macd_sig, macd_hist = compute_macd(close)
    trend            = compute_trend(close)
    bb               = compute_bollinger(close)

    avg_vol  = float(volume.tail(20).mean())
    last_vol = float(volume.iloc[-1])
    vol_ratio = round(last_vol / avg_vol, 2) if avg_vol > 0 else 1.0

    # ── Scoring ──────────────────────────────────────────────────────────────
    score, reasons = 0, []

    if price > s20 > s50:
        score += 1; reasons.append("Price above SMA20 & SMA50 — bullish alignment ✅")
    elif price < s20 < s50:
        score -= 1; reasons.append("Price below SMA20 & SMA50 — bearish alignment ❌")
    elif s20 > s50:
        score += 1; reasons.append("Golden cross (SMA20 > SMA50) — bullish ✅")
    else:
        score -= 1; reasons.append("Death cross (SMA20 < SMA50) — bearish ❌")

    if rsi < 30:
        score += 2; reasons.append(f"RSI {rsi} — oversold, strong buy signal 🟢🟢")
    elif rsi < 40:
        score += 1; reasons.append(f"RSI {rsi} — approaching oversold territory 🟢")
    elif rsi > 70:
        score -= 2; reasons.append(f"RSI {rsi} — overbought, consider selling 🔴🔴")
    elif rsi > 60:
        score -= 1; reasons.append(f"RSI {rsi} — approaching overbought 🔴")
    else:
        reasons.append(f"RSI {rsi} — neutral zone (40–60) ⚪")

    if macd > macd_sig and macd_hist > 0:
        score += 1; reasons.append("MACD above signal line — bullish momentum ✅")
    elif macd < macd_sig and macd_hist < 0:
        score -= 1; reasons.append("MACD below signal line — bearish momentum ❌")
    else:
        reasons.append("MACD crossing signal — watch for confirmation ⚠️")

    if trend == "Uptrend":
        score += 1; reasons.append("10-day price trend is upward ✅")
    elif trend == "Downtrend":
        score -= 1; reasons.append("10-day price trend is downward ❌")
    else:
        reasons.append("Price moving sideways — wait for breakout ⚪")

    if price < bb["lower"]:
        score += 1; reasons.append(f"Price below Bollinger lower band (${bb['lower']}) — potential bounce ✅")
    elif price > bb["upper"]:
        score -= 1; reasons.append(f"Price above Bollinger upper band (${bb['upper']}) — potential reversal ❌")

    # Chart data (last 40 days)
    hist40     = [round(float(x), 2) for x in close.tail(40).tolist()]
    sma20_hist = [round(float(x), 2) for x in sma20.tail(40).tolist()]

    # Company info
    try:
        info = stock.info
        name = info.get("longName") or info.get("shortName") or ticker.upper()
        sector   = info.get("sector", "")
        industry = info.get("industry", "")
        market_cap = info.get("marketCap", None)
        pe_ratio   = info.get("trailingPE", None)
        week52_high = info.get("fiftyTwoWeekHigh", None)
        week52_low  = info.get("fiftyTwoWeekLow", None)
    except Exception:
        name = ticker.upper()
        sector = industry = ""
        market_cap = pe_ratio = week52_high = week52_low = None

    return {
        "ticker":       ticker.upper(),
        "companyName":  name,
        "sector":       sector,
        "industry":     industry,
        "price":        price,
        "changePct":    chg,
        "sma20":        s20,
        "sma50":        s50,
        "rsi":          rsi,
        "macd":         macd,
        "macdSignal":   macd_sig,
        "macdHist":     macd_hist,
        "trend":        trend,
        "bb":           bb,
        "volRatio":     vol_ratio,
        "marketCap":    market_cap,
        "peRatio":      round(float(pe_ratio), 2) if pe_ratio else None,
        "week52High":   round(float(week52_high), 2) if week52_high else None,
        "week52Low":    round(float(week52_low), 2) if week52_low else None,
        "score":        score,
        "signal":       score_to_signal(score),
        "priceHistory": hist40,
        "smaHistory":   sma20_hist,
        "reasons":      reasons,
    }


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "ParmStockAdvisor API v2.0", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/analyze/{ticker}")
def analyze(ticker: str):
    """Analyze a single stock ticker."""
    try:
        data = analyze_ticker(ticker.upper().strip())
        return data
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/analyze-many")
def analyze_many(tickers: str):
    """
    Analyze multiple tickers at once.
    Usage: /analyze-many?tickers=AAPL,MSFT,TSLA
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    results = []
    errors  = []
    for t in ticker_list[:20]:  # max 20 at once
        try:
            results.append(analyze_ticker(t))
        except Exception as e:
            errors.append({"ticker": t, "error": str(e)})
    return {"results": results, "errors": errors}

@app.get("/sectors")
def get_sectors():
    """Return all predefined sector watchlists."""
    return {
        "Technology":  ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD", "INTC", "ORCL"],
        "Finance":     ["JPM", "BAC", "GS", "MS", "WFC", "BRK-B", "C", "AXP", "BLK", "V"],
        "Healthcare":  ["JNJ", "PFE", "ABBV", "MRK", "UNH", "LLY", "TMO", "ABT", "BMY", "AMGN"],
        "Energy":      ["XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "PSX", "VLO", "HAL"],
        "Consumer":    ["WMT", "COST", "TGT", "MCD", "SBUX", "NKE", "HD", "LOW", "TJX"],
        "Crypto ETF":  ["IBIT", "FBTC", "GBTC", "ETHA", "BITB", "ARKB"],
    }

@app.get("/quote/{ticker}")
def quick_quote(ticker: str):
    """Fast price-only quote — no heavy TA computation."""
    try:
        stock = yf.Ticker(ticker.upper(), session=session)
        info  = stock.fast_info
        return {
            "ticker": ticker.upper(),
            "price":  round(float(info.last_price), 2),
            "change": round(float(info.last_price - info.previous_close), 2),
            "changePct": round((info.last_price - info.previous_close) / info.previous_close * 100, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
