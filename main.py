"""
ParmStockAdvisor — FastAPI Backend
Fetches real-time data from Financial Modeling Prep (FMP) API.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Deploy to Railway/Render:
    Just push this folder — they auto-detect the requirements.txt
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import requests
import time

# ─── FMP Config ──────────────────────────────────────────────────────────────

FMP_API_KEY = "BDUFyoYbfR6jCYehbHXlT53Y7D8PIfur"
FMP_BASE    = "https://financialmodelingprep.com/api/v3"

app = FastAPI(title="ParmStockAdvisor API", version="3.0.0")

# Allow all origins so React Native (on any IP) can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── FMP Helpers ─────────────────────────────────────────────────────────────

def fmp_get(endpoint: str, params: dict = {}) -> dict | list:
    """Make a GET request to FMP and return parsed JSON."""
    params["apikey"] = FMP_API_KEY
    url = f"{FMP_BASE}/{endpoint}"
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    # FMP returns {"Error Message": "..."} on bad key/limit
    if isinstance(data, dict) and "Error Message" in data:
        raise ValueError(data["Error Message"])
    return data


def get_historical_prices(ticker: str, days: int = 130) -> pd.DataFrame:
    """Return a DataFrame with Date, Open, High, Low, Close, Volume columns."""
    data = fmp_get(f"historical-price-full/{ticker}", {"serietype": "line", "timeseries": days})
    if not data or "historical" not in data or not data["historical"]:
        raise ValueError(f"No historical data found for '{ticker}'")
    df = pd.DataFrame(data["historical"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.rename(columns={"close": "Close"}, inplace=True)
    return df


def get_quote(ticker: str) -> dict:
    """Return the latest FMP quote dict for a ticker."""
    data = fmp_get(f"quote/{ticker}")
    if not data or not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"No quote found for '{ticker}'")
    return data[0]


def get_profile(ticker: str) -> dict:
    """Return company profile from FMP."""
    try:
        data = fmp_get(f"profile/{ticker}")
        if data and isinstance(data, list):
            return data[0]
    except Exception:
        pass
    return {}


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
        "upper":  round(float(upper.iloc[-1]), 2),
        "middle": round(float(sma.iloc[-1]),   2),
        "lower":  round(float(lower.iloc[-1]), 2),
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
    ticker = ticker.upper().strip()

    # 1. Historical prices for TA
    df    = get_historical_prices(ticker, days=130)
    close = df["Close"]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    price = round(float(close.iloc[-1]), 2)
    prev  = round(float(close.iloc[-2]), 2)
    chg   = round((price - prev) / prev * 100, 2)
    s20   = round(float(sma20.iloc[-1]), 2)
    s50   = round(float(sma50.iloc[-1]), 2) if not pd.isna(sma50.iloc[-1]) else s20

    rsi                      = compute_rsi(close)
    macd, macd_sig, macd_hist = compute_macd(close)
    trend                    = compute_trend(close)
    bb                       = compute_bollinger(close)

    # 2. Live quote for volume
    quote      = get_quote(ticker)
    avg_vol    = float(quote.get("avgVolume", 1) or 1)
    last_vol   = float(quote.get("volume", avg_vol) or avg_vol)
    vol_ratio  = round(last_vol / avg_vol, 2) if avg_vol > 0 else 1.0

    # 3. Company profile
    profile    = get_profile(ticker)
    name       = profile.get("companyName") or quote.get("name") or ticker
    sector     = profile.get("sector", "")
    industry   = profile.get("industry", "")
    market_cap = quote.get("marketCap", None)
    pe_ratio   = quote.get("pe", None)
    week52_high = quote.get("yearHigh", None)
    week52_low  = quote.get("yearLow", None)

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

    return {
        "ticker":       ticker,
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
        "week52Low":    round(float(week52_low),  2) if week52_low  else None,
        "score":        score,
        "signal":       score_to_signal(score),
        "priceHistory": hist40,
        "smaHistory":   sma20_hist,
        "reasons":      reasons,
    }


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "ParmStockAdvisor API v3.0 (FMP)", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/analyze/{ticker}")
def analyze(ticker: str):
    """Analyze a single stock ticker."""
    try:
        data = analyze_ticker(ticker)
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
    results, errors = [], []
    for t in ticker_list[:10]:   # FMP free plan: keep batch small to save quota
        try:
            results.append(analyze_ticker(t))
            time.sleep(0.2)      # small delay between calls
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
        q = get_quote(ticker.upper().strip())
        price = float(q.get("price", 0))
        prev  = float(q.get("previousClose") or q.get("open") or price)
        chg   = round(price - prev, 2)
        chg_pct = round((chg / prev * 100), 2) if prev else 0.0
        return {
            "ticker":    ticker.upper(),
            "price":     round(price, 2),
            "change":    chg,
            "changePct": chg_pct,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
