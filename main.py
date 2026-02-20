"""
ParmStockAdvisor — FastAPI Backend
Fetches real-time data from Financial Modeling Prep (FMP) — Stable API.
Two free keys rotated automatically = 500 req/day.
15-minute cache prevents unnecessary API calls.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta

# ─── FMP Key Rotation ─────────────────────────────────────────────────────────
# Two free accounts = 500 req/day total.
# On 429 error, automatically switches to the next key.

FMP_KEYS = [
    "BDUFyoYbfR6jCYehbHXlT53Y7D8PIfur",   # Key 1
    "XklYG6GICnaUcpoe3SGihtG1fLXCiqED",   # Key 2
]
FMP_BASE    = "https://financialmodelingprep.com/stable"
_key_index  = 0

def get_active_key() -> str:
    return FMP_KEYS[_key_index]

def rotate_key():
    global _key_index
    _key_index = (_key_index + 1) % len(FMP_KEYS)

# ─── In-memory Cache ──────────────────────────────────────────────────────────
# 15-minute cache per ticker — avoids burning quota on repeated requests.

CACHE_TTL_MINUTES = 15
_cache: dict = {}

def cache_get(ticker: str):
    entry = _cache.get(ticker)
    if entry and datetime.utcnow() < entry["expires"]:
        return entry["data"]
    return None

def cache_set(ticker: str, data: dict):
    _cache[ticker] = {
        "data":    data,
        "expires": datetime.utcnow() + timedelta(minutes=CACHE_TTL_MINUTES),
    }

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="ParmStockAdvisor API", version="6.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── FMP Helpers ──────────────────────────────────────────────────────────────

def fmp_get(path: str, params: dict = {}) -> dict | list:
    """
    GET from FMP stable API with automatic key rotation on 429.
    Tries all available keys before giving up.
    """
    global _key_index
    last_error = None

    for attempt in range(len(FMP_KEYS)):
        key = get_active_key()
        p   = {**params, "apikey": key}
        url = f"{FMP_BASE}/{path}"
        try:
            r = requests.get(url, params=p, timeout=15)

            if r.status_code == 429:
                # Rate limited on this key — rotate and try next
                rotate_key()
                last_error = f"Key {attempt+1} rate limited (429)"
                time.sleep(0.5)
                continue

            r.raise_for_status()
            data = r.json()

            if isinstance(data, dict) and "Error Message" in data:
                raise ValueError(data["Error Message"])
            if isinstance(data, dict) and "message" in data:
                raise ValueError(f"FMP API error: {data['message']}")

            return data

        except requests.exceptions.HTTPError as e:
            if r.status_code == 429:
                rotate_key()
                last_error = str(e)
                time.sleep(0.5)
                continue
            raise

    raise Exception(f"All API keys exhausted. Last error: {last_error}. Daily quota likely reached — resets at midnight UTC.")


def get_historical_prices(ticker: str) -> pd.DataFrame:
    data = fmp_get("historical-price-eod/light", {
        "symbol": ticker,
        "limit":  130,
    })
    if not data or not isinstance(data, list):
        raise ValueError(f"No historical data found for '{ticker}'")

    df = pd.DataFrame(data)
    df.columns = [c.lower() for c in df.columns]

    close_col = next((c for c in ["close", "adjclose", "price"] if c in df.columns), None)
    if close_col is None:
        raise ValueError(f"Cannot find close price column. Available: {list(df.columns)}")

    df.rename(columns={close_col: "Close"}, inplace=True)
    vol_col = next((c for c in ["volume", "vol"] if c in df.columns), None)
    df.rename(columns={vol_col: "Volume"}, inplace=True) if vol_col else None
    if "Volume" not in df.columns:
        df["Volume"] = 0

    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df


def get_quote(ticker: str) -> dict:
    data = fmp_get("quote", {"symbol": ticker})
    if not data or not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"No quote found for '{ticker}'")
    return data[0]


def get_profile(ticker: str) -> dict:
    try:
        data = fmp_get("profile", {"symbol": ticker})
        if data and isinstance(data, list):
            return data[0]
    except Exception:
        pass
    return {}


# ─── Technical Analysis ───────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs    = gain / loss
    return round(float((100 - (100 / (1 + rs))).iloc[-1]), 2)


def compute_macd(series: pd.Series):
    ema12  = series.ewm(span=12, adjust=False).mean()
    ema26  = series.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal
    return (
        round(float(macd.iloc[-1]),   4),
        round(float(signal.iloc[-1]), 4),
        round(float(hist.iloc[-1]),   4),
    )


def compute_trend(series: pd.Series, days: int = 10) -> str:
    recent = series.tail(days)
    pct    = (recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0] * 100
    if pct > 0.5:  return "Uptrend"
    if pct < -0.5: return "Downtrend"
    return "Sideways"


def compute_bollinger(series: pd.Series, period: int = 20):
    sma   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    return {
        "upper":  round(float((sma + 2 * std).iloc[-1]), 2),
        "middle": round(float(sma.iloc[-1]),              2),
        "lower":  round(float((sma - 2 * std).iloc[-1]), 2),
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

    # Return from cache if still fresh (saves API quota)
    cached = cache_get(ticker)
    if cached:
        return {**cached, "cached": True}

    df     = get_historical_prices(ticker)
    close  = df["Close"]
    volume = df["Volume"]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    price = round(float(close.iloc[-1]), 2)
    prev  = round(float(close.iloc[-2]), 2)
    chg   = round((price - prev) / prev * 100, 2)
    s20   = round(float(sma20.iloc[-1]), 2)
    s50   = round(float(sma50.iloc[-1]), 2) if not pd.isna(sma50.iloc[-1]) else s20

    rsi                       = compute_rsi(close)
    macd, macd_sig, macd_hist = compute_macd(close)
    trend                     = compute_trend(close)
    bb                        = compute_bollinger(close)
    avg_vol                   = float(volume.tail(20).mean())
    last_vol                  = float(volume.iloc[-1])
    vol_ratio                 = round(last_vol / avg_vol, 2) if avg_vol > 0 else 1.0

    quote      = get_quote(ticker)
    live_price = float(quote.get("price", price))
    prev_close = float(quote.get("previousClose") or quote.get("open") or prev)
    live_chg   = round((live_price - prev_close) / prev_close * 100, 2) if prev_close else chg

    profile     = get_profile(ticker)
    name        = profile.get("companyName") or quote.get("name") or ticker
    sector      = profile.get("sector", "")
    industry    = profile.get("industry", "")
    market_cap  = quote.get("marketCap") or profile.get("mktCap") or None
    pe_ratio    = quote.get("pe", None)
    week52_high = quote.get("yearHigh", None)
    week52_low  = quote.get("yearLow",  None)

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

    result = {
        "ticker":       ticker,
        "companyName":  name,
        "sector":       sector,
        "industry":     industry,
        "price":        round(live_price, 2),
        "changePct":    live_chg,
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
        "priceHistory": [round(float(x), 2) for x in close.tail(40).tolist()],
        "smaHistory":   [round(float(x), 2) for x in sma20.tail(40).tolist()],
        "reasons":      reasons,
        "cached":       False,
    }

    cache_set(ticker, result)
    return result


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status":  "ok",
        "message": "ParmStockAdvisor API v6.0 (Dual-key + Cache)",
        "docs":    "/docs",
        "active_key": f"Key {_key_index + 1} of {len(FMP_KEYS)}",
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/cache-status")
def cache_status():
    now = datetime.utcnow()
    return {
        "cached_tickers": len(_cache),
        "active_key":     f"Key {_key_index + 1} of {len(FMP_KEYS)}",
        "entries": {
            ticker: {
                "expires_in_seconds": max(0, int((e["expires"] - now).total_seconds())),
                "fresh": now < e["expires"],
            }
            for ticker, e in _cache.items()
        },
    }

@app.delete("/cache")
def clear_cache():
    _cache.clear()
    return {"status": "cache cleared"}

@app.get("/debug/{ticker}")
def debug(ticker: str):
    raw_hist  = fmp_get("historical-price-eod/light", {"symbol": ticker.upper(), "limit": 3})
    raw_quote = fmp_get("quote", {"symbol": ticker.upper()})
    return {
        "historical_sample": raw_hist[:3] if isinstance(raw_hist, list) else raw_hist,
        "quote_sample":      raw_quote[0] if isinstance(raw_quote, list) and raw_quote else raw_quote,
    }

@app.get("/analyze/{ticker}")
def analyze(ticker: str):
    try:
        return analyze_ticker(ticker)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/analyze-many")
def analyze_many(tickers: str):
    """Usage: /analyze-many?tickers=AAPL,MSFT,TSLA"""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    results, errors = [], []
    for t in ticker_list[:10]:
        try:
            results.append(analyze_ticker(t))
            time.sleep(0.2)
        except Exception as e:
            errors.append({"ticker": t, "error": str(e)})
    return {"results": results, "errors": errors}

@app.get("/sectors")
def get_sectors():
    return {
        "Big Tech": [
            "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AMZN",
            "TSLA", "AMD", "INTC", "ORCL", "CRM", "ADBE", "QCOM",
            "TXN", "AVGO", "MU", "AMAT", "LRCX", "KLAC",
        ],
        "Finance": [
            "JPM", "BAC", "GS", "MS", "WFC", "C", "AXP", "BLK",
            "V", "MA", "PYPL", "COF", "USB", "PNC", "TFC",
            "SCHW", "BX", "KKR", "ICE", "CME",
        ],
        "Healthcare": [
            "JNJ", "PFE", "ABBV", "MRK", "UNH", "LLY", "TMO",
            "ABT", "BMY", "AMGN", "GILD", "ISRG", "CVS", "CI",
            "HUM", "BIIB", "REGN", "VRTX", "MRNA", "ZTS",
        ],
        "Energy": [
            "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC",
            "PSX", "VLO", "HAL", "DVN", "HES", "BKR",
            "MRO", "APA", "WMB",
        ],
        "Consumer": [
            "WMT", "COST", "TGT", "MCD", "SBUX", "NKE", "HD",
            "LOW", "TJX", "PG", "KO", "PEP",
            "PM", "MO", "CL", "KHC", "GIS",
        ],
        "Top ETFs": [
            "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO",
            "VGT", "XLK", "XLF", "XLE", "XLV", "XLI",
            "GLD", "SLV", "TLT", "HYG", "EEM", "VEA",
        ],
        "Crypto ETF": [
            "IBIT", "FBTC", "GBTC", "ETHA", "BITB", "ARKB",
            "HODL", "BTCO", "BRRR", "EZBC",
        ],
        "Industrial": [
            "BA", "CAT", "GE", "HON", "MMM", "RTX", "LMT",
            "NOC", "GD", "DE", "EMR", "ETN", "PH",
            "ITW", "FDX", "UPS", "CSX", "NSC", "UNP",
        ],
        "REITs": [
            "AMT", "PLD", "CCI", "EQIX", "PSA", "O",
            "WELL", "DLR", "AVB", "EQR", "SPG", "VTR",
            "SBAC", "ARE", "BXP", "KIM", "NNN",
        ],
        "AI & Growth": [
            "NVDA", "MSFT", "GOOGL", "META", "AMZN", "CRM",
            "PLTR", "PATH", "SNOW", "DDOG", "NET",
            "MDB", "ZS", "CRWD", "PANW", "S",
        ],
        "Dividends": [
            "JNJ", "KO", "PG", "MMM", "T", "VZ", "O",
            "ABBV", "XOM", "CVX", "IBM", "PEP", "MCD",
            "WMT", "HD", "LOW", "TGT", "COST", "UPS", "FDX",
        ],
        "India ADRs": [
            "INFY",  # Infosys
            "WIT",   # Wipro
            "HDB",   # HDFC Bank
            "IBN",   # ICICI Bank
            "TTM",   # Tata Motors
            "RDY",   # Dr. Reddy's
            "VEDL",  # Vedanta
            "AZRE",  # Azure Power
            "SIFY",  # Sify Technologies
            "YTRA",  # Yatra Online
        ],
    }

@app.get("/quote/{ticker}")
def quick_quote(ticker: str):
    try:
        q          = get_quote(ticker.upper().strip())
        price      = float(q.get("price", 0))
        prev_close = float(q.get("previousClose") or q.get("open") or price)
        chg        = round(price - prev_close, 2)
        chg_pct    = round(chg / prev_close * 100, 2) if prev_close else 0.0
        return {
            "ticker":    ticker.upper(),
            "price":     round(price, 2),
            "change":    chg,
            "changePct": chg_pct,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
