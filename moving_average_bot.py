import collections
import functools
import html
import io
import itertools
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, time as dt_time, UTC,timezone
from typing import Dict,Optional

# Networking / requests
import gspread
import requests
from oauth2client.service_account import ServiceAccountCredentials

# Data handling
import numpy as np
import pandas as pd

# Financial / trading
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame
import yfinance as yf
from finvizfinance.screener.overview import Overview

# Technical analysis
import ta
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD, SMAIndicator

# Plotting
import matplotlib.pyplot as plt

from collections import defaultdict
import pytz
from itertools import product
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser
# ================= CONFIG =================
ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# Telegram config
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
api_key = os.getenv("FMP_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
print("KEY:", os.getenv("APCA_API_KEY_ID"))
api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, base_url=ALPACA_BASE_URL)

EST = pytz.timezone("US/Eastern")

SMA_SHORT = int(os.getenv("SMA_SHORT", "20"))
SMA_LONG = int(os.getenv("SMA_LONG", "50"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "1"))
SIGNAL_EXPIRY_DAYS = int(os.getenv("SIGNAL_EXPIRY_DAYS", "5"))
NEWS_API_KEY="fa1d035152ac4016bd4d8647244d1748"
LOG_DIR = "logs"
SIGNAL_DIR = "signals"
MODEL_DIR = "models"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SIGNAL_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# logging
EST = pytz.timezone("US/Eastern")


#Ctrl+k Ctrl+C  - comment   
#Ctrl+ K Ctrl U  - uncomment

BACKTEST_DAYS = 60  # Adjustable, conservative by default
MODEL_DIR = "models/sklearn/"
LOG_DIR = "logs"
SIGNAL_DIR = "signals"
MODEL_DIR = "models"

INITIAL_CAPITAL = 10000
COMMISSION = 0.001  # 0.1%


os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SIGNAL_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CACHE_FILE = "sent_news_cache.json"
YAHOO_NEWS_FEED = "https://finance.yahoo.com/news/rssindex"
analyzer = SentimentIntensityAnalyzer()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
def log_message(msg):
    ts = datetime.now(EST).strftime("[%Y-%m-%d %H:%M:%S]")
    logging.info(f"{ts} {msg}")
    print(f"{ts} {msg}")

#################################################
########### Initialize ##########################
#################################################
# Alpaca client
api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, base_url=ALPACA_BASE_URL)
# create once (reuse)



# Google Sheets
if os.path.exists("google_creds.json"):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("google_creds.json", scope)
        client = gspread.authorize(creds)
        sheet = client.open("Daily_stocks")
        sheet_ws = sheet.worksheet("Tickers")
        sheet_ms = sheet.worksheet("Morning_Scanner")
        tickers_ws = sheet_ws.col_values(1)
        
        # ✅ Get all rows from "Morning_Scanner"
        rows = sheet_ms.get_all_records()

        # ✅ Extract only ticker symbols (first column)
        tickers_ms = [row["Ticker"] for row in rows if row.get("Ticker")]
        tickers = list(set(tickers_ws + tickers_ms))
    except Exception as e:
        log_message(f"⚠️ Google Sheets init failed: {e}")
        tickers = []
else:
    tickers = []
    log_message("⚠️ google_creds.json not found; continuing without Sheets.")

# Signals persistence (single file)
SIGNAL_FILE = os.path.join(SIGNAL_DIR, "sent_signals_master.txt")

# ================= UTILITIES =================
def safe_tz_localize_and_convert(df):
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")
    return df

def send_message(msg):
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        log_message("⚠️ Telegram not configured.")
        return
    MAX_LEN = 4000
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    chunks = [msg[i:i+MAX_LEN] for i in range(0, len(msg), MAX_LEN)]
    for idx, chunk in enumerate(chunks, 1):
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": ""}
        try:
            r = requests.post(url, data=data, timeout=10)
            if r.status_code != 200:
                log_message(f"❌ Telegram failed (chunk {idx}/{len(chunks)}): {r.status_code} {r.text}")
            else:
                log_message(f"✅ Telegram sent (chunk {idx}/{len(chunks)})")
        except Exception as e:
            log_message(f"❌ Telegram exception: {e}")


def send_photo(image_buf):
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                  data={"chat_id": TELEGRAM_CHAT_ID},
                  files={"photo": ("chart.png", image_buf)})
    

def create_features(bars):
    """
    Returns (features_df, full_df) where features_df is the ML-ready numeric matrix.
    Adds indicators (SMA, RSI, ATR, MACD diff, returns, vol ratios, ROC, ADX).
    """
    df = bars.copy()
    if df.empty:
        return pd.DataFrame(), df

    df = safe_tz_localize_and_convert(df) if df.index.tz is None else df
    df["SMA20"] = df["close"].rolling(20).mean()
    df["SMA50"] = df["close"].rolling(50).mean()
    df["SMA100"] = df["close"].rolling(100).mean()
    df["RSI"] = compute_rsi(df["close"], 14)
    df["ATR"] = compute_atr(df[["high", "low", "close"]], 14)
    macd, macd_signal = compute_macd(df["close"])
    df["MACD"] = macd
    df["MACD_SIGNAL"] = macd_signal
    df["MACD_DIFF"] = df["MACD"] - df["MACD_SIGNAL"]
    df["RET_1"] = df["close"].pct_change(1)
    df["RET_5"] = df["close"].pct_change(5)
    df["VOL_MA20"] = df["volume"].rolling(20).mean()
    df["VOL_RATIO"] = df["volume"] / (df["VOL_MA20"].replace(0, np.nan))
    df["ROC_10"] = df["close"].pct_change(10) * 100
    try:
        df["ADX"] = ADXIndicator(df["high"], df["low"], df["close"], 14).adx()
    except Exception:
        df["ADX"] = np.nan

    # Select features for model (ensure they exist)
    feature_cols = ["SMA20", "SMA50", "SMA100", "RSI", "ATR", "MACD_DIFF", "RET_1", "RET_5", "VOL_RATIO", "ROC_10", "ADX"]
    available = [c for c in feature_cols if c in df.columns]
    feat_df = df[available].copy()

    # Drop rows with NaNs at head
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna()
    return feat_df, df


# Save executed trade
def log_trade(trade: dict):
    """
    trade: {
        'Ticker': str,
        'Date': datetime,
        'Signal': 'LONG'/'SHORT',
        'Entry': float,
        'Exit': float,
    }
    """
    # --- Google Sheets ---
    if sheet:
        try:
            trades_ws = sheet.worksheet("ExecutedTrades")
        except gspread.WorksheetNotFound:
            trades_ws = sheet.add_worksheet(title="ExecutedTrades", rows="1000", cols="20")
            trades_ws.append_row(list(trade.keys()))
        trades_ws.append_row([trade[k] if k in trade else "" for k in trades_ws.row_values(1)])




# ================= INDICATORS =================
def compute_rsi(series, period=14):
    series = series.dropna()
    if series.empty:
        return pd.Series(dtype=float)
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(df, period=14):
    if df.shape[0] < 2:
        return pd.Series([np.nan]*len(df), index=df.index)
    df2 = df.copy()
    df2["H-L"] = df2["high"] - df2["low"]
    df2["H-Cp"] = (df2["high"] - df2["close"].shift(1)).abs()
    df2["L-Cp"] = (df2["low"] - df2["close"].shift(1)).abs()
    tr = df2[["H-L", "H-Cp", "L-Cp"]].max(axis=1)
    return tr.rolling(period).mean()

def compute_macd(close):
    close = close.dropna()
    if close.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def add_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Adds SMA, RSI, and ATR indicators to a price dataframe.
    """
    try:
        bars = bars.copy()
        bars['sma_short'] = bars['close'].rolling(window=20).mean()
        bars['sma_long'] = bars['close'].rolling(window=50).mean()
        bars['rsi'] = ta.momentum.RSIIndicator(bars['close'], window=14).rsi()
        bars['atr'] = ta.volatility.AverageTrueRange(
            high=bars['high'], low=bars['low'], close=bars['close'], window=14
        ).average_true_range()
        bars.dropna(inplace=True)
        return bars
    except Exception as e:
        print(f"⚠️ Indicator generation failed: {e}")
        return bars


# ===================== HELPERS =====================
def fetch_last_bar(ticker, limit=120, timeframe='1Min'):
    """
    Fetch the most recent valid (non-NaN) bar for the given ticker.
    Falls back to earlier bars if the last one is incomplete.
    """
    end_dt = datetime.now(EST)
    start_dt = end_dt - timedelta(minutes=limit)

    ticker = ticker.strip().upper()
    print(f"📊 Fetching bars for {ticker} from {start_dt:%Y-%m-%d %H:%M} to {end_dt:%Y-%m-%d %H:%M}")

    try:
        bars = api.get_bars(
            ticker,
            tradeapi.TimeFrame.Minute,
            start=start_dt.isoformat(),
            end=end_dt.isoformat(),
            feed='iex'  # more reliable than 'iex' for paper/live
        ).df

        if bars.empty:
            print(f"⚠️ No bars returned for {ticker}")
            return None, pd.DataFrame()

        bars = prepare_indicators(bars)
        # Drop rows that have NaN indicators (incomplete or insufficient history)
        valid_bars = bars.dropna(subset=['SMA_short', 'SMA_long', 'ATR', 'RSI'])

        if valid_bars.empty and limit < 300:
            print(f"🔁 Retrying {ticker} with extended range...")
            return fetch_last_bar(ticker, limit=300)

        # ✅ Use the last valid completed bar instead of the very last row
        last_bar = valid_bars.iloc[-1]
        print(f"✅ Using last valid bar for {ticker}: {last_bar.name} Close={last_bar['close']:.2f}")
        return last_bar, bars

    except Exception as e:
        print(f"⚠️ Error fetching bars for {ticker}: {e}")
        return None, pd.DataFrame()


# ================================
# Prepare indicators
# ================================
def prepare_indicators(df, filters=None):
    filters = filters or {'SMA_short_period': 20, 'SMA_long_period': 50, 'ATR_period': 14, 'RSI_period': 14}
    df = df.copy()
    df['SMA_short'] = df['close'].rolling(filters['SMA_short_period']).mean()
    df['SMA_long'] = df['close'].rolling(filters['SMA_long_period']).mean()
    # ATR
    df['H-L'] = df['high'] - df['low']
    df['H-C'] = (df['high'] - df['close'].shift()).abs()
    df['L-C'] = (df['low'] - df['close'].shift()).abs()
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['ATR'] = df['TR'].rolling(filters['ATR_period']).mean()
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(filters['RSI_period']).mean()
    avg_loss = loss.rolling(filters['RSI_period']).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def is_morning_window():
    """Run only between 9:30–11:30 AM Eastern."""
    now = datetime.now(pytz.timezone(TIMEZONE))
    start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end = now.replace(hour=11, minute=30, second=0, microsecond=0)
    return start <= now <= end

def fetch_latest_news(category,page_size):
    news_headlines = []
    url = f"https://newsapi.org/v2/everything?q={category}&language=en&pageSize={page_size}&apiKey={NEWS_API_KEY}"
    response  = requests.get(url)
    data = response.json()

    if response.status_code !=200:
        print(f"Failed to fetch news: {data.get('message', 'Unknown error')}")
        return {'error': 'Failed to fetch news'}
    else:
        news = data.get("articles",[])

        for news_item in news:
            news_headlines.append(
                {
                    "title": news_item.get("title"),
                    "link":news_item.get("url"),
                    "description":news_item.get("description")
                }
            )
        return news_headlines

# ================= MORNING SCAN TOP MOVERS AND LOOSERS  =================

def clean_text(text):
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove zero-width and non-printable characters
    text = re.sub(r'[\u200b-\u200d\uFEFF]', '', text)
    return escape_text(text)

def send_telegram_safe(message, chunk_size=4000):
    lines = message.split("\n")
    chunk = ""
    for line in lines:
        if len(chunk) + len(line) + 1 > chunk_size:
            send_message(chunk)
            chunk = ""
        chunk += line + "\n"
    if chunk:
        send_message(chunk)

def escape_text(text):
    if not text:
        return ""
    return html.escape(str(text), quote=False)


def get_top_gap_gainers(top_n=5):
    """
    Fetch top gap gainers, perform gap analysis, send Telegram alerts,
    and update Google Sheet with daily tickers for further analysis.
    """
    filters_dict = {
        'Current Volume': 'Over 500K',
        'Gap': 'Up 10%',
        'Industry': 'Stocks only (ex-Funds)'
    }

    overview = Overview()
    overview.set_filter(filters_dict=filters_dict)
    df = overview.screener_view()

    if df.empty:
        print("No data found for top gainers.")
        return

    # Keep only desired columns
    wanted_cols = ['Ticker', 'Price', 'Change', 'Volume', 'Average Volume (3 Month)', 'Gap', 'Volatility (Week)']
    df = df[[c for c in wanted_cols if c in df.columns]]

    # Convert numeric columns
    for col in ['Change', 'Volume', 'Average Volume (3 Month)', 'Gap']:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.replace(',', '', regex=False).astype(float)
            else:
                df[col] = df[col].astype(float)

    # Fix percentages
    df['Change'] = df['Change'] * 100
    if 'Gap' in df.columns:
        df['Gap'] = df['Gap'] * 100

    # Filter: today's volume > average 3-month volume
    if 'Volume' in df.columns and 'Average Volume (3 Month)' in df.columns:
        df = df[df['Volume'] > df['Average Volume (3 Month)']]

    # Sort descending by Change and pick top N
    df = df.sort_values(by='Change', ascending=False).head(top_n)

    # --- Send Telegram message ---
    message = "<b>📊 Morning Movers Watchlist</b>\n\n"
    newsmessage = "<b>📰 News for Top Gainers</b>\n"
    tickers_to_save = df['Ticker'].tolist()

    for _, row in df.iterrows():
        ticker = row['Ticker']
        message += (
            f"🔹 Ticker: {ticker}\n"
            f"💲 Price: {row['Price']}\n"
            f"✅ Gap: {row['Change']:.2f}%\n"
            f"🎯 Volume: {row['Volume']}\n\n"
        )

        # Fetch latest news
        newslinks = fetch_latest_news(ticker, page_size=2)
        if newslinks and newslinks != {'error': 'Failed to fetch news'}:
            for news_item in newslinks:
                title = html.escape(news_item.get('title') or "")
                link = html.escape(news_item.get('link') or "")
                newsmessage += f"Ticker: {ticker}\n📰 {title} - {link}\n\n"
    send_telegram_safe(message)
    send_telegram_safe(newsmessage)
    print("\n✅ Sent Telegram alerts for top gainers.")
    return df


def update_morning_scanner_with_ai(top_n=5):
    """
    Fetch top gainers, analyze technicals,
    send Telegram alerts, and update Morning_Scanner Google Sheet.
    """
    # Step 1: Fetch top gainers
    df = get_top_gap_gainers(top_n=top_n)
    if df is None or df.empty:
        log_message("⚠️ No top gainers found today.")
        log_message("⚠️ No top gainers found today.")
        return

    results = []

    results = []
    ai_forecasts = []

    for _, row in df.iterrows():
        ticker = row['Ticker']
        close = row['Price']
        try:
            # Fetch last 10 days bars
            end_dt = datetime.now(EST)
            start_dt = end_dt - timedelta(days=80)
            bars = api.get_bars(
                ticker, tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute),
                start=start_dt.isoformat(), end=end_dt.isoformat(),
                feed="iex"
            ).df

            if bars.empty:
                continue

            # Calculate technical indicators
            bars = calculate_indicators_for_alpaca(bars)
            last_bar = bars.iloc[-1]

            # Generate signal with
            signal = core_signal_live(
                last_bar=last_bar,
                sma_short=last_bar['SMA_short'],
                sma_long=last_bar['SMA_long'],
                atr=last_bar['ATR'],
                rsi=last_bar['RSI'],
                ticker=ticker,
                historical_bars=bars,
                current_time=datetime.now()
            )

            # Append result for sheet and Telegram
            results.append({
                'Ticker': ticker,
                'Direction': signal['signal'],
                'Price': close,
                'RSI': last_bar['RSI'],
                'ADX': last_bar.get('ADX', np.nan),
                'VolSpike': last_bar.get('VolSpike', np.nan),
                'SMA_Gap%': last_bar.get('SMA_Gap%', np.nan),
                'Score': signal['score']
            })


        except Exception as e:
            log_message(f"⚠️ Analysis failed for {ticker}: {e}")

    if not results:
        send_message("🌄 Morning scan complete — no actionable tickers today.")
        return

    # Step 2: Update Google Sheet (Morning_Scanner)
    columns = ["Ticker", "Direction", "Price", "RSI", "ADX", "VolSpike", "SMA_Gap%", "Score"]
    df_results = pd.DataFrame(results, columns=columns)
    try:
        if 'sheet' in globals():
            try:
                ws = sheet.worksheet("Morning_Scanner")
            except Exception:
                ws = sheet.add_worksheet(title="Morning_Scanner", rows="1000", cols=str(len(columns)))
            ws.clear()
            df_results = df_results.fillna("")
            df_results.replace([np.inf, -np.inf], "", inplace=True)
            ws.update([columns] + df_results.values.tolist())
            log_message(f"📊 Morning_Scanner updated with {len(df_results)} tickers.")
    except Exception as e:
        log_message(f"⚠️ Failed to update Morning_Scanner: {e}")
    df_results1 = pd.DataFrame(results, columns=columns)
    # Step 3: Send Telegram Alerts
    msg = "📊 Morning Scanner — Top Gainers Forecast\n\n"
    for _, r in df_results1.iterrows():
        msg += (f"🔹 {r['Ticker']} | {r['Direction']} | Price: {r['Price']:.2f}\n"
                f"RSI: {r['RSI']:.2f} | ADX: {r['ADX']} | Vol x{r['VolSpike']} | SMA Gap: {r['SMA_Gap%']}% | Score: {r['Score']:.2f}\n"
                )
        
    send_message(msg)
    log_message("✅ Telegram alert sent for Morning Scanner tickers with analysis.")


   
# ================= MORNING SCAN TOP MOVERS AND LOOSERS  END =================

# ================= PREVIOUS DAY ANALYSIS =================


CACHE_DIR = "cache_bars"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_bars_cached(ticker: str, timeframe="1Min", days=180):
    """
    Fetch bars with caching and automatic resampling.
    - Uses cached 1Min data to create higher timeframes (5Min, 15Min, 1Hour, 1Day)
    - Downloads only missing data from Alpaca.
    """
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_{timeframe}.csv")
    one_min_cache = os.path.join(CACHE_DIR, f"{ticker}_1Min.csv")

    # Map string to TimeFrame object
    tf_map = {
        "1Min":  TimeFrame(1, tradeapi.TimeFrameUnit.Minute),
        "5Min":  TimeFrame(5, tradeapi.TimeFrameUnit.Minute),
        "15Min": TimeFrame(15,tradeapi.TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, tradeapi.TimeFrameUnit.Hour),
        "1Day":  TimeFrame(1, tradeapi.TimeFrameUnit.Day),
    }
    tf_obj = tf_map.get(timeframe, TimeFrame(1, tradeapi.TimeFrameUnit.Day))

    df_existing = None
    end_date = datetime.now(pytz.UTC)

    # Load cached version if available
    if os.path.exists(cache_path):
        try:
            df_existing = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if df_existing.index.tz is None:
                df_existing.index = df_existing.index.tz_localize(pytz.UTC)
            last_ts = df_existing.index.max()
            start = last_ts + timedelta(minutes=1)
            print(f"🧠 Cached bars found for {ticker} ({timeframe}), last date: {last_ts.date()}")
        except Exception as e:
            print(f"⚠️ Cache read error for {ticker}: {e}")
            df_existing = pd.DataFrame()
            start = end_date - timedelta(days=days)
    else:
        df_existing = pd.DataFrame()
        start = end_date - timedelta(days=days)

    # --- If timeframe > 1Min and 1Min cache exists, resample instead of fetching ---
    if timeframe != "1Min" and os.path.exists(one_min_cache):
        print(f"♻️ Building {timeframe} bars from 1Min cache for {ticker}...")
        df_1m = pd.read_csv(one_min_cache, index_col=0, parse_dates=True)
        if df_1m.index.tz is None:
            df_1m.index = df_1m.index.tz_localize(pytz.UTC)
        df = resample_bars(df_1m, timeframe)
        df.to_csv(cache_path)
        return df

    # --- Otherwise, fetch from Alpaca ---
    try:
        df_new = api.get_bars(
            ticker,
            tf_obj,
            start=start.isoformat(),
            end=end_date.isoformat(),
            feed="iex"
        ).df
    except Exception as e:
        print(f"❌ Error fetching {timeframe} bars for {ticker}: {e}")
        return df_existing if df_existing is not None else pd.DataFrame()

    if df_new.empty and not df_existing.empty:
        print(f"⚠️ No new bars for {ticker}, using cached data.")
        return df_existing

    # Merge and save
    if not df_existing.empty:
        df = pd.concat([df_existing, df_new])
        df = df[~df.index.duplicated(keep='last')]
    else:
        df = df_new

    df.to_csv(cache_path)
    print(f"💾 Cached {len(df)} bars for {ticker} ({timeframe})")
    return df


def resample_bars(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample 1-minute data into higher timeframes.
    Supported: 5Min, 15Min, 1Hour, 1Day
    """
    rule_map = {
        "5Min": "5T",
        "15Min": "15T",
        "1Hour": "1H",
        "1Day": "1D"
    }
    if timeframe not in rule_map:
        return df_1m

    rule = rule_map[timeframe]
    print(f"📊 Resampling data to {timeframe} ({rule})...")

    df_resampled = df_1m.resample(rule, label='right', closed='right').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return df_resampled



# ============================
# Config / Paths
# ============================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def ensure_model_dir():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

# ===============================
# Indicators Calculation
# ===============================
def calculate_indicators(bars: pd.DataFrame, sma_short=20, sma_long=50, rsi_period=14, atr_period=14):
    df = bars.copy()
    
    # Calculate SMA and RSI indicators
    df['SMA_short'] = SMAIndicator(df['Close'], window=sma_short).sma_indicator()
    df['SMA_long'] = SMAIndicator(df['Close'], window=sma_long).sma_indicator()
    df['RSI'] = RSIIndicator(df['Close'], window=rsi_period).rsi()
    
    # Simple ATR calculation
    df['ATR'] = df['Close'].rolling(atr_period).apply(lambda x: x.max() - x.min(), raw=False)
    
    # Drop any rows with NaN values (from rolling windows)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def calculate_indicators_for_alpaca(bars: pd.DataFrame, sma_short=20, sma_long=50, rsi_period=14, atr_period=14):
    df = bars.copy()
    
    # Calculate SMA and RSI indicators
    df['SMA_short'] = SMAIndicator(df['close'], window=sma_short).sma_indicator()
    df['SMA_long'] = SMAIndicator(df['close'], window=sma_long).sma_indicator()
    df['RSI'] = RSIIndicator(df['close'], window=rsi_period).rsi()
    
    # Simple ATR calculation
    df['ATR'] = df['close'].rolling(atr_period).apply(lambda x: x.max() - x.min(), raw=False)
    
    # Drop any rows with NaN values (from rolling windows)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

# =========================
# CORE SIGNAL FOR BACKTEST
# =========================
# =========================
# UNIVERSAL CORE SIGNAL
# =========================

# ======================
# Simplified core signal
# ======================


import pandas as pd
import numpy as np
from datetime import timedelta

def core_signal(
    df: pd.DataFrame,
    filters: dict = None,
    model=None,
    last_trade_time=None,
    current_time=None,
    feature_columns: list = None,
    log_message=print
) -> dict:
    """
    🚀 Optimized production-ready signal generator.

    Features:
    - SMA, RSI, ATR calculated dynamically if missing
    - Cooldown logic
    - Threshold-based scoring
    - Safe output for backtesting and live trading
    """

    # === Default Filter Values ===
    filters = filters or {
        'SMA_short_period': 20,
        'SMA_long_period': 50,
        'RSI_period': 14,
        'ATR_period': 14,
        'ATR_threshold': 0.05,
        'SMA_score': 0.25,
        'RSI_oversold': 30,
        'RSI_overbought': 70,
        'RSI_score': 0.2,
        'STOP_ATR_MULT': 1.5,
        'TP_ATR_MULT': 3.0,
        'AI_score': 0.3,
        'Score_threshold': 0.05,
        'Cooldown_minutes': 15
    }

    # === Default Safe Output ===
    out = dict(
        signal='FLAT',
        entry_price=np.nan,
        stop=np.nan,
        tp=np.nan,
        atr=np.nan,
        score=0.0,
        Risk=0.0,
        Reward=0.0,
        PositionSize=0.0
    )

    if df.empty or 'close' not in df.columns:
        return out

    df = df.copy()

    # === Calculate Indicators Dynamically ===
    if 'SMA_short' not in df:
        df['SMA_short'] = df['close'].rolling(filters['SMA_short_period']).mean()
    if 'SMA_long' not in df:
        df['SMA_long'] = df['close'].rolling(filters['SMA_long_period']).mean()
    if 'ATR' not in df:
        df['TR'] = df[['high', 'low', 'close']].apply(
            lambda row: max(row['high'] - row['low'],
                            abs(row['high'] - row['close']),
                            abs(row['low'] - row['close'])),
            axis=1
        )
        df['ATR'] = df['TR'].rolling(filters['ATR_period']).mean()
    if 'RSI' not in df:
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(filters['RSI_period']).mean()
        avg_loss = loss.rolling(filters['RSI_period']).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        df['RSI'] = 100 - (100 / (1 + rs))

    last = df.iloc[-1]
    price = last['close']
    atr = last['ATR']
    sma_short = last['SMA_short']
    sma_long = last['SMA_long']
    rsi = last['RSI']

    if pd.isna(price) or pd.isna(atr):
        return out

    # === ATR Volatility Filter ===
    if atr / price > filters['ATR_threshold']:
        log_message(f"⚠️ High volatility skipped (ATR/Price={atr/price:.2%})")
        return out

    # === Cooldown Logic ===
    if last_trade_time and current_time:
        cooldown = timedelta(minutes=filters['Cooldown_minutes'])
        if (current_time - last_trade_time) < cooldown:
            return out

    # === Signal Scoring ===
    score = 0.0

    # SMA crossover
    if len(df) >= 2:
        prev = df.iloc[-2]
        if sma_short > sma_long and prev['SMA_short'] <= prev['SMA_long']:
            score += filters['SMA_score']
        elif sma_short < sma_long and prev['SMA_short'] >= prev['SMA_long']:
            score -= filters['SMA_score']

    # RSI extremes
    if not pd.isna(rsi):
        if rsi < filters['RSI_oversold']:
            score += filters['RSI_score']
        elif rsi > filters['RSI_overbought']:
            score -= filters['RSI_score']

    # === Build Final Signal ===
    threshold = filters['Score_threshold']
    STOP_MULT = filters['STOP_ATR_MULT']
    TP_MULT = filters['TP_ATR_MULT']

    if score > threshold:
        out.update(
            signal='LONG',
            entry_price=price,
            stop=price - STOP_MULT * atr,
            tp=price + TP_MULT * atr,
            atr=atr,
            score=score,
            Risk=STOP_MULT * atr,
            Reward=TP_MULT * atr
        )
        log_message(f"📈 LONG | Price={price:.2f} | ATR={atr:.4f} | Score={score:.2f}")
    elif score < -threshold:
        out.update(
            signal='SHORT',
            entry_price=price,
            stop=price + STOP_MULT * atr,
            tp=price - TP_MULT * atr,
            atr=atr,
            score=score,
            Risk=STOP_MULT * atr,
            Reward=TP_MULT * atr
        )
        log_message(f"📉 SHORT | Price={price:.2f} | ATR={atr:.4f} | Score={score:.2f}")

    return out



# ===============================
# Core Signal Live
# ===============================
def core_signal_live(
    last_bar: pd.Series,
    sma_short: float,
    sma_long: float,
    atr: float,
    rsi: float,
    ticker: str,
    model=None,
    lookback=60,
    historical_bars: pd.DataFrame = None,
    filters: dict = None,
    last_trade_time_val=None,
    current_time=None,
    log_message=print
) -> dict:

    filters = filters or {
        'ATR_threshold': 0.05,
        'SMA_score': 0.25,
        'RSI_oversold': 30,
        'RSI_overbought': 70,
        'RSI_score': 0.2,
        'STOP_ATR_MULT': 1.5,
        'TP_ATR_MULT': 3.0,
        'Score_threshold': 0.05,
        'Cooldown_minutes': 15
    }

    out = {
        'signal': 'FLAT',
        'entry_price': np.nan,
        'stop': np.nan,
        'tp': np.nan,
        'atr': np.nan,
        'score': 0.0,
        'Risk': 0.0,
        'Reward': 0.0,
        'PositionSize': 0.0
    }

    price = last_bar.get('close', np.nan)
    if pd.isna(price) or pd.isna(atr):
        return out

    # ATR filter
    if atr / price > filters['ATR_threshold']:
        log_message(f"⚠️ High volatility skipped (ATR/Price={atr/price:.2%})")
        return out

    # Cooldown check
    if last_trade_time_val and current_time:
        cooldown = timedelta(minutes=filters['Cooldown_minutes'])
        if (current_time - last_trade_time_val) < cooldown:
            return out

    # --- Score Calculation (SMA + RSI) ---
    score = 0.0
    if sma_short > sma_long:
        score += filters['SMA_score']
    elif sma_short < sma_long:
        score -= filters['SMA_score']

    if rsi < filters['RSI_oversold']:
        score += filters['RSI_score']
    elif rsi > filters['RSI_overbought']:
        score -= filters['RSI_score']


    # Final signal
    threshold = filters['Score_threshold']
    STOP_MULT = filters['STOP_ATR_MULT']
    TP_MULT = filters['TP_ATR_MULT']

    if score > threshold:
        out.update(
            signal='LONG',
            entry_price=price,
            stop=price - STOP_MULT * atr,
            tp=price + TP_MULT * atr,
            atr=atr,
            score=score,
            Risk=STOP_MULT * atr,
            Reward=TP_MULT * atr
        )
        log_message(f"📈 LONG | Price={price:.2f} | ATR={atr:.4f} | Score={score:.2f}")
    elif score < -threshold:
        out.update(
            signal='SHORT',
            entry_price=price,
            stop=price + STOP_MULT * atr,
            tp=price - TP_MULT * atr,
            atr=atr,
            score=score,
            Risk=STOP_MULT * atr,
            Reward=TP_MULT * atr
        )
        log_message(f"📉 SHORT | Price={price:.2f} | ATR={atr:.4f} | Score={score:.2f}")

    return out


#########################################################
############  BACK TESTING LOGIC ########################
############   STARTING          ########################
#########################################################

def backtest_strategy(model=None, filters=None, timeframe='1Min', days=180):
    """
    Backtests the enhanced core_signal logic with signal logging and optional Telegram reporting.
    
    Returns:
        df_results (pd.DataFrame): Performance metrics for all tickers.
        sig_df (pd.DataFrame): Log of all entry/exit signals.
    """
    end_date = datetime.now(EST)
    start_date = end_date - timedelta(days=days)
    print(f"🧩 Starting backtest from {start_date} to {end_date} for {len(tickers)} tickers")

    results = []
    trades = []
    signals_log = []
    equity_curves = {}

    for ticker in tickers:
        print(f"📜 Backtesting {ticker} ...")
        try:
            # --- Fetch & prepare data ---
            bars = get_bars_cached(ticker, timeframe, days=days)
            bars = prepare_indicators(bars)
            if bars.empty or len(bars) < max(filters.get('SMA_long_period', 50), 60):
                print(f"⚠️ Not enough data for {ticker}")
                continue

            _, full_df = create_features(bars)
            if full_df.empty:
                continue

            equity = INITIAL_CAPITAL
            equity_curve = [equity]
            position = 0
            entry_price = 0
            qty = 0
            pnl_list = []
            last_trade_time = None

            # --- Run core_signal on each bar ---
            for i in range(max(filters.get('SMA_long_period', 50), 60), len(full_df)):
                current = full_df.iloc[:i].copy()
                current_time = current.index[-1]

                sig = core_signal(
                    df=current,
                    filters=filters,
                    model=model,
                    last_trade_time=last_trade_time,
                    current_time=current_time,
                    log_message=lambda msg: None  # suppress print for backtest
                )

                # --- Signal Logging ---
                if sig['signal'] in ("LONG", "SHORT"):
                    signals_log.append({
                        "Ticker": ticker,
                        "Time": current_time,
                        "Signal": sig['signal'],
                        "Price": round(sig['entry_price'], 2),
                        "Stop": round(sig['stop'], 2) if sig['stop'] else None,
                        "TP": round(sig['tp'], 2) if sig['tp'] else None,
                        "Score": round(sig['score'], 3)
                    })
                    last_trade_time = current_time

                # --- Trade Execution ---
                if position == 0 and sig['signal'] in ("LONG", "SHORT"):
                    direction = 1 if sig['signal'] == "LONG" else -1
                    entry_price = sig['entry_price']
                    atr = sig['atr'] or 0.01
                    trade_risk = sig.get('Risk', 0.01) * equity
                    stop_mult = filters.get('STOP_ATR_MULT', 1.5)
                    trade_size = trade_risk / (atr * stop_mult if atr > 0 else 1)
                    qty = max(1, int(trade_size / entry_price))
                    position = direction

                    trades.append({
                        "Ticker": ticker,
                        "Entry Time": current_time,
                        "Entry Price": entry_price,
                        "Action": sig['signal'],
                        "Qty": qty
                    })

                elif position != 0:
                    close_price = current.iloc[-1]['close']
                    if sig['stop'] and sig['tp']:
                        stop_hit = close_price <= sig['stop'] if position == 1 else close_price >= sig['stop']
                        tp_hit = close_price >= sig['tp'] if position == 1 else close_price <= sig['tp']

                        if stop_hit or tp_hit:
                            exit_price = close_price
                            pnl = (exit_price - entry_price) * position * qty
                            equity += pnl
                            pnl_list.append(pnl)

                            trades[-1].update({
                                "Exit Time": current_time,
                                "Exit Price": exit_price,
                                "PnL": pnl
                            })

                            position = 0

                equity_curve.append(equity)

            equity_curves[ticker] = equity_curve

            # --- Metrics Calculation ---
            total_pnl = sum(pnl_list)
            win_rate = (
                np.mean([1 if t.get("PnL", 0) > 0 else 0 for t in trades[-len(pnl_list):]])
                if pnl_list else 0
            )
            returns = pd.Series(equity_curve).pct_change().fillna(0)
            sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252 * 6.5 * 60)
            max_dd = (1 - np.min(pd.Series(equity_curve) / np.maximum.accumulate(equity_curve))) * 100

            results.append({
                "Ticker": ticker,
                "Total PnL": round(total_pnl, 2),
                "Win Rate": round(win_rate * 100, 2),
                "Sharpe": round(sharpe, 2),
                "Max Drawdown %": round(max_dd, 2),
                "Final Equity": round(equity, 2)
            })

        except Exception as e:
            print(f"❌ Backtest error for {ticker}: {e}")
            continue

    # --- Finalize results ---
    df_results = pd.DataFrame(results)
    sig_df = pd.DataFrame(signals_log)

    print("📊 Backtest complete.")

    # --- Equity Curve Plot ---
    if equity_curves:
        plt.figure(figsize=(10, 5))
        for ticker, curve in equity_curves.items():
            plt.plot(curve, label=ticker)
        plt.title(f"Equity Curves ({start_date.date()} → {end_date.date()})")
        plt.xlabel("Bars")
        plt.ylabel("Equity ($)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        send_message_report(df_results, sig_df, buf, start_date, end_date)
    else:
        send_message("⚠️ No equity curves to plot.")

    return df_results, sig_df



# ===== Telegram Reporting =====
def send_message_report(df_results: pd.DataFrame, df_signals: pd.DataFrame, equity_buf: io.BytesIO, start_date, end_date):
    """Send backtest summary + last signals + chart to Telegram."""
    summary = f"📈 *Backtest Report* ({start_date} → {end_date})\n\n"
    for _, row in df_results.iterrows():
        summary += (
            f"💠 *{row['Ticker']}*\n"
            f"• PnL: ${row['Total PnL']:,}\n"
            f"• Win Rate: {row['Win Rate']}%\n"
            f"• Sharpe: {row['Sharpe']}\n"
            f"• Max DD: {row['Max Drawdown %']}%\n"
            f"• Final Equity: ${row['Final Equity']:,}\n\n"
        )

    if not df_signals.empty:
        recent = df_signals.tail(10)
        summary += "📊 *Recent Signals:*\n"
        for _, s in recent.iterrows():
            summary += f"• {s['Ticker']} {s['Signal']} @ {s['Price']}\n"

    send_message(summary)
    #send_photo(equity_buf)
    print("📩 Backtest report + chart sent to Telegram.")


#########################################################
############  BACK TESTING LOGIC ########################
############    ENDING           ########################
#########################################################


#########################################################
############  BACK TESTING LOGIC to find best filter ####
############    STARTING         ########################
#########################################################


# Define ranges for filter parameters
PARAM_GRID = {
    'ATR_threshold': [0.05, 0.1],
    'SMA_score': [0.1, 0.25],
    'RSI_oversold': [25, 30],
    'RSI_overbought': [60, 70],
    'RSI_score': [0.15, 0.2],
    'STOP_ATR_MULT': [1.5],
    'TP_ATR_MULT': [3.0],
    'AI_score': [0.2,0.3],
    'Score_threshold': [0.05, 0.1],
    'Cooldown_minutes': [15],
}

def generate_filter_combinations(grid):
    keys = list(grid.keys())
    values = list(grid.values())
    for combo in product(*values):
        yield dict(zip(keys, combo))


def optimize_filters(days=180, metric='Sharpe', model=None):
    best_result = None
    best_filters = None
    all_combos = list(generate_filter_combinations(PARAM_GRID))
    
    print(f"Testing {len(all_combos)} filter combinations...")
    
    for idx, filt in enumerate(all_combos, 1):
        print(f"Testing combination {idx}/{len(all_combos)}: {filt}")
        df_results, _ = backtest_strategy(model=model, filters=filt, days=days)

        if df_results.empty:
            continue
        
        # Evaluate metric (example: total PnL or Sharpe)
        metric_val = df_results['Total PnL'].sum() if metric == 'PnL' else df_results['Total PnL'].sum()  # placeholder
        
        if best_result is None or metric_val > best_result:
            best_result = metric_val
            best_filters = filt
    
    print("✅ Best filter combination found:")
    print(best_filters)
    send_message(best_filters)
    return best_filters



###########################################################################
############  BACK TESTING LOGIC to find best filter ######################
############                END        ####################################
##########################################################################

###########################################################################
############  POST MATKET DAILY SUMMARY AT 04:00 pm EST ######################
############                START        ####################################
##########################################################################

# ================================
# Daily Summary Helper
# ================================
import matplotlib.pyplot as plt
import io
import base64

def post_trading_analysis(tickers):
    """
    Performs end-of-day cleanup, equity updates,
    backups, and logging.
    """
    try:
        positions, equity, last_update = load_equity_sheet()

        # 2️⃣ Update equity sheet with last update timestamp
        now_iso = datetime.now(EST).isoformat()
        for t in tickers:
            save_equity_sheet(t, equity.get(t, 1000), now_iso)

        # 3️⃣ Generate summary
        total_equity = sum(equity.values())
        summary_lines = ["📊 End-of-Day Trading Summary\n"]
        for t in sorted(equity.keys()):
            #ai_info = ai_results.get(t, {})
            summary_lines.append(
                f"{t}: Equity={equity[t]:.2f} "
            )
        summary_lines.append(f"\n💰 Total Equity: {total_equity:.2f}")
        msg = "\n".join(summary_lines)
        send_message(msg)
        log_message(msg)

        # 4️⃣ Backup data and models
        backup_folder = "backup_" + datetime.now().strftime("%Y%m%d")
        os.makedirs(backup_folder, exist_ok=True)
        # Save positions and equity
        pd.DataFrame([{"Ticker": t, "Equity": equity[t], "Position": positions[t]} for t in tickers]) \
            .to_csv(os.path.join(backup_folder, "positions_equity.csv"), index=False)
        log_message(f"💾 Backups saved in {backup_folder}")

    except Exception as e:
        log_message(f"❌ Failed during post-trading analysis: {e}")
        send_message(f"❌ Failed post-trading analysis: {e}")

def sanitize_tickers(tickers):
    """
    Converts a list of tickers to strings, handling tuples if they exist.
    """
    sanitized = []
    for t in tickers:
        if isinstance(t, tuple):
            sanitized.append(str(t[0]).upper().strip())  # take first element of tuple
        else:
            sanitized.append(str(t).upper().strip())
    return sanitized

def load_historical_data(ticker, period="6mo", interval="1d"):
    """
    Load historical OHLCV data for a ticker.
    """
    try:
        bars = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if bars.empty:
            print(f"⚠️ No data found for {ticker}")
            return None
        
        # Keep only relevant columns and convert to lowercase
        bars = bars[['Open', 'High', 'Low', 'Close', 'Volume']]
        bars.reset_index(inplace=True)
        return bars

    except Exception as e:
        print(f"⚠️ Error loading historical data for {ticker}: {e}")
        return None

def nightly_job(tickers):
    """
    Fully automated nightly routine:
    1️⃣ Post-trading analysis
    3️⃣ Equity & last update sheet refresh
    4️⃣ Data backup
    5️⃣ Readiness checks for next trading day
    """
    log_message("🌙 Starting nightly job...")
    print(tickers)
    try:
        # 1️⃣ Post-trading analysis
        log_message("📊 Running post-trading analysis...")
        post_trading_analysis(tickers)
        tickers = sanitize_tickers(tickers)

        # 3️⃣ Refresh equity sheet
        log_message("💰 Updating equity sheet...")
        positions, equity, last_update = load_equity_sheet()
        now_iso = datetime.now(EST).isoformat()
        for t in tickers:
            save_equity_sheet(t, equity.get(t, 1000), now_iso)

        # 4️⃣ Backup all data
        log_message("💾 Performing backups...")
        backup_folder = f"backup_{datetime.now().strftime('%Y%m%d')}"
        os.makedirs(backup_folder, exist_ok=True)
        pd.DataFrame([{"Ticker": t, "Equity": equity[t], "Position": positions[t]} for t in tickers]) \
            .to_csv(os.path.join(backup_folder, "positions_equity.csv"), index=False)
        log_message(f"✅ Backup saved to {backup_folder}")

        # 5️⃣ Readiness check for tomorrow
        log_message("🔎 Performing readiness checks for tomorrow...")

        log_message("🌙 Nightly job complete. System ready for next trading day!")

    except Exception as e:
        log_message(f"❌ Nightly job failed: {e}")
        send_message(f"❌ Nightly job failed: {e}")

###########################################################################
############  POST MATKET DAILY SUMMARY AT 04:00 pm EST ######################
############                START        ####################################
##########################################################################


########################################################################################
############   LIVE TRADING LOGIC AND HELPERS START ####################################
########################################################################################
POSITIONS = {t: None for t in tickers}  # Track open positions for live trading 
LAST_TRADE_TIME = {t: None for t in tickers}  # Cooldown tracking
# Risk and position tracking
# Globals used in execute_trade
positions = defaultdict(lambda: None)
equity = defaultdict(lambda: 1000)
last_trade_time = defaultdict(lambda: None)
RISK_PER_TRADE = 0.01       # Starting equity per ticker

EQUITY_WS = "TickerEquity"
equity_ws = sheet.worksheet(EQUITY_WS)

def load_equity_sheet():
    """
    Loads tickers, positions, equity, and last update time from Google Sheet.
    Returns:
        positions: dict[ticker] -> position info
        equity: dict[ticker] -> total equity
        last_update: dict[ticker] -> last updated datetime string
    """
    df = pd.DataFrame(equity_ws.get_all_records())
    positions = {}
    equity = {}
    last_update = {}

    for _, row in df.iterrows():
        ticker = row.get("Ticker")
        if not ticker:
            continue

        # Parse position safely
        pos_val = row.get("Position")
        positions[ticker] = eval(pos_val) if pos_val else None

        # Parse equity safely
        equity[ticker] = float(row.get("Equity", 1000))

        # Parse last update safely
        last_update[ticker] = row.get("LastUpdate", "")

    return positions, equity, last_update


def save_equity_sheet(ticker, equity, last_update=None):
    """
    Update or insert a single ticker equity record with timestamp
    """
    try:
        df = pd.DataFrame(equity_ws.get_all_records())

        # convert datetime to string
        if isinstance(last_update, datetime):
            last_update = last_update.isoformat()

        # update existing or append new
        if ticker in df['Ticker'].values:
            idx = df.index[df['Ticker'] == ticker][0]
            equity_ws.update(f"B{idx+2}", [[(equity)]])
            equity_ws.update(f"C{idx+2}", [[last_update]])
        else:
            equity_ws.append_row([ticker, (equity), last_update])

    except Exception as e:
        print(f"⚠️ Failed to update equity for {ticker}: {e}")

def init_equity(tickers, default_equity=1000):
    """
    Initialize equity per ticker from Google Sheet or create missing tickers.
    Adds a 'LastUpdate' column if not present.
    Returns:
        dict[ticker] -> equity value
    """
    equity = {}

    if sheet:
        # Try to open or create the worksheet
        try:
            ws = sheet.worksheet("TickerEquity")
        except gspread.WorksheetNotFound:
            ws = sheet.add_worksheet(title="TickerEquity", rows="100", cols="4")
            ws.append_row(["Ticker", "Equity", "LastUpdate"])

        # Fetch existing data
        existing_records = ws.get_all_records()
        existing_tickers = [r["Ticker"] for r in existing_records]

        # Initialize or append missing tickers
        for t in tickers:
            if t in existing_tickers:
                # Use existing equity
                rec = next(r for r in existing_records if r["Ticker"] == t)
                equity[t] = rec.get("Equity", default_equity)
            else:
                ws.append_row([t, default_equity, datetime.now().isoformat()])
                equity[t] = default_equity
    else:
        # Excel fallback
        file_path = "equity.xlsx"
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
        else:
            df = pd.DataFrame(columns=["Ticker", "Equity", "LastUpdate"])

        for t in tickers:
            if t in df["Ticker"].values:
                equity[t] = df.loc[df["Ticker"] == t, "Equity"].values[0]
            else:
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            [{"Ticker": t, "Equity": default_equity, "LastUpdate": datetime.now()}]
                        ),
                    ],
                    ignore_index=True,
                )
                equity[t] = default_equity
        df.to_excel(file_path, index=False)

    return equity


# ---------------------------
# Execute Trade
# ---------------------------
def execute_trade(ticker, signal):
    """
    Execute a trade based on the signal, update positions, equity, last update,
    and send detailed Telegram message with reasoning.
    """
    global positions, equity, last_trade_time

    now = datetime.now().isoformat()

    # Initialize equity if missing
    if ticker not in equity:
        equity[ticker] = 1000

    pos = positions.get(ticker)
    risk = signal.get('Risk', 0.0) or 1  # avoid division by zero
    qty = max(1, int(RISK_PER_TRADE * equity[ticker] / risk))

    # --- OPEN TRADE ---
    if pos is None and signal['signal'] in ("LONG", "SHORT"):
        positions[ticker] = {
            "entry_price": signal['entry_price'],
            "direction": 1 if signal['signal'] == 'LONG' else -1,
            "qty": qty,
            "stop": signal['stop'],
            "tp": signal['tp']
        }
        last_trade_time[ticker] = now

        msg = (
            f"🟢 Opened {signal['signal']} {ticker}\n"
            f"Entry: {signal['entry_price']:.2f} | Qty: {qty}\n"
            f"Stop: {signal['stop']:.2f} | TP: {signal['tp']:.2f}\n"
            f"Reasoning: SMA_short={'above' if signal['entry_price']>signal['stop'] else 'below'} SMA_long, "
            f"RSI={signal.get('RSI', 'N/A')}, ATR={signal.get('ATR', 'N/A')}"
        )
        print(msg)
        send_telegram_safe(msg)

        log_trade({
            "Ticker": ticker, "Date": now, "Signal": signal['signal'],
            "Entry": signal['entry_price'], "Exit": "",
            "PnL": 0, "Qty": qty, "Equity": equity[ticker]
        })

        save_equity_sheet(ticker, equity[ticker], now)

    # --- CLOSE TRADE ---
    elif pos is not None:
        # Determine actual exit price
        exit_price = signal.get('close', signal['entry_price'])

        # Check exit condition
        exit_cond = (
            (pos['direction'] == 1 and (exit_price <= pos['stop'] or exit_price >= pos['tp'])) or
            (pos['direction'] == -1 and (exit_price >= pos['stop'] or exit_price <= pos['tp']))
        )

        if exit_cond:
            pnl = (exit_price - pos['entry_price']) * pos['direction'] * pos['qty']
            equity[ticker] += pnl

            msg = (
                f"🔴 Closed {ticker}\n"
                f"Entry: {pos['entry_price']:.2f} | Exit: {exit_price:.2f} | Qty: {pos['qty']}\n"
                f"PnL: {pnl:.2f} | New Equity: {equity[ticker]:.2f}\n"
                f"Reasoning: Stop {'hit' if (exit_price <= pos['stop'] if pos['direction']==1 else exit_price >= pos['stop']) else 'target reached'}"
            )
            print(msg)
            send_telegram_safe(msg)

            positions[ticker] = None
            log_trade({
                "Ticker": ticker, "Date": now, "Signal": "CLOSE",
                "Entry": pos['entry_price'], "Exit": exit_price,
                "PnL": pnl, "Qty": pos['qty'], "Equity": equity[ticker]
            })

            save_equity_sheet(ticker, equity[ticker], now)


###########################################################
#################################################


#################################################
##############################################################


# Create client
def create_alpaca_client():
    if not ALPACA_KEY or not ALPACA_SECRET:
        log_message("❌ Alpaca API keys not found in environment. Order execution disabled.")
        return None
    try:
        client = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL, api_version='v2')
        account = client.get_account()
        log_message(f"🔌 Connected to Alpaca (paper={ALPACA_BASE_URL.endswith('paper-api.alpaca.markets')}). Account status: {account.status}")
        return client
    except Exception as e:
        log_message(f"⚠️ Error creating Alpaca client: {e}")
        return None

ALPACA = create_alpaca_client()

# Utility: is market open
def is_market_open(client) -> bool:
    try:
        clock = client.get_clock()
        return clock.is_open
    except Exception as e:
        log_message(f"⚠️ Could not fetch Alpaca clock: {e}. Assuming market closed.")
        return False

# Utility: get current buying power / equity
def get_account_equity(client) -> Optional[float]:
    try:
        acct = client.get_account()
        # Use cash or equity depending on margin
        bp = float(acct.cash) if acct.cash is not None else float(acct.equity)
        return bp
    except Exception as e:
        log_message(f"⚠️ Could not fetch account equity: {e}")
        return None

# Helper: check existing open orders for ticker
def has_open_order_for_symbol(client, symbol: str) -> bool:
    try:
        open_orders = client.list_orders(status='open', symbols=[symbol])
        return len(open_orders) > 0
    except Exception as e:
        # fallback: return False but log
        log_message(f"⚠️ Could not check open orders for {symbol}: {e}")
        return False

# Place an order (bracket order with stop loss & take profit)
def place_bracket_order(
    client,
    symbol: str,
    side: str,                 # 'buy' or 'sell' (sell for short)
    qty: int,
    limit_price: Optional[float] = None,
    stop_loss_price: Optional[float] = None,
    take_profit_price: Optional[float] = None,
    time_in_force: str = "day"
):
    """
    Submits a bracket order:
    - 'side' should be 'buy' for long entries, 'sell' for short entries.
    """
    if client is None:
        log_message("⚠️ Alpaca client not initialized; cannot place orders.")
        return None

    if qty < 1:
        log_message(f"⚠️ Computed qty < 1 for {symbol}; skipping order.")
        return None

    if has_open_order_for_symbol(client, symbol):
        log_message(f"⚠️ There is already an open order for {symbol}; skipping to avoid duplicates.")
        return None

    try:
        order_params = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": "market" if limit_price is None else "limit",
            "time_in_force": time_in_force,
            "order_class": "bracket",
            "take_profit": {"limit_price": str(take_profit_price)} if take_profit_price is not None else None,
            "stop_loss": {"stop_price": str(stop_loss_price)} if stop_loss_price is not None else None
        }

        # Remove None keys (Alpaca will reject if None present)
        order_params = {k: v for k, v in order_params.items() if v is not None}

        log_message(f"➡️ Submitting bracket order: {order_params}")
        submitted = client.submit_order(**order_params)
        log_message(f"✅ Order submitted for {symbol}: id={submitted.id} side={side} qty={qty}")
        return submitted
    except Exception as e:
        log_message(f"❌ Error submitting order for {symbol}: {e}")
        return None

# Position sizing: risk-based using ATR
def compute_position_size(equity: float, risk_per_trade: float, atr: float, entry_price: float, atr_multiplier: float = 1.0) -> int:
    """
    equity: total account cash/equity (float)
    risk_per_trade: fraction of equity to risk per trade (e.g. 0.01)
    atr: current ATR (in price units)
    entry_price: current price of the symbol
    atr_multiplier: multiple of ATR to use as risk per share (e.g. 1*ATR or 1.5*ATR)
    Returns integer share qty (floor), minimum 0.
    """
    if equity is None or equity <= 0:
        return 0
    if atr is None or atr <= 0:
        return 0

    risk_dollars = equity * risk_per_trade
    risk_per_share = atr * atr_multiplier
    if risk_per_share <= 0:
        return 0

    qty = math.floor(risk_dollars / risk_per_share)
    # limit by buying power relative to entry price (conservative)
    max_affordable = math.floor(equity / entry_price) if entry_price > 0 else qty
    qty = max(0, min(qty, max_affordable))
    return qty

# Integrate order execution into live loop
# Add these config params near other top-level constants
RISK_PER_TRADE = 0.01        # 1% of equity risk per trade
ATR_MULTIPLIER_STOP = 1.5    # stop = entry - ATR*1.5 (for longs)
ATR_MULTIPLIER_TP = 3.0      # take-profit distance multiplier (e.g. 3x ATR)
MIN_ENTRY_QTY = 1            # minimum shares to place an order
COOLDOWN_SECONDS = 60 * 5    # optional per-ticker cooldown to avoid rapid reorders

_last_order_time = defaultdict(lambda: None)  # track last order time per ticker

def execute_signal_via_alpaca(ticker: str, signal: str, last_bar: dict, model=None):
    """
    Given a BUY/SELL signal, compute qty and place a bracket order via Alpaca.
    """
    client = ALPACA
    if client is None:
        log_message("⚠️ Alpaca client not configured; skipping execution.")
        return None

    # Only trade while market open
    if not is_market_open(client):
        log_message("🕒 Market is closed; skipping order placement.")
        return None

    # Prevent rapid reorders per ticker
    last_time = _last_order_time.get(ticker)
    if last_time is not None:
        if (datetime.now(EST) - last_time).total_seconds() < COOLDOWN_SECONDS:
            log_message(f"⏳ Cooldown active for {ticker}, skipping signal.")
            return None

    # Account equity / buying power for sizing
    equity = get_account_equity(client)
    if equity is None:
        log_message("⚠️ No account equity available; skipping order.")
        return None

    # Extract price and ATR from last_bar (ensure keys match your source)
    entry_price = last_bar.get("close") or last_bar.get("close") or last_bar.get("price")
    atr = last_bar.get("ATR")
    if entry_price is None or atr is None:
        log_message(f"⚠️ Missing entry price or ATR for {ticker}; cannot size position.")
        return None

    # Compute qty
    if signal.upper() == "BUY":
        qty = compute_position_size(equity, RISK_PER_TRADE, atr, entry_price, ATR_MULTIPLIER_STOP)
        side = "buy"
        stop_price = max(0.01, entry_price - atr * ATR_MULTIPLIER_STOP)
        take_profit_price = entry_price + atr * ATR_MULTIPLIER_TP
    elif signal.upper() == "SELL":
        # For SELL signals, attempt a short if account allows margin.
        qty = compute_position_size(equity, RISK_PER_TRADE, atr, entry_price, ATR_MULTIPLIER_STOP)
        side = "sell"   # a sell order will short if position not owned and account allows
        stop_price = min(999999, entry_price + atr * ATR_MULTIPLIER_STOP)
        take_profit_price = entry_price - atr * ATR_MULTIPLIER_TP
    else:
        log_message(f"⚠️ Unknown signal '{signal}' for {ticker}")
        return None

    if qty < MIN_ENTRY_QTY:
        log_message(f"⚠️ Computed qty for {ticker} < {MIN_ENTRY_QTY} (qty={qty}); skipping.")
        return None

    # Place bracket order (market entry + TP & SL)
    order = place_bracket_order(
        client=client,
        symbol=ticker,
        side=side,
        qty=qty,
        limit_price=None,  # use market entry
        stop_loss_price=round(stop_price, 2),
        take_profit_price=round(take_profit_price, 2),
        time_in_force="day"
    )

    if order:
        _last_order_time[ticker] = datetime.now(EST)
    return order
###################################################################################################
##################   ADDITONAL FUNC FOR LIVE TRADING MONITORING    ################################
######################    STARTTED                   ##############################################
###################################################################################################

# -----------------------------
# DATA FETCHING
# -----------------------------
def fetch_intraday_data(ticker):
    try:
        file_path = f"data/{ticker}_latest.csv"

        if os.path.exists(file_path):
            bars = pd.read_csv(file_path)
        else:
            print(f"⚠️ No data file found for {ticker}, fetching fresh data...")
            bars = get_bars_cached(ticker)  # Replace with your own fetch function
        return bars
    except Exception as e:
        send_message(f"Error fetching data for {ticker}: {e}")
        return None

# -----------------------------
# VOLUME SPIKE DETECTION
# -----------------------------
def check_volume_spike(df):
    if df is None or len(df) < 20:
        return False
    avg_vol = df['volume'].rolling(20).mean().iloc[-1]
    current_vol = df['volume'].iloc[-1]
    return current_vol > 2 * avg_vol

# -----------------------------
# NEWS / SENTIMENT
# -----------------------------
def get_news_sentiment(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}"
    try:
        resp = requests.get(url).json()
        articles = resp.get('articles', [])
        sentiment = 0
        positive_words = ["gain", "beat", "strong", "up"]
        negative_words = ["loss", "miss", "weak", "down"]
        for article in articles[:5]:
            content = (article.get('title', '') + article.get('description', '')).lower()
            sentiment += sum(1 for w in positive_words if w in content)
            sentiment -= sum(1 for w in negative_words if w in content)
        return sentiment / 5
    except Exception as e:
        send_message(f"Error fetching news for {ticker}: {e}")
        return 0





#################################################################
##################   LIVE TRADING MONITORING    ################################
######################    STARTTED                   ############################
#################################################################
# -------------------------------------------------------
# Example integration point in your live_trading_session:
# after obtaining signal from core_signal_live(), call:
#
#    if signal is not None:
#        execute_signal_via_alpaca(ticker, signal, last_bar, model=ai_model)
#
# -------------------------------------------------------
def live_trading_session():
    EST = pytz.timezone("US/Eastern")
    init_equity(tickers)
   
    now_est = datetime.now(EST)

    # Stop the bot at or after 4:30 PM EST
    market_close = now_est.replace(hour=16, minute=30, second=0, microsecond=0)
    if now_est >= market_close:
        log_message("🛑 Market closed — ending live trading loop.")
        return

    log_message("🚀 Starting live paper trading loop...")

    try:
        positions, equity, last_update = load_equity_sheet()
        last_signal = defaultdict(lambda: None)

        for ticker in tickers:
            try:
                # Fetch latest bars
                last_bar, bars = fetch_last_bar(ticker)
                df = fetch_intraday_data(ticker)
                # ✅ Check if bars are missing or empty
                if bars is None or len(bars) == 0 or last_bar is None:
                    log_message(f"⚠️ No bars returned or invalid data for {ticker}")
                    continue

                # ✅ Extract technical indicators safely
                sma_short = last_bar.get('SMA_short')
                sma_long = last_bar.get('SMA_long')
                atr = last_bar.get('ATR')
                rsi = last_bar.get('RSI')

                # ✅ Skip tickers with incomplete or NaN data
                if any(x is None or np.isnan(x) for x in [sma_short, sma_long, atr, rsi]):
                    log_message(f"⚠️ Missing indicator values for {ticker}, skipping.")
                    continue

                bars = calculate_indicators(bars)
                last_bar = bars.iloc[-1]
                signal = core_signal_live(
                    last_bar=last_bar,
                    sma_short=last_bar['SMA_short'],
                    sma_long=last_bar['SMA_long'],
                    atr=last_bar['ATR'],
                    rsi=last_bar['RSI'],
                    ticker=ticker,
                    historical_bars=bars,
                    current_time=datetime.now()
                    )

                if signal['signal'] != 'FLAT':
                    # Prevent duplicate trades
                    if positions[ticker] is None or positions[ticker]['entry_price'] != signal['entry_price']:
                        execute_trade(ticker, signal)

                    # 🚀 Execute trade via Alpaca
                # if signal is not None and signal in ["BUY", "SELL"]:
                    #   execute_signal_via_alpaca(
                    #      ticker=ticker,
                    #       signal=signal,
                    #       last_bar=last_bar,
                    #       model=ai_model
                    #   )
                
                    # -------------------------
                    # Volume spike detection
                    # -------------------------
                if check_volume_spike(df):
                    send_message(f"🔹 Volume spike detected for {ticker}")

                # -------------------------
                # News / sentiment
                # -------------------------
                sentiment = get_news_sentiment(ticker)
                if sentiment < -0.5:
                    send_message(f"⚠️ Negative sentiment for {ticker}, consider avoiding trades")

                   
            except Exception as inner_e:
                log_message(f"⚠️ Error processing {ticker}: {inner_e}")
                continue

    except Exception as e:
        log_message(f"⚠️ Error during live loop: {e}")
   
#####################################################################################
############### News Integration and Analysis #######################################
#####################################################################################

# -------------------------------------------------
# Yahoo Finance RSS Fetch
# -------------------------------------------------
def fetch_yahoo_news(limit=20):
    """Fetch latest Yahoo Finance news from RSS."""
    feed = feedparser.parse(YAHOO_NEWS_FEED)
    news_items = []
    for entry in feed.entries[:limit]:
        news_items.append({
            "title": entry.title,
            "link": entry.link,
            "summary": entry.get("summary", ""),
            "published": entry.get("published", ""),
        })
    return news_items


# -------------------------------------------------
# Utility functions
# -------------------------------------------------
def extract_tickers_from_text(text):
    """Extract ticker symbols from headline/summary text."""
    tickers = set()
    tickers.update(re.findall(r"\(([A-Z]{1,5})\)", text))
    tickers.update(re.findall(r"/quote/([A-Z]{1,5})", text))
    tickers.update(re.findall(r"\b([A-Z]{2,5})\b", text))
    # filter noise
    blacklist = {"THE", "AND", "FOR", "WITH", "FROM", "NASDAQ", "NYSE", "ETF", "WALL", "STREET"}
    return {t for t in tickers if t.isupper() and 1 < len(t) <= 5 and t not in blacklist}


def get_price(ticker):
    """Fetch current price from Yahoo Finance."""
    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={ticker}"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        return data["quoteResponse"]["result"][0]["regularMarketPrice"]
    except Exception:
        return None


def load_sent_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {"sent_ids": []}


def save_sent_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def send_to_telegram(message: str):
    """Send formatted message to Telegram chat."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    r = requests.post(url, json=payload, timeout=10)
    if not r.ok:
        print("❌ Telegram send failed:", r.text)


# -------------------------------------------------
# News Formatting and Summary
# -------------------------------------------------
def format_news_item(n):
    """Format one news item into a Telegram-friendly string."""
    sent = n["sentiment"]
    emoji = "🟢" if sent > 0.25 else "🔴" if sent < -0.25 else "⚪"

    tickers_text = ""
    if n["tickers"]:
        t_list = []
        for t in n["tickers"]:
            price = get_price(t)
            if price:
                t_list.append(f"{t} (${price:.2f})")
            else:
                t_list.append(t)
        tickers_text = " | ".join(t_list)

    return f"{emoji} <a href='{n['link']}'>{n['title']}</a>\n🧩 {tickers_text}\n"


def generate_market_mood_summary(news_items):
    """Generate a short summary like '6 bullish, 3 bearish, 2 neutral'."""
    bullish = sum(1 for n in news_items if n["sentiment"] > 0.25)
    bearish = sum(1 for n in news_items if n["sentiment"] < -0.25)
    neutral = len(news_items) - bullish - bearish
    return f"📊 <b>Market Mood:</b> 🟢 {bullish} bullish | 🔴 {bearish} bearish | ⚪ {neutral} neutral\n"

def analyse_news_daily():
    cache = load_sent_cache()
    sent_ids = set(cache.get("sent_ids", []))
    news_items = fetch_yahoo_news(limit=20)

    new_msgs = []
    for n in news_items:
        uid = n["link"]
        if uid in sent_ids:
            continue  # skip repeated news

        # sentiment
        full_text = f"{n['title']} {n['summary']}"
        sent_score = analyzer.polarity_scores(full_text)["compound"]

        # extract tickers
        tickers = extract_tickers_from_text(full_text)

        n["sentiment"] = sent_score
        n["tickers"] = list(tickers)
        new_msgs.append(n)
        sent_ids.add(uid)

    if not new_msgs:
        print("✅ No new news to send.")
        return

    # Market mood summary
    summary_line = generate_market_mood_summary(new_msgs)

    # Format all messages
    msg = "<b>📰 Latest Market News (Yahoo Finance)</b>\n"
    msg += summary_line + "\n"
    for n in new_msgs[:8]:
        msg += format_news_item(n)
    msg += f"\n⏰ Updated: {time.strftime('%Y-%m-%d %H:%M EST', time.gmtime())}"

    send_to_telegram(msg)
    print(f"✅ Sent {len(new_msgs)} new articles.")
    save_sent_cache({"sent_ids": list(sent_ids)})

# ================================
# Entrypoint
# ================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python moving_average_bot.py [morning|live|backtest]")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "morning":
        update_morning_scanner_with_ai()
     
    elif mode == "live":
        try:
            live_trading_session()
        except Exception as e:
            logging.error(f"Error in live trading loop: {e}")
            send_message(f"❌ Live trading error: {e}")

    elif mode == "analysis":
        nightly_job(tickers)

    elif mode == "backtest":
        best_filters = optimize_filters()
        print("Best filters for live trading:", best_filters)
    
    elif mode == "news":
        analyse_news_daily()


    elif mode == "backtest":
        best_filters = optimize_filters()
        print("Best filters for live trading:", best_filters)
