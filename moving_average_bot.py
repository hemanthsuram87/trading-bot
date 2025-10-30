import os
import time
import pytz
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time as dt_time,UTC
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import logging
# TA libs (some functions use ta)
import ta
from ta.trend import ADXIndicator
import yfinance as yf
# Alpaca
import alpaca_trade_api as tradeapi
from finvizfinance.screener.overview import Overview

import matplotlib.pyplot as plt
from ta.trend import ADXIndicator, MACD
from ta.momentum import RSIIndicator

import joblib
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ta.trend import ADXIndicator
import io

import html
import re
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# ================= CONFIG =================
ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# Telegram config
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
api_key = os.getenv("FMP_API_KEY")
NEWS_API_KEY = "fa1d035152ac4016bd4d8647244d1748"
api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, base_url=ALPACA_BASE_URL)

EST = pytz.timezone("US/Eastern")

# Optional email alerts
EMAIL_SMTP = os.getenv("EMAIL_SMTP")  # e.g. smtp.gmail.com:587
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO = os.getenv("EMAIL_TO")
EMAIL_USER = os.getenv("EMAIL_USER")  # optional
EMAIL_PASS = os.getenv("EMAIL_PASS")  # optional

# Auto close settings
CLOSE_LOSS_PCT = float(os.getenv("CLOSE_LOSS_PCT", "-5"))  # negative percent threshold to close (e.g. -5 -> -5%)
MAX_HOLD_MINUTES = int(os.getenv("MAX_HOLD_MINUTES", "240"))  # max minutes to keep a position

SMA_SHORT = int(os.getenv("SMA_SHORT", "20"))
SMA_LONG = int(os.getenv("SMA_LONG", "50"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "1"))
SIGNAL_EXPIRY_DAYS = int(os.getenv("SIGNAL_EXPIRY_DAYS", "5"))

LOG_DIR = "logs"
SIGNAL_DIR = "signals"
MODEL_DIR = "models"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SIGNAL_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# logging
EST = pytz.timezone("US/Eastern")

# Auto close settings
CLOSE_LOSS_PCT = float(os.getenv("CLOSE_LOSS_PCT", "-5"))  # negative percent threshold to close (e.g. -5 -> -5%)
MAX_HOLD_MINUTES = int(os.getenv("MAX_HOLD_MINUTES", "240"))  # max minutes to keep a position

SMA_SHORT = 20 
SMA_LONG = 50 
ATR_PERIOD = 14 
ATR_MULTIPLIER = 1 
SIGNAL_EXPIRY_DAYS = 2

BACKTEST_DAYS = 60  # Adjustable, conservative by default
MODEL_DIR = "models/sklearn/"
LOG_DIR = "logs"
SIGNAL_DIR = "signals"
MODEL_DIR = "models"

INITIAL_CAPITAL = 10000
COMMISSION = 0.001  # 0.1%
RISK_PER_TRADE = 0.02  # 2% risk per trade


os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SIGNAL_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# logging
EST = pytz.timezone("US/Eastern")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
def log_message(msg):
    ts = datetime.now(EST).strftime("[%Y-%m-%d %H:%M:%S]")
    logging.info(f"{ts} {msg}")

# Alpaca client
api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, base_url=ALPACA_BASE_URL)

# Google Sheets
if os.path.exists("google_creds.json"):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("google_creds.json", scope)
        client = gspread.authorize(creds)
        sheet = client.open("Daily_stocks")
        tickers_ws = sheet.worksheet("Tickers")
        tickers = tickers_ws.col_values(1)
    except Exception as e:
        log_message(f"⚠️ Google Sheets init failed: {e}")
        tickers = []
else:
    tickers = []
    log_message("⚠️ google_creds.json not found; continuing without Sheets.")

# Signals persistence (single file)
SIGNAL_FILE = os.path.join(SIGNAL_DIR, "sent_signals_master.txt")

def parse_signal_date(signal_id):
    try:
        # Format: TICKER-BUY-YYYY-MM-DD HH:MM:SS
        # rsplit safe in case ticker contains dashes
        timestamp_str = signal_id.rsplit("-", 1)[-1]
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

def cleanup_old_signals(signals):
    now = datetime.now()
    valid = set()
    for s in signals:
        d = parse_signal_date(s)
        if d is None:
            valid.add(s)
        elif now - d <= timedelta(days=SIGNAL_EXPIRY_DAYS):
            valid.add(s)
    return valid

def load_sent_signals():
    if not os.path.exists(SIGNAL_FILE):
        return set()
    with open(SIGNAL_FILE, "r") as f:
        signals = set(line.strip() for line in f if line.strip())
    cleaned = cleanup_old_signals(signals)
    if cleaned != signals:
        save_all_signals(cleaned)
    return cleaned

def save_sent_signal(signal_id):
    os.makedirs(os.path.dirname(SIGNAL_FILE), exist_ok=True)
    with open(SIGNAL_FILE, "a") as f:
        f.write(signal_id + "\n")

def save_all_signals(signals):
    os.makedirs(os.path.dirname(SIGNAL_FILE), exist_ok=True)
    with open(SIGNAL_FILE, "w") as f:
        for s in sorted(signals):
            f.write(s + "\n")

sent_signals = load_sent_signals()

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
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "Markdown"}
        try:
            r = requests.post(url, data=data, timeout=10)
            if r.status_code != 200:
                log_message(f"❌ Telegram failed (chunk {idx}/{len(chunks)}): {r.status_code} {r.text}")
            else:
                log_message(f"✅ Telegram sent (chunk {idx}/{len(chunks)})")
        except Exception as e:
            log_message(f"❌ Telegram exception: {e}")



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

# ================= AI FORECAST (safeguarded) =================
tf.random.set_seed(42)
np.random.seed(42)

def deep_learning_forecast(ticker, bars, sheet=None, lookback=60, forecast_steps=1, retrain=False):
    try:
        # --- Features ---
        df = bars[['open', 'high', 'low', 'close', 'volume']].dropna().reset_index(drop=True)
        if len(df) < lookback + forecast_steps + 1:
            log_message(f"⚠️ Not enough data for LSTM {ticker}")
            return None

        # Compute returns
        df['return'] = df['close'].pct_change().fillna(0)
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'return']

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[feature_cols].values)

        X, y = [], []
        for i in range(lookback, len(scaled) - forecast_steps):
            X.append(scaled[i-lookback:i])
            # classify next step return as direction
            next_return = df['return'].iloc[i + forecast_steps]
            y.append(1 if next_return > 0 else 0)  # 1 = bullish, 0 = bearish
        X, y = np.array(X), np.array(y)
        if X.size == 0:
            return None

        # Walk-forward style split (last 20% for validation)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # --- Model path ---
        model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm_trend.h5")
        model = None
        if os.path.exists(model_path) and not retrain:
            try:
                model = load_model(model_path, compile=False)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                log_message(f"✅ Loaded model for {ticker}")
            except Exception as e:
                log_message(f"⚠️ Model load failed ({ticker}): {e}")
                model = None

        if model is None:
            model = Sequential([
                Bidirectional(LSTM(64, return_sequences=True), input_shape=(lookback, len(feature_cols))),
                Dropout(0.2),
                Bidirectional(LSTM(32)),
                Dropout(0.2),
                Dense(1, activation='sigmoid')  # output probability
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=0, callbacks=[es])
            model.save(model_path)
            log_message(f"💾 Model trained & saved for {ticker}")

        # --- Forecast ---
        last_seq = scaled[-lookback:].reshape((1, lookback, len(feature_cols)))
        prob = model.predict(last_seq, verbose=0)[0][0]
        trend = "BULLISH" if prob > 0.5 else "BEARISH"
        confidence = round(prob*100, 2) if trend=="BULLISH" else round((1-prob)*100,2)
        current_price = df['close'].iloc[-1]

        # Approximate forecast price using last close + mean return
        forecast_price = current_price * (1 + df['return'].iloc[-lookback:].mean())

        # --- Save to sheet if available ---
        if sheet:
            try:
                try:
                    ws = sheet.worksheet("AI_Forecast")
                except Exception:
                    ws = sheet.add_worksheet(title="AI_Forecast", rows="1000", cols="10")
                ws.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                               str(ticker), float(round(current_price,2)), float(round(forecast_price,2)), str(trend), float(round(confidence))])
            except Exception as e:
                log_message(f"⚠️ Sheets update failed: {e}")

        return {
            "ticker": ticker,
            "trend": trend,
            "confidence": confidence,
            "current": round(current_price, 2),
            "forecast": round(forecast_price, 2)
        }

    except Exception as e:
        log_message(f"⚠️ AI forecast failed for {ticker}: {e}")
        return None


# ================= SIGNAL ANALYSIS (core) =================
def analyze_ticker(ticker, bars):
    if bars.empty or len(bars) < SMA_LONG:
        log_message(f"⚠️ Not enough bars for {ticker}")
        return

    bars = safe_tz_localize_and_convert(bars.copy())

    # Ensure numeric columns exist
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in bars.columns:
            log_message(f"⚠️ Missing {col} for {ticker}")
            return

    # Compute indicators on dataframe (not single scalars)
    bars["SMA20"] = bars["close"].rolling(SMA_SHORT).mean()
    bars["SMA50"] = bars["close"].rolling(SMA_LONG).mean()
    bars["RSI"] = compute_rsi(bars["close"], 14)
    bars["ATR"] = compute_atr(bars[["high", "low", "close"]], ATR_PERIOD)
    try:
        adx_indicator = ADXIndicator(high=bars["high"], low=bars["low"], close=bars["close"], window=14)
        bars["ADX"] = adx_indicator.adx()
    except Exception:
        bars["ADX"] = np.nan
    bars["MACD"], bars["MACD_Signal"] = compute_macd(bars["close"])
    bars["AvgVol"] = bars["volume"].rolling(20).mean()
    bars = bars.reset_index()
    bars["TradeTime"] = bars["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Higher timeframe confirmation safely
    higher_trend = "unknown"
    try:
        higher_bars = api.get_bars(ticker, tradeapi.TimeFrame.Hour, limit=100, feed="iex").df
        higher_bars = safe_tz_localize_and_convert(higher_bars)
        higher_bars["SMA20"] = higher_bars["close"].rolling(20).mean()
        higher_bars["SMA50"] = higher_bars["close"].rolling(50).mean()
        if len(higher_bars) >= 2:
            higher_trend = "bullish" if higher_bars["SMA20"].iloc[-1] > higher_bars["SMA50"].iloc[-1] else "bearish"
    except Exception:
        higher_trend = "unknown"

    signals_messages = []
    for i in range(1, len(bars)):
        prev, curr = bars.iloc[i-1], bars.iloc[i]
        if np.isnan(prev["SMA20"]) or np.isnan(prev["SMA50"]) or np.isnan(curr["SMA20"]) or np.isnan(curr["SMA50"]):
            continue

        atr_value = curr["ATR"] if not pd.isna(curr["ATR"]) else 0.0
        stop_distance = atr_value * ATR_MULTIPLIER
        volatility_pct = (atr_value / curr["close"]) * 100 if curr["close"] else 0.0
        volume_spike = curr["volume"] > 1.5 * (curr["AvgVol"] if not pd.isna(curr["AvgVol"]) else 1)

        rsi_cond_buy = curr["RSI"] > 55 if not pd.isna(curr["RSI"]) else False
        rsi_cond_sell = curr["RSI"] < 45 if not pd.isna(curr["RSI"]) else False
        adx_cond = curr["ADX"] > 25 if not pd.isna(curr["ADX"]) else False
        macd_bull = curr["MACD"] > curr["MACD_Signal"] if not pd.isna(curr["MACD"]) and not pd.isna(curr["MACD_Signal"]) else False
        macd_bear = curr["MACD"] < curr["MACD_Signal"] if not pd.isna(curr["MACD"]) and not pd.isna(curr["MACD_Signal"]) else False
        vol_cond = 0.5 < volatility_pct < 3
        volume_cond = volume_spike
        higher_cond_buy = higher_trend == "bullish"
        higher_cond_sell = higher_trend == "bearish"

        # BUY
        if prev["SMA20"] < prev["SMA50"] and curr["SMA20"] > curr["SMA50"]:
            signal_id = f"{ticker}-BUY-{curr['TradeTime']}"
            if signal_id not in sent_signals:
                conditions = {
                    "RSI > 55": rsi_cond_buy,
                    "ADX > 25": adx_cond,
                    "MACD bullish": macd_bull,
                    "ATR in range": vol_cond,
                    "Volume spike": volume_cond,
                    "Higher TF bullish": higher_cond_buy,
                }
                passed = [f"✅ {k}" for k,v in conditions.items() if v]
                failed = [f"❌ {k}" for k,v in conditions.items() if not v]
                reason = "\n".join(passed + failed)
                msg = (f"🟢 {ticker} BUY at {curr['close']:.2f} on {curr['TradeTime']}\n"
                       f"Reason: SMA20 crossed above SMA50\nATR={atr_value:.2f}, StopDist={stop_distance:.2f}\n{reason}")
                signals_messages.append(msg)
                sent_signals.add(signal_id)
                save_sent_signal(signal_id)

        # SELL
        elif prev["SMA20"] > prev["SMA50"] and curr["SMA20"] < curr["SMA50"]:
            signal_id = f"{ticker}-SELL-{curr['TradeTime']}"
            if signal_id not in sent_signals:
                conditions = {
                    "RSI < 45": rsi_cond_sell,
                    "ADX > 25": adx_cond,
                    "MACD bearish": macd_bear,
                    "ATR in range": vol_cond,
                    "Volume spike": volume_cond,
                    "Higher TF bearish": higher_cond_sell,
                }
                passed = [f"✅ {k}" for k,v in conditions.items() if v]
                failed = [f"❌ {k}" for k,v in conditions.items() if not v]
                reason = "\n".join(passed + failed)
                msg = (f"🔴 {ticker} SELL at {curr['close']:.2f} on {curr['TradeTime']}\n"
                       f"Reason: SMA20 crossed below SMA50\nATR={atr_value:.2f}, StopDist={stop_distance:.2f}\n{reason}")
                signals_messages.append(msg)
                sent_signals.add(signal_id)
                save_sent_signal(signal_id)

    if signals_messages:
        consolidated_msg = "📈 *Trade Signals Summary*\n\n" + "\n\n".join(signals_messages)
        send_message(consolidated_msg)
        log_message(f"📩 Sent signals for {ticker}")

# ===================== HELPERS =====================


def is_morning_window():
    """Run only between 9:30–11:30 AM Eastern."""
    now = datetime.now(pytz.timezone(TIMEZONE))
    start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end = now.replace(hour=11, minute=30, second=0, microsecond=0)
    return start <= now <= end

def fetch_latest_news(category,page_size,api_key):
    news_headlines = []
    url = f"https://newsapi.org/v2/everything?q={category}&language=en&pageSize={page_size}&apiKey={api_key}"
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

def safe_text(text):
    if not text:
        return ""
    return html.escape(str(text), quote=False)  # escapes < > & but leaves quotes

def get_top_gap_gainers(top_n=10):
    filters_dict = {
        'Current Volume': 'Over 500K',
        'Gap': 'Up 10%',
        'Industry':'Stocks only (ex-Funds)'
    }

    overview = Overview()
    overview.set_filter(filters_dict=filters_dict)
    df = overview.screener_view()

    if df.empty:
        print("No data found for top gainers.")
        return

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

    # Filter only if today's volume > average volume
    if 'Volume' in df.columns and 'Average Volume (3 Month)' in df.columns:
        df = df[df['Volume'] > df['Average Volume (3 Month)']]

    # Sort descending by Change and pick top N
    df = df.sort_values(by='Change', ascending=False).head(top_n)

    message = "<b>📊 Morning Movers Watchlist</b>\n\n"
    newsmessage="<b> News for above Top Gainers </b> \n"
    for _, row in df.iterrows():
        message += (
            f"🔹 Ticker: {row['Ticker']}\n"
            f"💲 Price: {row['Price']}\n"
            f"✅ Gap: {row['Change']:.2f}%\n"
            f"🎯 Volume: {row['Volume']}\n"
        )

        # Fetch latest news safely
        newslinks = fetch_latest_news(row['Ticker'], page_size=2, api_key=NEWS_API_KEY)
        if newslinks and newslinks != {'error': 'Failed to fetch news'}:
            for news_item in newslinks:
               title = html.escape(news_item.get('title') or "")
               link = html.escape(news_item.get('link') or "")
               newsmessage += (f"Ticker : {row['Ticker']} \n"
                    f"📰 {title} - {link}\n")
               print(newsmessage)
               
        message += "\n"  # Separate each ticker nicely
        newsmessage += "\n"
    send_telegram_safe(message)
    send_telegram_safe(newsmessage)
    print("\n✅ Sent Telegram alert and saved watchlist.")
   
# ================= MORNING SCAN TOP MOVERS AND LOOSERS  END =================

# ================= PREVIOUS DAY ANALYSIS =================
def previous_day_analysis():
    now = datetime.now(EST)
    today = now.date()
    weekday = today.weekday()
    if weekday >= 5:
        last_trading_day = today - timedelta(days=(weekday - 4))
        label = "Previous Trading Day (Friday)"
    else:
        last_trading_day = today
        label = "Current Trading Day"
    start_dt = datetime.combine(last_trading_day, dt_time(9,30), tzinfo=EST)
    end_dt = datetime.combine(last_trading_day, dt_time(16,0), tzinfo=EST)
    start_date = start_dt.strftime("%Y-%m-%dT%H:%M:%S-04:00")
    end_date = end_dt.strftime("%Y-%m-%dT%H:%M:%S-04:00")
    log_message(f"📊 Running {label} analysis {start_date} -> {end_date}")
    ai_forecasts = []
    for ticker in tickers:
        try:
            bars = api.get_bars(ticker, tradeapi.TimeFrame.Minute, start=start_date, end=end_date, adjustment="raw", feed="iex").df
            if bars.empty:
                log_message(f"No bars for {ticker}")
                continue
            analyze_ticker(ticker, bars)
            res = deep_learning_forecast(ticker, bars, sheet if 'sheet' in globals() else None)
            if res:
                ai_forecasts.append(res)
        except Exception as e:
            log_message(f"Error fetching bars for {ticker}: {e}")
    if ai_forecasts:
        s = "🤖 AI Forecasts:\n" + "\n".join([f"{f['ticker']}: {f['trend']} {f['forecast']}" for f in ai_forecasts])
        send_message(s)


# -------------------- FULL INTEGRATION: REGIME / FEATURES / MODEL / RISK / EXEC --------------------


DRY_RUN = os.getenv("DRY_RUN", "true").lower() in ("1", "true", "yes")
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.5"))  # percent of equity to risk
MAX_SHARES_PER_TRADE = int(os.getenv("MAX_SHARES_PER_TRADE", "1000"))
MIN_EQUITY_FOR_TRADING = float(os.getenv("MIN_EQUITY_FOR_TRADING", "1000"))

TRAIN_DAYS = int(os.getenv("TRAIN_DAYS", "90"))  # days of historical data for training
TRAIN_MIN_SAMPLES = int(os.getenv("TRAIN_MIN_SAMPLES", "120"))

# ---------------- Regime Detection ----------------
def detect_market_regime(bars):
    """
    Simple regime detection:
      - 'Bullish' if short SMA > long SMA and ADX > 20
      - 'Bearish' if short SMA < long SMA and ADX > 20
      - 'Sideways' otherwise
    Returns dict with type and confidence (0-100)
    """
    try:
        df = bars.copy()
        if len(df) < 60:
            return {"type": "Insufficient Data", "confidence": 0.0}

        df = safe_tz_localize_and_convert(df) if df.index.tz is None else df
        df["SMA20"] = df["close"].rolling(20).mean()
        df["SMA100"] = df["close"].rolling(100).mean()
        try:
            df["ADX"] = ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
        except Exception:
            df["ADX"] = np.nan

        sma20 = df["SMA20"].iloc[-1]
        sma100 = df["SMA100"].iloc[-1]
        adx = df["ADX"].iloc[-1] if not pd.isna(df["ADX"].iloc[-1]) else 0.0

        # basic rules
        if pd.isna(sma20) or pd.isna(sma100):
            return {"type": "Insufficient SMA", "confidence": 0.0}

        if sma20 > sma100 and adx > 20:
            conf = min(95, 40 + (adx))  # arbitrary mapping
            return {"type": "Bullish", "confidence": float(conf)}
        elif sma20 < sma100 and adx > 20:
            conf = min(95, 40 + (adx))
            return {"type": "Bearish", "confidence": float(conf)}
        else:
            conf = max(30, 60 - (abs(sma20 - sma100) / (sma100 + 1) * 100))
            return {"type": "Sideways", "confidence": float(conf)}
    except Exception as e:
        log_message(f"⚠️ detect_market_regime failed: {e}")
        return {"type": "Error", "confidence": 0.0}

# ---------------- Feature Engineering ----------------
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

# ---------------- Labels utility ----------------
def make_labels_for_direction(bars, horizon=5, threshold=0.001):
    """
    Binary label: 1 if price after 'horizon' bars increases by > threshold fraction, else 0.
    horizon: integer number of bars into the future (e.g., 5)
    threshold: fraction, e.g., 0.001 => 0.1%
    """
    df = bars.copy()
    df = df.reset_index(drop=True)
    df["future_close"] = df["close"].shift(-horizon)
    df["pct_future"] = (df["future_close"] - df["close"]) / df["close"]
    labels = (df["pct_future"] > threshold).astype(int)
    labels.index = df.index
    return labels.dropna()

# ---------------- Model storage helpers ----------------
def model_file_for(ticker):
    safe = ticker.replace("/", "_").replace("^", "_")
    return os.path.join(MODEL_DIR, f"{safe}_rf.joblib")

def train_and_save_model(ticker, feat_df, labels, force_retrain=False):
    """
    Trains a RandomForestClassifier and saves it to disk.
    Returns the trained model or None.
    """
    try:
        if feat_df.empty or labels.empty:
            log_message(f"⚠️ No data to train for {ticker}")
            return None

        # Align by index
        feat_df = feat_df.loc[labels.index.intersection(feat_df.index)]
        labels = labels.loc[feat_df.index]
        if len(feat_df) < TRAIN_MIN_SAMPLES:
            log_message(f"⚠️ Not enough samples to train {ticker} (n={len(feat_df)})")
            return None

        X = feat_df.values
        y = labels.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=2)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        joblib.dump(clf, model_file_for(ticker))
        log_message(f"💾 Trained model for {ticker} (acc={acc:.3f}) -> {model_file_for(ticker)}")
        return clf
    except Exception as e:
        log_message(f"❌ train_and_save_model failed for {ticker}: {e}")
        return None

def load_model_if_exists(ticker):
    path = model_file_for(ticker)
    try:
        if os.path.exists(path):
            return joblib.load(path)
    except Exception as e:
        log_message(f"⚠️ Failed to load model {path}: {e}")
    return None

# ---------------- Risk & Sizing ----------------
def calculate_position_size(account_equity, risk_pct, entry_price, stop_price, max_shares=None):
    """
    Fixed-fraction sizing using dollar risk:
      qty = floor( (equity * risk_pct/100) / (abs(entry-stop)) )
    Returns integer share qty (>=0).
    """
    try:
        if entry_price is None or stop_price is None or entry_price == stop_price:
            return 0
        dollar_risk = account_equity * (risk_pct / 100.0)
        per_share_risk = abs(entry_price - stop_price)
        if per_share_risk <= 0:
            return 0
        qty = int(math.floor(dollar_risk / per_share_risk))
        if max_shares:
            qty = min(qty, max_shares)
        return max(0, qty)
    except Exception as e:
        log_message(f"⚠️ calculate_position_size failed: {e}")
        return 0

# ---------------- Execution Engine ----------------
def execute_trade(action, symbol, qty, limit_price=None, client=None, dry_run=DRY_RUN):
    """
    Execute (or dry-run) a simple market order using Alpaca.
    Returns result dict with status and details.
    """
    result = {"symbol": symbol, "action": action, "qty": qty, "status": "skipped"}
    try:
        if qty <= 0:
            result["status"] = "no_shares"
            log_message(f"⚠️ Not executing {action} {symbol}: qty={qty}")
            return result

        if dry_run:
            result["status"] = "dry_run"
            log_message(f"🔁 Dry-run: {action} {qty}x {symbol} (no real order sent)")
            return result

        api_client = client if client is not None else api
        side = "buy" if action.upper().startswith("B") else "sell"
        if limit_price:
            order = api_client.submit_order(symbol=symbol, qty=qty, side=side, type="limit", time_in_force="day", limit_price=str(limit_price))
        else:
            order = api_client.submit_order(symbol=symbol, qty=qty, side=side, type="market", time_in_force="day")
        result["status"] = "sent"
        result["order"] = getattr(order, "_raw", str(order))
        log_message(f"✅ Sent order: {action} {qty} {symbol}")
        return result
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        log_message(f"❌ Order execution failed for {symbol}: {e}")
        return result

# ---------------- Upgraded live_trading_loop (user-friendly telegram messages) ----------------
def live_trading_loop():
    log_message("🚀 Starting upgraded live_trading_loop (ensemble + user-friendly messages)")
    now = datetime.now(EST)
    if now.weekday() >= 5:
        log_message("ℹ️ Weekend detected — skipping.")
        return

    # attempt to get account equity
    try:
        acct = api.get_account()
        equity = float(acct.equity)
    except Exception:
        equity = MIN_EQUITY_FOR_TRADING

    summary_messages = []
    for ticker in tickers:
        try:
            log_message(f"🔍 Fetching bars for {ticker}...")
            end = datetime.now(EST)
            start = end - timedelta(hours=6)
            bars = api.get_bars(ticker, tradeapi.TimeFrame.Minute, start=start.isoformat(), end=end.isoformat(), feed="iex").df
            if bars.empty or len(bars) < 60:
                log_message(f"⚠️ Insufficient bars for {ticker} (have {len(bars)})")
                continue

            feat_df, full_df = create_features(bars)
            # basic indicator snapshot
            last = full_df.iloc[-1]
            sma20 = float(full_df["SMA20"].iloc[-1]) if "SMA20" in full_df.columns else float("nan")
            sma50 = float(full_df["SMA50"].iloc[-1]) if "SMA50" in full_df.columns else float("nan")
            rsi = float(full_df["RSI"].iloc[-1]) if "RSI" in full_df.columns else float("nan")
            atr = float(full_df["ATR"].iloc[-1]) if "ATR" in full_df.columns else float("nan")
            macd_latest = float(full_df["MACD"].iloc[-1]) if "MACD" in full_df.columns else float("nan")
            macd_signal_latest = float(full_df["MACD_SIGNAL"].iloc[-1]) if "MACD_SIGNAL" in full_df.columns else float("nan")
            adx = float(full_df["ADX"].iloc[-1]) if "ADX" in full_df.columns and not pd.isna(full_df["ADX"].iloc[-1]) else float("nan")

            # regime
            regime = detect_market_regime(bars)

            # model prediction
            model = load_model_if_exists(ticker)
            ai_signal = "Neutral"
            ai_prob = 0.5
            if model is not None and not feat_df.empty:
                try:
                    latest_feat = feat_df.iloc[[-1]].values
                    pred_class = model.predict(latest_feat)[0]
                    proba = model.predict_proba(latest_feat)[0]
                    ai_signal = "Bullish" if pred_class == 1 else "Bearish"
                    ai_prob = float(max(proba))
                except Exception as e:
                    log_message(f"⚠️ Model predict failed for {ticker}: {e}")

            # ensemble scoring (simple weighted blend)
            # weights: SMA 0.25, MACD 0.2, RSI 0.15, ADX 0.1, Regime 0.1, AI 0.2
            score = 0.0
            reasons = []
            # SMA
            if not math.isnan(sma20) and not math.isnan(sma50):
                if sma20 > sma50:
                    score += 0.25; reasons.append("SMA20 > SMA50 (trend up)")
                else:
                    score -= 0.25; reasons.append("SMA20 < SMA50 (trend down)")
            # MACD
            if not math.isnan(macd_latest) and not math.isnan(macd_signal_latest):
                if macd_latest > macd_signal_latest:
                    score += 0.20; reasons.append("MACD positive (momentum up)")
                else:
                    score -= 0.20; reasons.append("MACD negative (momentum down)")
            # RSI
            if not math.isnan(rsi):
                if rsi < 40:
                    score += 0.15; reasons.append(f"RSI {rsi:.1f} (oversold bias)")
                elif rsi > 60:
                    score -= 0.15; reasons.append(f"RSI {rsi:.1f} (overbought caution)")
                else:
                    reasons.append(f"RSI {rsi:.1f} (neutral)")
            # ADX
            if not math.isnan(adx) and adx > 25:
                score *= 1.05; reasons.append(f"ADX {adx:.1f} (trend strong)")
            # regime
            if regime and isinstance(regime, dict):
                if regime["type"].lower().startswith("bull"):
                    score += 0.10; reasons.append("Market regime bullish")
                elif regime["type"].lower().startswith("bear"):
                    score -= 0.10; reasons.append("Market regime bearish")
            # AI
            if ai_signal == "Bullish":
                score += ai_prob * 0.20; reasons.append(f"AI bullish ({ai_prob*100:.1f}%)")
            elif ai_signal == "Bearish":
                score -= ai_prob * 0.20; reasons.append(f"AI bearish ({ai_prob*100:.1f}%)")
            else:
                reasons.append("AI neutral")

            # normalize to confidence 0..1
            confidence = min(abs(score), 1.0)

            # thresholds
            if score > 0.25:
                action = "BUY"
            elif score < -0.25:
                action = "SELL"
            else:
                action = "HOLD"

            # compute stop / tp using ATR
            stop_price = None
            take_profit = None
            if not math.isnan(atr) and atr > 0:
                if action == "BUY":
                    stop_price = last["close"] - 1.5 * atr
                    take_profit = last["close"] + 3.0 * atr
                elif action == "SELL":
                    stop_price = last["close"] + 1.5 * atr
                    take_profit = last["close"] - 3.0 * atr

            # sizing
            qty = 0
            if action in ("BUY", "SELL") and stop_price is not None:
                qty = calculate_position_size(equity, RISK_PER_TRADE_PCT, float(last["close"]), float(stop_price), max_shares=MAX_SHARES_PER_TRADE)

            # Execution (dry-run by default)
            exec_result = {"status": "skipped"}
            if action in ("BUY", "SELL") and qty > 0:
                exec_result = execute_trade(action, ticker, qty, limit_price=None, dry_run=DRY_RUN)

            # Format message
            pos_text = "No position"
            try:
                positions = {p.symbol: p for p in api.list_positions()}  # may raise
                pos_text = f"{positions[ticker].qty} @ avg ${float(positions[ticker].avg_entry_price):.2f}" if ticker in positions else "No position"
            except Exception:
                pos_text = "Unknown"

            forecast_result = deep_learning_forecast(ticker, bars, sheet if 'sheet' in globals() else None)
            trend = forecast_result.get("trend", "N/A") if forecast_result else "N/A"
            forecast_conf = forecast_result.get("confidence", 0) if forecast_result else 0

            signal, reasons = interpret_signals(
                ticker, sma20, sma50, rsi, macd_latest, macd_signal_latest, adx, trend, forecast_conf
            )

            reasons_text = "\n".join(f"   • {r}" for r in reasons)

            message = (
                f"📊 *{ticker}*\n"
                f"• Price: ${float(last['close']):.2f} | SMA20 {sma20:.2f} / SMA50 {sma50:.2f}\n"
                f"• RSI: {rsi:.1f} | ATR: {atr:.2f} | MACD: {macd_latest:.3f}/{macd_signal_latest:.3f} | ADX: {'' if math.isnan(adx) else f'{adx:.1f}'}\n"
                f"• Regime: {regime.get('type', 'N/A')} ({regime.get('confidence', 0.0):.1f}%)\n"
                f"• AI: {ai_signal} ({ai_prob*100:.1f}%)\n\n"
                f"🔔 *Decision: {action}*  — Confidence {confidence*100:.0f}%\n"
                f"• Qty (suggested): {qty}\n"
                f"• Stop: {'' if stop_price is None else f'${stop_price:.2f}'} | TP: {'' if take_profit is None else f'${take_profit:.2f}'}\n"
                f"• Exec status: {exec_result.get('status')}\n"
                f"• Current Position: {pos_text}\n\n"
                f"🧠 *Rationale:*\n" + "\n".join(f"→ {r}" for r in reasons) + "\n\n"
                f"⏱️ {datetime.now(EST).strftime('%Y-%m-%d %H:%M:%S')} (EST)\n\n"
                f"🧠 *Reasoning:*\n{reasons_text}"
            )

            summary_messages.append(message)
            log_message(f"{ticker} => action={action} confidence={confidence:.2f} qty={qty} exec={exec_result.get('status')}")

        except Exception as e:
            log_message(f"❌ Error processing {ticker}: {e}")
            continue

    if summary_messages:
        header = f"📈 *Live Scan — {datetime.now(EST).strftime('%b %d %H:%M')}*\n\n"
        send_message(header + "\n\n".join(summary_messages))
        log_message("📩 Sent live scan to Telegram")
    else:
        log_message("ℹ️ No summary messages to send this run")




def interpret_signals(ticker, sma20, sma50, rsi, macd, macd_signal, adx, trend, forecast_conf):
    explanations = []
    signal = "HOLD"

    # SMA Crossover logic
    if sma20 > sma50:
        explanations.append("📈 SMA20 crossed above SMA50 → bullish trend forming")
        signal = "BUY"
    elif sma20 < sma50:
        explanations.append("📉 SMA20 below SMA50 → bearish trend likely")
        signal = "SELL"

    # RSI
    if rsi < 30:
        explanations.append("💎 RSI < 30 → oversold zone, potential rebound")
        signal = "BUY"
    elif rsi > 70:
        explanations.append("🔥 RSI > 70 → overbought zone, risk of pullback")
        signal = "SELL"

    # MACD
    if macd > macd_signal:
        explanations.append("⚡ MACD above signal → short-term momentum up")
        signal = "BUY"
    elif macd < macd_signal:
        explanations.append("💤 MACD below signal → weakening momentum")
        signal = "SELL"

    # ADX strength
    if adx >= 25:
        explanations.append("💪 ADX > 25 → strong trend detected")
    elif adx < 20:
        explanations.append("😴 ADX < 20 → weak / sideways trend")

    # AI forecast
    if trend == "Up" and forecast_conf >= 60:
        explanations.append(f"🤖 AI Forecast → {trend} trend with {forecast_conf}% confidence")
        signal = "BUY"
    elif trend == "Down" and forecast_conf >= 60:
        explanations.append(f"🤖 AI Forecast → {trend} trend with {forecast_conf}% confidence")
        signal = "SELL"
    else:
        explanations.append(f"🤖 AI Forecast → Neutral/Uncertain ({forecast_conf}%)")

    return signal, explanations

#============Back testing Logic===================

def backtest_strategy_with_report():
    """
    Backtests the ensemble + AI trading logic over historical data and sends
    a detailed performance report + chart to Telegram.
    """
    end_date = datetime.now(EST)
    start_date = end_date - timedelta(days=7)  # last 7 days default
    log_message(f"🧩 Starting backtest from {start_date} to {end_date} for {len(tickers)} tickers")

    results = []
    trades = []
    equity_curves = {}

    for ticker in tickers:
        log_message(f"📜 Backtesting {ticker} ...")
        try:
            bars = api.get_bars(
                ticker,
                tradeapi.TimeFrame.Minute,
                start=pd.Timestamp(start_date).isoformat(),
                end=pd.Timestamp(end_date).isoformat(),
                feed="iex"
            ).df

            if bars.empty or len(bars) < 100:
                log_message(f"⚠️ Not enough data for {ticker}")
                continue

            feat_df, full_df = create_features(bars)
            if full_df.empty:
                continue

            model = load_model_if_exists(ticker)

            equity = INITIAL_CAPITAL
            equity_curve = [equity]
            position = 0
            entry_price = 0
            qty = 0
            pnl_list = []

            for i in range(60, len(full_df)):
                current = full_df.iloc[:i].copy()
                last = current.iloc[-1]

                sma20 = last.get("SMA20", np.nan)
                sma50 = last.get("SMA50", np.nan)
                rsi = last.get("RSI", np.nan)
                atr = last.get("ATR", np.nan)
                macd = last.get("MACD", np.nan)
                macd_signal = last.get("MACD_SIGNAL", np.nan)
                adx = last.get("ADX", np.nan)
                regime = detect_market_regime(current)

                ai_signal = "Neutral"
                ai_prob = 0.5
                if model is not None and not feat_df.empty:
                    latest_feat = feat_df.iloc[[i - 1]].values
                    pred_class = model.predict(latest_feat)[0]
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(latest_feat)[0]
                        ai_prob = float(max(proba))
                    ai_signal = "Bullish" if pred_class == 1 else "Bearish"

                # ===== Ensemble Logic =====
                score = 0.0
                if not math.isnan(sma20) and not math.isnan(sma50):
                    score += 0.25 if sma20 > sma50 else -0.25
                if not math.isnan(macd) and not math.isnan(macd_signal):
                    score += 0.20 if macd > macd_signal else -0.20
                if not math.isnan(rsi):
                    if rsi < 40: score += 0.15
                    elif rsi > 60: score -= 0.15
                if not math.isnan(adx) and adx > 25:
                    score *= 1.05
                if regime and isinstance(regime, dict):
                    if regime["type"].lower().startswith("bull"):
                        score += 0.10
                    elif regime["type"].lower().startswith("bear"):
                        score -= 0.10
                if ai_signal == "Bullish":
                    score += ai_prob * 0.20
                elif ai_signal == "Bearish":
                    score -= ai_prob * 0.20

                # ===== Action =====
                if score > 0.25:
                    action = "BUY"
                elif score < -0.25:
                    action = "SELL"
                else:
                    action = "HOLD"

                close = last["close"]
                if not math.isnan(atr) and atr > 0:
                    stop = close - 1.5 * atr if action == "BUY" else close + 1.5 * atr
                    tp = close + 3.0 * atr if action == "BUY" else close - 3.0 * atr
                else:
                    stop, tp = None, None

                # ===== Simulate =====
                if position == 0 and action in ("BUY", "SELL"):
                    position = 1 if action == "BUY" else -1
                    entry_price = close
                    risk = 0.01
                    trade_size = equity * risk / (atr * 1.5 if atr > 0 else 1)
                    trade_size = min(trade_size, 1000)
                    qty = max(1, int(trade_size / close))
                    trades.append({
                        "ticker": ticker,
                        "entry_time": current.index[-1],
                        "entry_price": entry_price,
                        "action": action,
                        "qty": qty
                    })
                elif position != 0:
                    if stop and tp:
                        if (position == 1 and (close <= stop or close >= tp)) or \
                           (position == -1 and (close >= stop or close <= tp)):
                            exit_price = close
                            pnl = (exit_price - entry_price) * position * qty
                            equity += pnl
                            pnl_list.append(pnl)
                            trades[-1].update({
                                "exit_time": current.index[-1],
                                "exit_price": exit_price,
                                "pnl": pnl
                            })
                            position = 0
                equity_curve.append(equity)

            equity_curves[ticker] = equity_curve

            # ===== Metrics =====
            total_pnl = sum(pnl_list)
            win_rate = np.mean([1 if t.get("pnl", 0) > 0 else 0 for t in trades[-len(pnl_list):]]) if pnl_list else 0
            returns = pd.Series(equity_curve).pct_change().fillna(0)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252 * 6.5 * 60)
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
            log_message(f"❌ Backtest error for {ticker}: {e}")
            continue

    df = pd.DataFrame(results)
    log_message("📊 Backtest complete.")
    if df.empty:
        send_message("⚠️ No backtest results to report.")
        return df

    # ===== Plot Equity Curves =====
    plt.figure(figsize=(10, 5))
    for ticker, curve in equity_curves.items():
        plt.plot(curve, label=ticker)
    plt.title(f"Equity Curves ({start_date} → {end_date})")
    plt.xlabel("Trades")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # ===== Format Telegram Message =====
    summary = f"📈 *Backtest Report* ({start_date} → {end_date})\n\n"
    for _, row in df.iterrows():
        summary += (
            f"💠 *{row['Ticker']}*\n"
            f"• PnL: ${row['Total PnL']:,}\n"
            f"• Win Rate: {row['Win Rate']}%\n"
            f"• Sharpe: {row['Sharpe']}\n"
            f"• Max DD: {row['Max Drawdown %']}%\n"
            f"• Final Equity: ${row['Final Equity']:,}\n\n"
        )

    # ===== Send to Telegram =====
    send_message(summary)
    send_photo(buf)

    log_message("📩 Backtest report + chart sent to Telegram.")
    return df

def send_photo(image_buf):
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                  data={"chat_id": TELEGRAM_CHAT_ID},
                  files={"photo": ("chart.png", image_buf)})
# ================= MORNING SCAN =================
def morning_scan():
    log_message("🌅 Starting Morning Scan")
    potential = []
    ai_forecasts = []
    for ticker in tickers:
        try:
            end_dt = datetime.now(EST)
            start_dt = end_dt - timedelta(days=10)
            bars = api.get_bars(ticker, tradeapi.TimeFrame.Minute, start=start_dt.isoformat(), end=end_dt.isoformat(), feed="iex").df
           
            if bars.empty or len(bars) < 20:
                log_message(f"⚠️ Not enough data for {ticker}")
                continue
            bars = safe_tz_localize_and_convert(bars)
            # indicators
            bars["SMA20"] = bars["close"].rolling(20).mean()
            bars["SMA50"] = bars["close"].rolling(50).mean()
            bars["RSI"] = compute_rsi(bars["close"], 14)
            bars["ADX"] = ta.trend.adx(bars["high"], bars["low"], bars["close"], window=14)
            bars["ATR"] = compute_atr(bars[["high","low","close"]], ATR_PERIOD)
            bars["MACD"], bars["MACD_Signal"] = compute_macd(bars["close"])
            bars["AvgVol"] = bars["volume"].rolling(20).mean()

            # AI forecast
            if len(bars) >= 60:
                res = deep_learning_forecast(ticker, bars, sheet if 'sheet' in globals() else None)
                if res:
                    ai_forecasts.append(res)

            prev = bars.iloc[-2]
            curr = bars.iloc[-1]
            sma_gap = abs((curr["SMA20"] - curr["SMA50"]) / (curr["SMA50"] if curr["SMA50"] else 1)) * 100
            macd_gap = abs(curr["MACD"] - curr["MACD_Signal"])
            atr_pct = (curr["ATR"] / curr["close"]) * 100 if curr["close"] else 0
            vol_ratio = curr["volume"] / (curr["AvgVol"] if not pd.isna(curr["AvgVol"]) else 1)
            bullish_ready = prev["SMA20"] <= prev["SMA50"] and curr["SMA20"] > curr["SMA50"]
            bearish_ready = prev["SMA20"] >= prev["SMA50"] and curr["SMA20"] < curr["SMA50"]
            if bullish_ready or bearish_ready:
                direction = "BULLISH" if bullish_ready else "BEARISH"
                score = sum([
                    int(bullish_ready or bearish_ready),
                    int(curr["ADX"] > 20 if not pd.isna(curr["ADX"]) else 0),
                    int(vol_ratio > 1.0),
                    int(macd_gap < 1.0)
                ])
                potential.append({"Ticker": ticker, "Direction": direction, "Price": round(curr["close"], 2), "RSI": round(curr["RSI"],1) if not pd.isna(curr["RSI"]) else None, "ADX": round(curr["ADX"],1) if not pd.isna(curr["ADX"]) else None, "VolSpike": round(vol_ratio,2), "SMA_Gap%": round(sma_gap,2), "Score": score})
        except Exception as e:
            log_message(f"⚠️ Morning scan error for {ticker}: {e}")

    if ai_forecasts:
        msg = "🤖 *AI Forecasts Summary — Morning Scan*\n\n"
        for f in ai_forecasts:
            reason = "Predicted ↑" if f['trend']=="BULLISH" else "Predicted ↓"
            msg += f"{f['ticker']}: Current {f['current']:.2f} | Predicted {f['forecast']:.2f} | {f['trend']} | {reason} | Conf {f['confidence']}%\n"+"\n"
        send_message(msg)

    if not potential:
        send_message("🌄 Morning scan complete — no potential setups found.")
        return

    df = pd.DataFrame(potential).sort_values("Score", ascending=False)
    top10 = df.head(10)

    # Update Google sheet safely
    try:
        if 'sheet' in globals():
            try:
                ws = sheet.worksheet("Morning_Scanner")
            except Exception:
                ws = sheet.add_worksheet(title="Morning_Scanner", rows="1000", cols="10")
            ws.clear()
            ws.update([top10.columns.values.tolist()] + top10.values.tolist())
            log_message("📊 Morning scanner updated to Sheets")
    except Exception as e:
        log_message(f"⚠️ Write to sheet failed: {e}")

    message = "🌅 *Morning Scanner — Potential Stocks for Today*\n\n"
    for _, r in top10.iterrows():
        msg += f"{r['Ticker']}: {r['Direction']} | Price: {r['Price']} | RSI: {r['RSI']} | ADX: {r['ADX']} | Vol x{r['VolSpike']} | SMA Gap: {r['SMA_Gap%']}%\n"
    
    send_message(msg)
    log_message("✅ Morning scan completed")




if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python moving_average_bot.py [morning|analysis|live|backtest|auto_train]")
        sys.exit(1)
    mode = sys.argv[1]
    if mode == "morning":
        get_top_gap_gainers()
        morning_scan()
    elif mode == "analysis":
        previous_day_analysis()
    elif mode == "live":
        live_trading_loop()
    elif mode == "backtest":
       backtest_strategy_with_report()
    elif mode == "auto_train":
       live_trading_loop()
