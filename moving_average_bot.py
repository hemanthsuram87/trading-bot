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
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
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
print("KEY:", os.getenv("APCA_API_KEY_ID"))
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


BACKTEST_DAYS = 60  # Adjustable, conservative by default
MODEL_DIR = "models/sklearn/"
LOG_DIR = "logs"
SIGNAL_DIR = "signals"
MODEL_DIR = "models"

INITIAL_CAPITAL = 10000
COMMISSION = 0.001  # 0.1%


# ====== Best filter parameters for backtesting ======
BEST_FILTERS_BACKTEST = {
    'SMA_score': 0.1,
    'RSI_oversold': 40,
    'RSI_overbought': 60,
    'RSI_score': 0.05,
    'ATR_threshold': 0.0,       # allow all trades
    'STOP_ATR_MULT': 1.0,
    'TP_ATR_MULT': 1.0,
    'AI_score': 0.1,
    'Score_threshold': 0.0,     # allow all signals
    'Volume_filter': False,
    'Risk': 0.01
}

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SIGNAL_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# logging
EST = pytz.timezone("US/Eastern")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
def log_message(msg):
    ts = datetime.now(EST).strftime("[%Y-%m-%d %H:%M:%S]")
    logging.info(f"{ts} {msg}")
    print(f"{ts} {msg}")

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
        'PnL': float,
        'Qty': int,
        'Equity': float
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
    else:
        # --- Excel fallback ---
        file_path = "executed_trades.xlsx"
        df = pd.DataFrame([trade])
        if os.path.exists(file_path):
            df_existing = pd.read_excel(file_path)
            df = pd.concat([df_existing, df], ignore_index=True)
        df.to_excel(file_path, index=False)



def init_equity(sheet_client, tickers, default_equity=1000):
    """
    Initialize equity per ticker from Google Sheet or create missing tickers.
    Only adds tickers that are not already present.
    Returns a dict: {ticker: equity_value}
    """
    equity = {}
    if sheet_client:
        # Try to open the worksheet
        try:
            ws = sheet_client.worksheet("TickerEquity")
        except gspread.WorksheetNotFound:
            ws = sheet_client.add_worksheet(title="TickerEquity", rows="100", cols="5")
            ws.append_row(["Ticker", "Equity", "LastUpdate"])

        # Fetch existing tickers
        existing_records = ws.get_all_records()
        existing_tickers = [row["Ticker"] for row in existing_records]

        # Populate equity dict and add missing tickers
        for t in tickers:
            if t in existing_tickers:
                # Use existing equity
                equity[t] = next(row["Equity"] for row in existing_records if row["Ticker"] == t)
            else:
                # Add new ticker
                ws.append_row([t, default_equity, datetime.now().isoformat()])
                equity[t] = default_equity
    else:
        # Fallback: Excel
        file_path = "equity.xlsx"
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
        else:
            df = pd.DataFrame(columns=["Ticker","Equity","LastUpdate"])
        for t in tickers:
            if t in df['Ticker'].values:
                equity[t] = df.loc[df['Ticker']==t,'Equity'].values[0]
            else:
                df = pd.concat([df, pd.DataFrame([{"Ticker":t,"Equity":default_equity,"LastUpdate":datetime.now()}])], ignore_index=True)
                equity[t] = default_equity
        df.to_excel(file_path, index=False)

    return equity

def update_equity_sheet(ticker, equity_value):
    if sheet:
        try:
            ws = sheet.worksheet("TickerEquity")
            cell = ws.find(ticker)
            if cell:
                ws.update_cell(cell.row, 2, round(equity_value, 2))
                ws.update_cell(cell.row, 3, datetime.now().isoformat())
        except Exception as e:
            print(f"⚠️ Update equity failed for {ticker}: {e}")
    else:
        # Excel fallback
        file_path = "equity.xlsx"
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
        else:
            df = pd.DataFrame(columns=["Ticker","Equity","LastUpdate"])
        if ticker in df['Ticker'].values:
            df.loc[df['Ticker']==ticker,'Equity'] = equity_value
            df.loc[df['Ticker']==ticker,'LastUpdate'] = datetime.now()
        else:
            df = pd.concat([df, pd.DataFrame([{"Ticker":ticker,"Equity":equity_value,"LastUpdate":datetime.now()}])], ignore_index=True)
        df.to_excel(file_path, index=False)

# =======================
# Initialize global equity
# =======================
equity = init_equity(sheet, tickers, default_equity=1000)


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


# ===================== HELPERS =====================
def fetch_last_bar(ticker, limit=60, timeframe='1Min'):
    end_dt = datetime.now(EST)
    start_dt = end_dt - timedelta(minutes=limit)
    bars = api.get_bars(
        ticker,
        tradeapi.TimeFrame.Minute,
        start=start_dt.isoformat(),
        end=end_dt.isoformat(),
        limit=limit,
        adjustment='raw',
        feed='iex'
    ).df
    if not bars.empty:
        bars = prepare_indicators(bars)  # Precompute indicators
        return bars.iloc[-1], bars
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

def safe_text(text):
    if not text:
        return ""
    return html.escape(str(text), quote=False)  # escapes < > & but leaves quotes

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


def update_morning_scanner_with_ai(top_n=5):
    """
    Fetch top gainers, analyze technicals + AI forecast,
    send Telegram alerts, and update Morning_Scanner Google Sheet.
    """
    # Step 1: Fetch top gainers
    df = get_top_gap_gainers(top_n=top_n)
    if df is None or df.empty:
        print("⚠️ No top gainers found today.")
        return
    
    potential = []
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

            if bars.empty or len(bars) < 20:
                continue

            bars = safe_tz_localize_and_convert(bars)
            feat_df, df = create_features(bars)  # precompute SMA, RSI, ATR, etc.
             # ✅ check existence of indicators
            
            prev = feat_df.iloc[-2]
            curr = feat_df.iloc[-1]
            
            # Technical signals
            sma_gap = abs((curr["SMA20"] - curr["SMA50"]) / (curr["SMA50"] or 1)) * 100
            rsi = round(curr["RSI"], 1)
            adx = round(curr.get("ADX", 0), 1)
            vol_ratio = curr["VOL_RATIO"]
            bullish_ready = prev["SMA20"] <= prev["SMA50"] and curr["SMA20"] > curr["SMA50"]
            bearish_ready = prev["SMA20"] >= prev["SMA50"] and curr["SMA20"] < curr["SMA50"]
            direction = "BULLISH" if bullish_ready else "BEARISH" if bearish_ready else "NEUTRAL"
            score = sum([int(bullish_ready or bearish_ready), int(adx > 20), int(vol_ratio > 1.0)])

            # AI Forecast integration
            if len(bars) >= 60:  # require sufficient bars
                ai_signal, ai_conf = ai_predict(bars, ticker, lookback=60)
                if ai_signal:
                    ai_forecasts.append({"ticker": ticker, "trend": ai_signal, "confidence": ai_conf, "current": close})
                    direction += f" | AI:{ai_signal}"
                    score += ai_conf / 20  # scale confidence to score
            print(f"AI Score {score}   direction {direction}")
            
            potential.append([
                ticker, direction, round(close, 2),
                rsi, adx, round(vol_ratio, 2),
                round(sma_gap, 2), round(score, 2)
            ])

        except Exception as e:
            log_message(f"⚠️ Analysis failed for {ticker}: {e}")

    if not potential:
        send_message("🌄 Morning scan complete — no actionable tickers today.")
        return

    # Step 3: Update Google Sheet
    columns = ["Ticker", "Direction", "Price", "RSI", "ADX", "VolSpike", "SMA_Gap%", "Score"]
    df_potential = pd.DataFrame(potential, columns=columns)
    try:
        if 'sheet' in globals():
            try:
                ws = sheet.worksheet("Morning_Scanner")
            except Exception:
                ws = sheet.add_worksheet(title="Morning_Scanner", rows="1000", cols=str(len(columns)))
            ws.clear()
            ws.update([columns] + df_potential.values.tolist())
            log_message(f"📊 Morning_Scanner updated with {len(df_potential)} tickers including AI.")
    except Exception as e:
        log_message(f"⚠️ Failed to update Morning_Scanner: {e}")

    # Step 4: Send Telegram Alerts
    msg = "📊 Morning Scanner — Top Gainers with AI Forecast\n\n"
    for _, r in df_potential.iterrows():
        msg += (f"🔹 {r['Ticker']} | {r['Direction']} | Price: {r['Price']}\n"
                f"RSI: {r['RSI']} | ADX: {r['ADX']} | Vol x{r['VolSpike']} | SMA Gap: {r['SMA_Gap%']}% | Score: {r['Score']}\n\n")

    if ai_forecasts:
        msg += "\n🤖 AI Forecasts Summary:\n"
        for f in ai_forecasts:
            msg += (f"{f['ticker']}: Predicted {f['trend']} | "
                    f"Current {f['current']:.2f} | Confidence {f['confidence']*100:.1f}%\n")
    send_message(msg)
    log_message("✅ Telegram alert sent for Morning Scanner tickers with AI analysis.")

   
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
        "1Min":  TimeFrame(1, TimeFrameUnit.Minute),
        "5Min":  TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day":  TimeFrame(1, TimeFrameUnit.Day),
    }
    tf_obj = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Day))

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



###############################################################
################ AI MODEL FOR TRADING HELPERS #################
###############################################################

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

# ============================
# Config / Paths
# ============================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================
# Feature preparation
# ============================
def prepare_features(df: pd.DataFrame, lookback=10):
    """
    Prepares lagged features for AI model.
    Args:
        df: DataFrame with OHLCV and indicators (sma_short, sma_long, rsi, atr, volume)
        lookback: number of bars for rolling features
    Returns:
        X: feature matrix
        y: labels (1=Bullish, 0=Bearish)
    """
    df = df.copy()
    # Example label: 1 if next close > current close, else 0
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Rolling / lag features
    features = ['close', 'sma_short', 'sma_long', 'rsi', 'atr', 'volume']
    X = []
    y = []
    
    for i in range(lookback, len(df)-1):
        vals = df[features].iloc[i-lookback:i].values.flatten()
        if not np.any(pd.isna(vals)):
            X.append(vals)
            y.append(df['target'].iloc[i])
    
    return np.array(X), np.array(y)

# ============================
# Train or update AI model
# ============================
def train_ai_model(df: pd.DataFrame, ticker: str, lookback=10):
    """
    Train or update XGBoost model for a ticker.
    Saves model and scaler to disk.
    """
    X, y = prepare_features(df, lookback)
    if len(X) == 0:
        print(f"⚠️ Not enough data to train {ticker}")
        return None

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Save model and scaler
    joblib.dump(model, f"{MODEL_DIR}/{ticker}_xgb_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/{ticker}_scaler.pkl")
    print(f"✅ Trained AI model for {ticker}")

    return model, scaler

# ============================
# Load existing model
# ============================
def load_ai_model(ticker: str):
    try:
        model = joblib.load(f"{MODEL_DIR}/{ticker}_xgb_model.pkl")
        scaler = joblib.load(f"{MODEL_DIR}/{ticker}_scaler.pkl")
        return model, scaler
    except Exception:
        return None, None

# ============================
# AI prediction for core_signal
# ============================
def ai_predict(df: pd.DataFrame, ticker: str, lookback=10):
    model, scaler = load_ai_model(ticker)
    if model is None or scaler is None:
        return None, 0.5  # default neutral

    # Take last lookback bars
    features = ['close', 'sma_short', 'sma_long', 'rsi', 'atr', 'volume']
    if len(df) < lookback:
        return None, 0.5

    X = df[features].iloc[-lookback:].values.flatten().reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred_class = model.predict(X_scaled)[0]
    pred_prob = float(max(model.predict_proba(X_scaled)[0]))
    signal = "Bullish" if pred_class == 1 else "Bearish"
    return signal, pred_prob


###############################################################
#############  AI Model traing end ############################
###############################################################


# =========================
# CORE SIGNAL FOR BACKTEST
# =========================
# =========================
# UNIVERSAL CORE SIGNAL
# =========================

# ======================
# Simplified core signal
# ======================
import numpy as np
import pandas as pd
from datetime import timedelta

def core_signal(
    df: pd.DataFrame,
    filters: dict = None,
    model=None,
    last_trade_time=None,
    current_time=None,
    log_message=print
) -> dict:
    """
    🚀 Production-grade core signal generator.
    Works for both backtesting and live trading.
    Always returns a safe, well-structured dictionary.

    Features:
    - SMA, RSI, ATR, and optional AI-based scoring (calculated dynamically if not present).
    - Cooldown logic to prevent duplicate signals.
    - Adaptive threshold-based scoring system.
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

    # === Calculate SMA, ATR, RSI dynamically if missing ===
    if 'SMA_short' not in df.columns:
        df['SMA_short'] = df['close'].rolling(filters['SMA_short_period']).mean()
    if 'SMA_long' not in df.columns:
        df['SMA_long'] = df['close'].rolling(filters['SMA_long_period']).mean()
    if 'ATR' not in df.columns:
        df['H-L'] = df['high'] - df['low']
        df['H-C'] = (df['high'] - df['close'].shift()).abs()
        df['L-C'] = (df['low'] - df['close'].shift()).abs()
        df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
        df['ATR'] = df['TR'].rolling(filters['ATR_period']).mean()
    if 'RSI' not in df.columns:
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

    if len(df) >= 2:
        prev = df.iloc[-2]
        sma_short_prev = prev['SMA_short']
        sma_long_prev = prev['SMA_long']

        # Only count new crossovers
        if sma_short > sma_long and sma_short_prev <= sma_long_prev:
            score += filters['SMA_score']
        elif sma_short < sma_long and sma_short_prev >= sma_long_prev:
            score -= filters['SMA_score']

    if not pd.isna(rsi):
        if rsi < filters['RSI_oversold']:
            score += filters['RSI_score']
        elif rsi > filters['RSI_overbought']:
            score -= filters['RSI_score']

    # Optional AI Model Contribution
    if model is not None:
        try:
            features = df.select_dtypes(include=[np.number]).tail(1)
            if not features.empty:
                pred = model.predict(features)[0]
                proba = (
                    model.predict_proba(features)[0]
                    if hasattr(model, "predict_proba")
                    else [0.5, 0.5]
                )
                ai_prob = float(max(proba))
                score += filters['AI_score'] * ai_prob * (1 if pred == 1 else -1)
                log_message(f"🤖 AI signal: {pred}, prob={ai_prob:.2f}")
        except Exception as e:
            log_message(f"⚠️ AI model error: {e}")

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


def core_signal_live(
    last_bar: pd.Series,
    sma_short: float,
    sma_long: float,
    atr: float,
    rsi: float,
    model=None,
    lookback=60,
    historical_bars: pd.DataFrame = None,
    filters: dict = None,
    last_trade_time=None,
    current_time=None,
    log_message=print
) -> dict:
    """
    Core signal generator for live trading.
    Score is based only on SMA/RSI/ATR filters.
    AI model provides additional forecast but does NOT affect score.
    """

    # --- Default filters ---
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
        'PositionSize': 0.0,
        'AI_signal': None,
        'AI_prob': None
    }

    price = last_bar.get('close', np.nan)
    if pd.isna(price) or pd.isna(atr):
        return out

    # --- ATR filter ---
    if atr / price > filters['ATR_threshold']:
        log_message(f"⚠️ High volatility skipped (ATR/Price={atr/price:.2%})")
        return out

    # --- Cooldown check ---
    if last_trade_time and current_time:
        cooldown = timedelta(minutes=filters['Cooldown_minutes'])
        if (current_time - last_trade_time) < cooldown:
            return out

    # --- Score Calculation (SMA + RSI) ---
    score = 0.0

    # SMA crossover
    if sma_short > sma_long:
        score += filters['SMA_score']
    elif sma_short < sma_long:
        score -= filters['SMA_score']

    # RSI filter
    if rsi < filters['RSI_oversold']:
        score += filters['RSI_score']
    elif rsi > filters['RSI_overbought']:
        score -= filters['RSI_score']

    # --- AI prediction (reference only) ---
    if model is not None and historical_bars is not None and len(historical_bars) >= lookback:
        try:
            features = historical_bars[['close', 'SMA_short', 'SMA_long', 'RSI', 'ATR', 'volume']].iloc[-lookback:]
            features = features.values.flatten().reshape(1, -1)
            if hasattr(model, 'scaler'):
                features = model.scaler.transform(features)
            pred_class = model.predict(features)[0]
            proba = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
            out['AI_signal'] = 'LONG' if pred_class == 1 else 'SHORT'
            out['AI_prob'] = float(max(proba))
            log_message(f"🤖 AI forecast: {out['AI_signal']}, probability={out['AI_prob']:.2f}")
        except Exception as e:
            log_message(f"⚠️ AI model error: {e}")

    # --- Final signal based on score ---
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
import itertools

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


from itertools import product

########################################################################################
############   LIVE TRADING LOGIC AND HELPERS START ####################################
########################################################################################
POSITIONS = {t: None for t in tickers}  # Track open positions for live trading 
LAST_TRADE_TIME = {t: None for t in tickers}  # Cooldown tracking
# Risk and position tracking
RISK_PER_TRADE = 0.01
positions = {t: None for t in tickers}          # Track open positions
last_trade_time = {t: None for t in tickers}    # Cooldown tracking
equity = {t: 10000 for t in tickers}            # Starting equity per ticker


# =======================
# Update execute_trade to persist equity
# =======================
def execute_trade(ticker, signal):
    global positions, equity, last_trade_time
    pos = positions[ticker]
    qty = max(1, int(RISK_PER_TRADE * equity[ticker] / signal['Risk'])) if signal['Risk'] else 1
    now = datetime.now()

    if pos is None and signal['signal'] in ("LONG","SHORT"):
        positions[ticker] = {"entry_price":signal['entry_price'],
                             "direction":1 if signal['signal']=='LONG' else -1,
                             "qty":qty,
                             "stop":signal['stop'],
                             "tp":signal['tp']}
        last_trade_time[ticker] = now
        msg = f"🟢 Opened {signal['signal']} {ticker} at {signal['entry_price']:.2f} | Qty: {qty} AI_Signal = {signal['AI_signal']} and Prob = {signal['AI_Prob']}"
        print(msg)
        send_message(msg)
        log_trade({"Ticker":ticker,"Date":now,"Signal":signal['signal'],
                   "Entry":signal['entry_price'],"Exit":"","PnL":0,"Qty":qty,"Equity":equity[ticker]})
        update_equity_sheet(ticker, equity[ticker])

    elif pos is not None:
        cur_price = signal['entry_price']
        exit_cond = ((pos['direction']==1 and (cur_price<=pos['stop'] or cur_price>=pos['tp'])) or
                     (pos['direction']==-1 and (cur_price>=pos['stop'] or cur_price<=pos['tp'])))
        if exit_cond:
            pnl = (cur_price - pos['entry_price'])*pos['direction']*pos['qty']
            equity[ticker] += pnl
            msg = f"🔴 Closed {ticker} | PnL: {pnl:.2f} | New Equity: {equity[ticker]:.2f} AI_Signal = {signal['AI_signal']} and Prob = {signal['AI_Prob']}"
            print(msg)
            send_message(msg)
            positions[ticker] = None
            log_trade({"Ticker":ticker,"Date":now,"Signal":"CLOSE",
                       "Entry":pos['entry_price'],"Exit":cur_price,"PnL":pnl,
                       "Qty":pos['qty'],"Equity":equity[ticker]})
            update_equity_sheet(ticker, equity[ticker])


# ================================
# Live trading loop
# ================================
def live_trading_loop():
    filters = {
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
    print("KEY:", os.getenv("APCA_API_KEY_ID"))
    for ticker in tickers:
        last_bar, bars = fetch_last_bar(ticker)
        if last_bar is None:
            continue

        sma_short = last_bar['SMA_short']
        sma_long = last_bar['SMA_long']
        atr = last_bar['ATR']
        rsi = last_bar['RSI']

        if np.isnan([sma_short, sma_long, atr, rsi]).any():
            continue

        signal = core_signal_live(
            last_bar=last_bar,
            sma_short=sma_short,
            sma_long=sma_long,
            atr=atr,
            rsi=rsi,
            filters=filters,
            last_trade_time=last_trade_time[ticker],
            current_time=datetime.now(),
            log_message=logging.info
        )

        if signal['signal'] in ("LONG", "SHORT"):
            last_trade_time[ticker] = datetime.now()

        execute_trade(ticker, signal)


##### This is for Real Alpaca trading ################                
def execute_alpaca_trade(ticker, signal):
    """Place real Alpaca order with stop-loss and take-profit"""
   # global POSITIONS, LAST_TRADE_TIME, RISK_PER_TRADE,
    global API
    position = POSITIONS[ticker]
    account = API.get_account()
    equity = float(account.cash)
    qty = max(1, int(RISK_PER_TRADE * equity / signal['Risk'])) if signal['Risk'] else 1

    # Open position if flat
    if position is None and signal['signal'] in ("LONG", "SHORT"):
        side = "buy" if signal['signal'] == 'LONG' else "sell"
        try:
            order = API.submit_order(
                symbol=ticker,
                qty=qty,
                side=side,
                type='market',
                time_in_force='gtc'
            )
            POSITIONS[ticker] = {
                "entry_price": signal['entry_price'],
                "direction": 1 if signal['signal'] == 'LONG' else -1,
                "qty": qty,
                "stop": signal['stop'],
                "tp": signal['tp'],
                "order_id": order.id
            }
            LAST_TRADE_TIME[ticker] = datetime.now()
            print(f"🟢 {side.upper()} order submitted for {ticker} | Qty: {qty}")
        except Exception as e:
            print(f"❌ Order failed for {ticker}: {e}")

    # Manage exit if position exists
    elif position is not None:
        current_price = signal['entry_price']
        if (position['direction'] == 1 and (current_price <= position['stop'] or current_price >= position['tp'])) or \
           (position['direction'] == -1 and (current_price >= position['stop'] or current_price <= position['tp'])):
            side = "sell" if position['direction'] == 1 else "buy"
            try:
                API.submit_order(
                    symbol=ticker,
                    qty=position['qty'],
                    side=side,
                    type='market',
                    time_in_force='gtc'
                )
                print(f"🔴 Closed {ticker} at {current_price:.2f}")
                POSITIONS[ticker] = None
            except Exception as e:
                print(f"❌ Close order failed for {ticker}: {e}")



########################################################################################
############   LIVE TRADING LOGIC AND HELPERS END ####################################
########################################################################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python moving_average_bot.py [morning|analysis|live|backtest|auto_train]")
        sys.exit(1)
    mode = sys.argv[1]

    if mode == "morning":
        update_morning_scanner_with_ai()
    elif mode == "live":      
        try:
            live_trading_loop()
        except Exception as e:
            logging.error(f"Error in live trading loop: {e}")
            send_message(f"❌ Error in live trading loop: {e}")
    elif mode == "backtest":
       #auto_train_models(retrain=False)
       best_filters = optimize_filters()
       print("Use this filter set for live trading:", best_filters)
