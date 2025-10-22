import os
import pandas as pd
import numpy as np
import requests
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ================= CONFIG =================
ALPACA_KEY = os.getenv("ALPACA_KEY", "YOUR_ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "YOUR_ALPACA_SECRET")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GOOGLE_CREDS_FILE = "service_account.json"
LOG_DIR = "logs"

SMA_SHORT = 20
SMA_LONG = 50
ATR_PERIOD = 14
RSI_PERIOD = 14
ATR_MULTIPLIER = 1
BB_PERIOD = 20
BB_STD = 2
DRY_RUN = True  # True = backtest, False = live trading

# ================= HELPERS =================
def send_message(msg):
    print(msg)
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                          data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
        except Exception as e:
            print(f"Telegram error: {e}")

def log_message(msg):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"trade_signals_{datetime.now().strftime('%Y-%m-%d')}.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(f"[{timestamp}] {msg}")

def compute_atr(df, period=14):
    df['H-L'] = df['high'] - df['low']
    df['H-Cp'] = abs(df['high'] - df['close'].shift(1))
    df['L-Cp'] = abs(df['low'] - df['close'].shift(1))
    tr = df[['H-L', 'H-Cp', 'L-Cp']].max(axis=1)
    return tr.rolling(period).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def compute_bollinger_bands(series, period=20, std=2):
    sma = series.rolling(period).mean()
    upper = sma + std * series.rolling(period).std()
    lower = sma - std * series.rolling(period).std()
    return upper, lower

# ================= GOOGLE SHEETS =================
def get_tickers_from_sheet(sheet_name="Daily_stocks", worksheet_name="Tickers"):
    scope = ["https://spreadsheets.google.com/feeds",
             "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDS_FILE, scope)
    client = gspread.authorize(creds)
    sheet = client.open(sheet_name).worksheet(worksheet_name)
    return sheet.col_values(1)

# ================= INIT =================
api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL, api_version='v2')
today = datetime.now().date()
yesterday = today - timedelta(days=1)
mode = "LIVE" if not DRY_RUN else "BACKTEST"
log_message(f"📊 Starting {mode} analysis for {yesterday}")

try:
    tickers = get_tickers_from_sheet()
except Exception as e:
    log_message(f"⚠️ Error fetching tickers from Google Sheet: {e}")
    tickers = ["AAPL", "TSLA"]

# ================= MAIN LOOP =================
summary_results = []
for ticker in tickers:
    try:
        start_dt = f"{yesterday}T09:30:00-04:00"
        end_dt = f"{yesterday}T16:00:00-04:00"
        bars = api.get_bars(ticker, tradeapi.TimeFrame.Minute, start=start_dt, end=end_dt, feed="iex").df
        if bars.empty or len(bars) < SMA_LONG:
            log_message(f"⚠️ Not enough data for {ticker}")
            continue

        # Indicators
        bars["SMA20"] = bars["close"].rolling(SMA_SHORT).mean()
        bars["SMA50"] = bars["close"].rolling(SMA_LONG).mean()
        bars["ATR"] = compute_atr(bars, ATR_PERIOD)
        bars["RSI"] = compute_rsi(bars["close"], RSI_PERIOD)
        bars["MACD"], bars["MACD_signal"] = compute_macd(bars["close"])
        bars["BB_upper"], bars["BB_lower"] = compute_bollinger_bands(bars["close"], BB_PERIOD, BB_STD)
        bars = bars.reset_index()
        bars["Signal"] = ""
        bars["Reason"] = ""
        bars["TradeTime"] = bars["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

        for i in range(1, len(bars)):
            prev = bars.iloc[i-1]
            curr = bars.iloc[i]
            atr_value = curr["ATR"]
            stop_distance = atr_value * ATR_MULTIPLIER
            reason_parts = []

            # SMA crossover
            if prev["SMA20"] < prev["SMA50"] and curr["SMA20"] > curr["SMA50"]:
                signal = "BUY 🔼"
                reason_parts.append("SMA20 crossed above SMA50 → bullish trend")
            elif prev["SMA20"] > prev["SMA50"] and curr["SMA20"] < curr["SMA50"]:
                signal = "SELL 🔽"
                reason_parts.append("SMA20 crossed below SMA50 → bearish trend")
            else:
                signal = ""

            # RSI filter
            if signal == "BUY 🔼" and curr["RSI"] < 70:
                reason_parts.append(f"RSI={curr['RSI']:.2f} → bullish momentum")
            elif signal == "SELL 🔽" and curr["RSI"] > 30:
                reason_parts.append(f"RSI={curr['RSI']:.2f} → bearish momentum")

            # MACD filter
            if signal:
                if (signal=="BUY 🔼" and curr["MACD"] > curr["MACD_signal"]) or \
                   (signal=="SELL 🔽" and curr["MACD"] < curr["MACD_signal"]):
                    reason_parts.append(f"MACD confirms trend ({curr['MACD']:.2f} vs {curr['MACD_signal']:.2f})")

            # Bollinger Band confirmation
            if signal == "BUY 🔼" and curr["close"] < curr["BB_lower"]:
                reason_parts.append(f"Price touched lower Bollinger Band ({curr['BB_lower']:.2f})")
            elif signal == "SELL 🔽" and curr["close"] > curr["BB_upper"]:
                reason_parts.append(f"Price touched upper Bollinger Band ({curr['BB_upper']:.2f})")

            # ATR
            reason_parts.append(f"ATR={atr_value:.2f}, Stop-risk distance={stop_distance:.2f}")

            if signal:
                bars.at[i, "Signal"] = signal
                bars.at[i, "Reason"] = " | ".join(reason_parts)
                msg = f"{ticker} {signal} at {curr['close']:.2f} on {curr['TradeTime']} ({bars.at[i,'Reason']})"
                log_message(msg)
                send_message(msg)

                # Live trade execution
                if not DRY_RUN:
                    qty = max(int(1000 / stop_distance), 1)
                    api.submit_order(symbol=ticker, qty=qty, side="buy" if signal=="BUY 🔼" else "sell",
                                     type="market", time_in_force="day")

        log_message(f"✅ Completed analysis for {ticker}")

    except Exception as e:
        log_message(f"⚠️ Error analyzing {ticker}: {e}")
        continue

log_message("📊 Daily analysis completed.")


