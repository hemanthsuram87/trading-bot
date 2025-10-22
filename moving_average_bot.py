import os
import time
import pandas as pd
import numpy as np
import requests
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ================= CONFIG =================
ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GOOGLE_CREDS_FILE = "service_account.json"

SMA_SHORT = 20
SMA_LONG = 50
ATR_PERIOD = 14
ATR_MULTIPLIER = 1
RISK_PER_TRADE = 0.01
LOG_DIR = "logs"
CHECK_INTERVAL = 60  # seconds

LIVE_RUN = True  # True = monitor live market, False = previous-day analysis
def log_message(msg):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"trade_signals_{datetime.now().strftime('%Y-%m-%d')}.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(f"[{timestamp}] {msg}")
# ================= INIT =================
api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL, api_version='v2')

# Check if market is open
clock = api.get_clock()
LIVE_RUN = clock.is_open  # True if market is open, False otherwise

if LIVE_RUN:
    log_message("🟢 Market is open — starting live trading monitoring.")
else:
    log_message("🔴 Market is closed — running previous-day analysis.")

# Google Sheets setup
scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDS_FILE, scope)
client = gspread.authorize(creds)
sheet = client.open("Daily_stocks")
tickers_ws = sheet.worksheet("Tickers")
tickers = tickers_ws.col_values(1)

# ================= HELPERS =================
def send_message(msg):
    print(msg)
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                          data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
        except Exception as e:
            print(f"Telegram error: {e}")

def compute_atr(df, period=14):
    df['H-L'] = df['high'] - df['low']
    df['H-Cp'] = abs(df['high'] - df['close'].shift(1))
    df['L-Cp'] = abs(df['low'] - df['close'].shift(1))
    tr = df[['H-L', 'H-Cp', 'L-Cp']].max(axis=1)
    return tr.rolling(period).mean()

def compute_shares_by_atr(cash, atr_value, price):
    stop_distance = atr_value * ATR_MULTIPLIER
    if stop_distance <= 0 or np.isnan(stop_distance):
        return 0
    dollar_risk = cash * RISK_PER_TRADE
    return max(int(dollar_risk / stop_distance), 0)

def market_open():
    clock = api.get_clock()
    return clock.is_open

# ================= MAIN LOOP =================
# Track last signal per ticker to avoid duplicates
last_signal_sent = {}

def analyze_ticker(ticker, bars):
    bars["SMA20"] = bars["close"].rolling(SMA_SHORT).mean()
    bars["SMA50"] = bars["close"].rolling(SMA_LONG).mean()
    bars["ATR"] = compute_atr(pd.DataFrame({
        "high": bars["high"],
        "low": bars["low"],
        "close": bars["close"]
    }), ATR_PERIOD)

    # Only look at the last two candles
    if len(bars) < SMA_LONG:
        return []

    prev = bars.iloc[-2]
    curr = bars.iloc[-1]

    atr_value = curr["ATR"]
    stop_distance = atr_value * ATR_MULTIPLIER
    trade_time = curr.name if isinstance(curr.name, datetime) else datetime.now()
    trade_time = trade_time.strftime("%Y-%m-%d %H:%M:%S")

    ticker_signal = None
    reason = ""

    # BUY signal
    if prev["SMA20"] < prev["SMA50"] and curr["SMA20"] > curr["SMA50"]:
        ticker_signal = "BUY"
        reason = f"SMA20 crossed above SMA50 → bullish trend | ATR={atr_value:.2f}, Stop-risk distance={stop_distance:.2f}"

    # SELL signal
    elif prev["SMA20"] > prev["SMA50"] and curr["SMA20"] < curr["SMA50"]:
        ticker_signal = "SELL"
        reason = f"SMA20 crossed below SMA50 → bearish trend | ATR={atr_value:.2f}, Stop-risk distance={stop_distance:.2f}"

    # If no signal or duplicate → skip
    if not ticker_signal:
        return []

    if last_signal_sent.get(ticker) == ticker_signal:
        return []  # same direction already alerted

    # Record and send
    last_signal_sent[ticker] = ticker_signal
    msg = f"{ticker} {ticker_signal} at {curr['close']:.2f} on {trade_time} ({reason})"
    log_message(msg)
    send_message(msg)

    return [{"ticker": ticker, "signal": ticker_signal, "price": curr["close"], "reason": reason, "time": trade_time}]


# Previous day analysis
def previous_day_analysis():
    yesterday = (datetime.now() - timedelta(days=1)).date()
    log_message(f"📊 Starting previous-day analysis for {yesterday}")
    print(f"Tickers picked up for today {tickers}")
    for ticker in tickers:
        try:
            start_date = f"{yesterday}T09:30:00-04:00"
            end_date = f"{yesterday}T16:00:00-04:00"
            bars = api.get_bars(ticker, tradeapi.TimeFrame.Minute, start=start_date, end=end_date, feed="iex").df
            if bars.empty:
                log_message(f"No bars for {ticker}")
                continue
            analyze_ticker(ticker, bars)
        except Exception as e:
            log_message(f"⚠️ Error analyzing {ticker}: {e}")

# Live trading check
def live_trading_loop():
    log_message("🚀 Starting live trading monitoring...")
   
    for ticker in tickers:
        try:
            bars = api.get_bars(
                ticker, tradeapi.TimeFrame.Minute,
                limit=SMA_LONG + 5, feed="iex"
            ).df
            if bars.empty:
                continue
            analyze_ticker(ticker, bars)
        except Exception as e:
            log_message(f"⚠️ Error analyzing {ticker}: {e}")
    time.sleep(CHECK_INTERVAL)
   log_message("⏰ Market closed. Live monitoring ended.")


# ================= EXECUTION =================
if __name__ == "__main__":
    if LIVE_RUN:
        live_trading_loop()
    else:
        previous_day_analysis()
