import os
import time
import pytz
import numpy as np
import pandas as pd
import requests
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ================= CONFIG =================
ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, base_url=ALPACA_BASE_URL)

EST = pytz.timezone("US/Eastern")

tickers = ["AAPL", "NVDA", "TSLA", "AMZN", "MSFT"]

# Telegram config
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_message(msg):
    if TELEGRAM_TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})

def log_message(msg):
    timestamp = datetime.now(EST).strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp} {msg}")

# ================= GOOGLE SHEETS SETUP =================
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)
sheet = client.open("TradingBot_Data")

# ================= INDICATOR ANALYSIS =================
def analyze_ticker(ticker, bars):
    if bars.empty:
        return

    df = bars.copy()
    df["SMA20"] = SMAIndicator(df["close"], window=20).sma_indicator()
    df["SMA50"] = SMAIndicator(df["close"], window=50).sma_indicator()
    df["RSI"] = RSIIndicator(df["close"], window=14).rsi()
    macd = MACD(df["close"])
    df["MACD"] = macd.macd()
    df["Signal"] = macd.macd_signal()

    latest = df.iloc[-1]
    signal = None
    reason = ""

    if latest["SMA20"] > latest["SMA50"] and latest["RSI"] < 70 and latest["MACD"] > latest["Signal"]:
        signal = "BUY"
        reason = "SMA20 crossed above SMA50 with RSI < 70 and MACD positive."
    elif latest["SMA20"] < latest["SMA50"] and latest["RSI"] > 30 and latest["MACD"] < latest["Signal"]:
        signal = "SELL"
        reason = "SMA20 crossed below SMA50 with RSI > 30 and MACD negative."

    if signal:
        msg = f"📊 {ticker}: {signal} | {reason}"
        log_message(msg)
        send_message(msg)
        save_to_sheet(ticker, signal, reason, latest)

# ================= SAVE TO SHEET =================
def save_to_sheet(ticker, signal, reason, latest):
    try:
        ws = sheet.worksheet("Signals")
    except Exception:
        ws = sheet.add_worksheet(title="Signals", rows="1000", cols="10")

    ws.append_row([
        datetime.now(EST).strftime("%Y-%m-%d %H:%M:%S"),
        ticker,
        signal,
        reason,
        latest["close"],
        latest["SMA20"],
        latest["SMA50"],
        latest["RSI"],
        latest["MACD"]
    ])

# ================= DEEP LEARNING FORECAST =================
def deep_learning_forecast(ticker, bars, lookback=60, forecast_steps=10):
    try:
        df = bars[["close"]].dropna()
        if len(df) < lookback:
            return None

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)

        X, y = [], []
        for i in range(lookback, len(scaled) - forecast_steps):
            X.append(scaled[i - lookback:i])
            y.append(scaled[i + forecast_steps][0])
        X, y = np.array(X), np.array(y)

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
            LSTM(32),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=15, batch_size=32, verbose=0,
                  callbacks=[EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)])

        last_seq = scaled[-lookback:].reshape((1, lookback, 1))
        forecast_scaled = model.predict(last_seq)[0][0]
        forecast_price = scaler.inverse_transform([[forecast_scaled]])[0][0]

        current = df["close"].iloc[-1]
        trend = "BULLISH" if forecast_price > current else "BEARISH"
        confidence = round(abs(forecast_price - current) / current * 100, 2)

        log_message(f"🤖 {ticker} LSTM Forecast: {trend} | Predicted: {forecast_price:.2f} | Conf: {confidence}%")

        ws = None
        try:
            ws = sheet.worksheet("AI_Forecast")
        except Exception:
            ws = sheet.add_worksheet(title="AI_Forecast", rows="1000", cols="10")

        ws.append_row([
            datetime.now(EST).strftime("%Y-%m-%d %H:%M:%S"),
            ticker, current, round(forecast_price, 2), trend, confidence
        ])
        return {"trend": trend, "confidence": confidence}

    except Exception as e:
        log_message(f"⚠️ AI forecast failed for {ticker}: {e}")
        return None

# ================= PREVIOUS DAY ANALYSIS =================
def previous_day_analysis():
    now = datetime.now(EST)
    today = now.date()
    start_date = (today - timedelta(days=3)).strftime("%Y-%m-%dT09:30:00-04:00")
    end_date = now.strftime("%Y-%m-%dT%H:%M:%S-04:00")

    log_message(f"📊 Running analysis for latest available trading day...")

    for ticker in tickers:
        try:
            bars = api.get_bars(ticker, tradeapi.TimeFrame.Minute,
                                start=start_date, end=end_date, adjustment="raw", feed="iex").df
            if not bars.empty:
                analyze_ticker(ticker, bars)
                deep_learning_forecast(ticker, bars)
            else:
                log_message(f"No bars found for {ticker}")
        except Exception as e:
            log_message(f"Error fetching bars for {ticker}: {e}")

    log_message("📌 Full-day analysis completed.")

# ================= LIVE TRADING LOOP =================
def live_trading_loop():
    log_message("🚀 Starting live trading loop (runs every 10 mins)")
    while True:
        now = datetime.now(EST)
        if now.weekday() >= 5:  # Skip weekends
            log_message("🕒 Weekend — sleeping 6 hours.")
            time.sleep(6 * 3600)
            continue

        if now.hour < 9 or (now.hour == 9 and now.minute < 30) or now.hour >= 16:
            log_message("⏸️ Market closed — waiting...")
            time.sleep(600)
            continue

        log_message(f"📈 Running live analysis at {now.strftime('%H:%M')}")
        for ticker in tickers:
            try:
                end = datetime.now(EST)
                start = end - timedelta(hours=6)
                bars = api.get_bars(ticker, tradeapi.TimeFrame.Minute,
                                    start=start.isoformat(), end=end.isoformat(), feed="iex").df
                if not bars.empty:
                    analyze_ticker(ticker, bars)
                    deep_learning_forecast(ticker, bars)
            except Exception as e:
                log_message(f"Live loop error for {ticker}: {e}")

        log_message("⏳ Sleeping 10 minutes...")
        time.sleep(600)

# ===================Morning scan ================
def morning_scan():
    log_message("🌅 Running Morning Scan...")
    for ticker in tickers:
        try:
            bars = api.get_bars(ticker, tradeapi.TimeFrame.Hour, limit=100, feed="iex").df
            if not bars.empty:
                deep_learning_forecast(ticker, bars)
                transformer_forecast(ticker, bars)
        except Exception as e:
            log_message(f"⚠️ Error scanning {ticker}: {e}")
    log_message("✅ Morning scan complete.")

# ================= ENTRY POINT =================
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "live"

    if mode == "analysis":
        previous_day_analysis()
    elif mode == "live":
        live_trading_loop()
    elif mode == "morning":
        morning_scan()
    else:
        log_message(f"Unknown mode: {mode}")
