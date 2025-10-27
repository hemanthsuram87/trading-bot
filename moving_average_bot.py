import os
import time
import pytz
import numpy as np
import pandas as pd
import requests
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, ADXIndicator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import ta
from ta.volatility import AverageTrueRange
import yfinance as yf

# ================= CONFIG =================
ALPACA_KEY = "PK237O7I63OSG4RKO32AQMCQVG"
ALPACA_SECRET = "EFez3L8WnR3DouNoXRYk2LEsCGpXejCTwd9ebUopxt1e"
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
# Telegram config
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
api_key = os.getenv("FMP_API_KEY")
api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, base_url=ALPACA_BASE_URL)

EST = pytz.timezone("US/Eastern")

SMA_SHORT = 20
SMA_LONG = 50
ATR_PERIOD = 14
ATR_MULTIPLIER = 1
CHECK_INTERVAL = 60  # seconds
LOG_DIR = "logs"
SIGNAL_DIR = "signals"

# ================= INIT =================
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SIGNAL_DIR, exist_ok=True)

def get_signal_file():
    today_str = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(SIGNAL_DIR, f"sent_signals_{today_str}.txt")

def load_sent_signals():
    file_path = get_signal_file()
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def save_sent_signal(signal_id):
    file_path = get_signal_file()
    with open(file_path, "a") as f:
        f.write(signal_id + "\n")

sent_signals = load_sent_signals()

def send_message(msg):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})

def log_message(msg):
    timestamp = datetime.now(EST).strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp} {msg}")

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / (loss + 1e-6)
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    df = df.copy()
    df["H-L"] = df["high"] - df["low"]
    df["H-Cp"] = abs(df["high"] - df["close"].shift(1))
    df["L-Cp"] = abs(df["low"] - df["close"].shift(1))
    tr = df[["H-L", "H-Cp", "L-Cp"]].max(axis=1)
    return tr.rolling(period).mean()

def compute_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# ================= GOOGLE SHEETS SETUP =================
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("google_creds.json", scope)
client = gspread.authorize(creds)
sheet = client.open("Daily_stocks")
tickers_ws = sheet.worksheet("Tickers")
tickers = tickers_ws.col_values(1)

# ================= INDICATOR ANALYSIS =================
def analyze_ticker_backup(ticker, bars):
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC").tz_convert("America/New_York")
    else:
        bars.index = bars.index.tz_convert("America/New_York")

    bars["SMA20"] = bars["close"].rolling(SMA_SHORT).mean()
    bars["SMA50"] = bars["close"].rolling(SMA_LONG).mean()
    bars["RSI"] = compute_rsi(bars["close"], 14)
    bars["ATR"] = compute_atr(bars[["high", "low", "close"]], ATR_PERIOD)
    adx_indicator = ADXIndicator(high=bars["high"], low=bars["low"], close=bars["close"], window=14)
    bars["ADX"] = adx_indicator.adx()
    bars["MACD"], bars["MACD_Signal"] = compute_macd(bars["close"])
    bars["AvgVol"] = bars["volume"].rolling(20).mean()

    bars = bars.reset_index()
    bars["TradeTime"] = bars["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Higher timeframe (hourly) confirmation
    try:
        higher_bars = api.get_bars(ticker, tradeapi.TimeFrame.Hour, limit=100, feed="iex").df
        higher_bars["SMA20"] = higher_bars["close"].rolling(20).mean()
        higher_bars["SMA50"] = higher_bars["close"].rolling(50).mean()
        higher_trend = "bullish" if higher_bars["SMA20"].iloc[-1] > higher_bars["SMA50"].iloc[-1] else "bearish"
    except Exception:
        higher_trend = "unknown"

    # ==================== Accumulate messages ====================
    signals_messages = []

    for i in range(1, len(bars)):
        prev, curr = bars.iloc[i - 1], bars.iloc[i]
        atr_value = curr["ATR"]
        stop_distance = atr_value * ATR_MULTIPLIER
        volatility_pct = (atr_value / curr["close"]) * 100
        volume_spike = curr["volume"] > 1.5 * curr["AvgVol"]

        # Conditions
        rsi_cond_buy = curr["RSI"] > 55
        rsi_cond_sell = curr["RSI"] < 45
        adx_cond = curr["ADX"] > 25
        macd_bull = curr["MACD"] > curr["MACD_Signal"]
        macd_bear = curr["MACD"] < curr["MACD_Signal"]
        vol_cond = 0.5 < volatility_pct < 3
        volume_cond = volume_spike
        higher_cond_buy = higher_trend == "bullish"
        higher_cond_sell = higher_trend == "bearish"

        # BUY signal
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
                passed = [f"✅ {k}" for k, v in conditions.items() if v]
                failed = [f"❌ {k}" for k, v in conditions.items() if not v]

                reason = "\n".join(passed + failed)
                msg = (f"🟢 {ticker} BUY at {curr['close']:.2f} on {curr['TradeTime']}\n"
                       f"Reason: SMA20 crossed above SMA50 (Bullish trend)\n"
                       f"ATR={atr_value:.2f}, StopDist={stop_distance:.2f}\n{reason}")
                signals_messages.append(msg)
                sent_signals.add(signal_id)
                save_sent_signal(signal_id)

        # SELL signal
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
                passed = [f"✅ {k}" for k, v in conditions.items() if v]
                failed = [f"❌ {k}" for k, v in conditions.items() if not v]

                reason = "\n".join(passed + failed)
                msg = (f"🔴 {ticker} SELL at {curr['close']:.2f} on {curr['TradeTime']}\n"
                       f"Reason: SMA20 crossed below SMA50 (Bearish trend)\n"
                       f"ATR={atr_value:.2f}, StopDist={stop_distance:.2f}\n{reason}")
                signals_messages.append(msg)
                sent_signals.add(signal_id)
                save_sent_signal(signal_id)

    # ==================== Send consolidated Telegram message ====================
    if signals_messages:
        consolidated_msg = "📈 *Trade Signals Summary*\n\n" + "\n\n".join(signals_messages)
        send_message(consolidated_msg)
        log_message("📩 Sent consolidated trade signals for " + ticker)



def analyze_ticker(ticker, bars):
    if bars.empty:
        log_message(f"⚠️ No data for {ticker}, skipping.")
        return

    # ======== Handle timezone ========
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC").tz_convert("America/New_York")
    else:
        bars.index = bars.index.tz_convert("America/New_York")

    # ======== Calculate indicators ========
    bars["SMA20"] = bars["close"].rolling(SMA_SHORT).mean()
    bars["SMA50"] = bars["close"].rolling(SMA_LONG).mean()
    bars["RSI"] = compute_rsi(bars["close"], 14)
    bars["ATR"] = compute_atr(bars[["high", "low", "close"]], ATR_PERIOD)
    adx_indicator = ADXIndicator(high=bars["high"], low=bars["low"], close=bars["close"], window=14)
    bars["ADX"] = adx_indicator.adx()
    bars["MACD"], bars["MACD_Signal"] = compute_macd(bars["close"])
    bars["AvgVol"] = bars["volume"].rolling(20).mean()

    bars = bars.reset_index()
    bars["TradeTime"] = bars["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # ======== Higher timeframe confirmation ========
    try:
        higher_bars = api.get_bars(ticker, tradeapi.TimeFrame.Hour, limit=100, feed="iex").df
        higher_bars["SMA20"] = higher_bars["close"].rolling(20).mean()
        higher_bars["SMA50"] = higher_bars["close"].rolling(50).mean()
        higher_trend = "bullish" if higher_bars["SMA20"].iloc[-1] > higher_bars["SMA50"].iloc[-1] else "bearish"
    except Exception:
        higher_trend = "unknown"

    # ======== Accumulate signals ========
    signals_messages = []

    for i in range(1, len(bars)):
        prev, curr = bars.iloc[i - 1], bars.iloc[i]

        # Skip rows where SMA is NaN
        if np.isnan(prev["SMA20"]) or np.isnan(prev["SMA50"]) or np.isnan(curr["SMA20"]) or np.isnan(curr["SMA50"]):
            continue

        # Debug log for crossovers
        if prev["SMA20"] < prev["SMA50"] and curr["SMA20"] > curr["SMA50"]:
            log_message(f"DEBUG {ticker}: Potential BUY crossover at {curr['TradeTime']} SMA20={curr['SMA20']:.2f} SMA50={curr['SMA50']:.2f}")
        elif prev["SMA20"] > prev["SMA50"] and curr["SMA20"] < curr["SMA50"]:
            log_message(f"DEBUG {ticker}: Potential SELL crossover at {curr['TradeTime']} SMA20={curr['SMA20']:.2f} SMA50={curr['SMA50']:.2f}")

        # ======== Indicators for signal reason ========
        atr_value = curr["ATR"]
        stop_distance = atr_value * ATR_MULTIPLIER
        volatility_pct = (atr_value / curr["close"]) * 100
        volume_spike = curr["volume"] > 1.5 * curr["AvgVol"]

        rsi_cond_buy = curr["RSI"] > 55
        rsi_cond_sell = curr["RSI"] < 45
        adx_cond = curr["ADX"] > 25
        macd_bull = curr["MACD"] > curr["MACD_Signal"]
        macd_bear = curr["MACD"] < curr["MACD_Signal"]
        vol_cond = 0.5 < volatility_pct < 3
        volume_cond = volume_spike
        higher_cond_buy = higher_trend == "bullish"
        higher_cond_sell = higher_trend == "bearish"

        # ======== BUY signal ========
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
                passed = [f"✅ {k}" for k, v in conditions.items() if v]
                failed = [f"❌ {k}" for k, v in conditions.items() if not v]

                reason = "\n".join(passed + failed)
                msg = (f"🟢 {ticker} BUY at {curr['close']:.2f} on {curr['TradeTime']}\n"
                       f"Reason: SMA20 crossed above SMA50 (Bullish trend)\n"
                       f"ATR={atr_value:.2f}, StopDist={stop_distance:.2f}\n{reason}")
                signals_messages.append(msg)
                sent_signals.add(signal_id)
                save_sent_signal(signal_id)

        # ======== SELL signal ========
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
                passed = [f"✅ {k}" for k, v in conditions.items() if v]
                failed = [f"❌ {k}" for k, v in conditions.items() if not v]

                reason = "\n".join(passed + failed)
                msg = (f"🔴 {ticker} SELL at {curr['close']:.2f} on {curr['TradeTime']}\n"
                       f"Reason: SMA20 crossed below SMA50 (Bearish trend)\n"
                       f"ATR={atr_value:.2f}, StopDist={stop_distance:.2f}\n{reason}")
                signals_messages.append(msg)
                sent_signals.add(signal_id)
                save_sent_signal(signal_id)

    # ======== Send consolidated Telegram message ========
    if signals_messages:
        consolidated_msg = "📈 *Trade Signals Summary*\n\n" + "\n\n".join(signals_messages)
        send_message(consolidated_msg)
        log_message(f"📩 Sent consolidated trade signals for {ticker}")


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
# Optional: deterministic results
tf.random.set_seed(42)
np.random.seed(42)

def deep_learning_forecast(ticker, bars, sheet, lookback=60, forecast_steps=1, retrain=False):
    """
    Production-ready AI forecast with LSTM.
    - Auto-handles Keras deserialization issues by retraining when needed.
    - Saves and reloads per-ticker model.
    - Updates Google Sheets + Telegram automatically.
    - Uses log_message(msg) for all warnings/errors/info.
    """
    try:
        # ==================== Data Prep ====================
        df = bars[["close"]].dropna().reset_index(drop=True)
        if len(df) < lookback + forecast_steps:
            log_message(f"⚠️ Not enough data for {ticker}")
            return None

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)

        X, y = [], []
        for i in range(lookback, len(scaled) - forecast_steps):
            X.append(scaled[i - lookback:i])
            y.append(scaled[i + forecast_steps][0])
        X, y = np.array(X), np.array(y)

        # ==================== Train/Val Split ====================
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

        # ==================== Model Handling ====================
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{ticker}_lstm.h5")

        model = None
        if os.path.exists(model_path) and not retrain:
            try:
                model = load_model(model_path, compile=False)
                model.compile(optimizer="adam", loss="mse")
                log_message(f"✅ Loaded model for {ticker}")
            except Exception as e:
                log_message(f"⚠️ Model load failed for {ticker}: {e}. Retraining...")
                retrain = True

        if model is None or retrain:
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
                LSTM(32),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mse")
            es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=20, batch_size=32, verbose=0, callbacks=[es]
            )
            model.save(model_path)
            log_message(f"💾 Model retrained and saved for {ticker}")

        # ==================== Forecast ====================
        last_seq = scaled[-lookback:].reshape((1, lookback, 1))
        forecast_scaled = model.predict(last_seq)[0][0]
        forecast_price = scaler.inverse_transform([[forecast_scaled]])[0][0]
        current = df["close"].iloc[-1]

        # ==================== Confidence (based on val RMSE) ====================
        y_pred_val = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        avg_val_price = scaler.inverse_transform([[np.mean(y_val)]])[0][0]
        confidence = max(0, 100 - (rmse / avg_val_price * 100))

        # ==================== Trend ====================
        trend = "BULLISH" if forecast_price > current else "BEARISH"

        # ==================== Message ====================
        msg_text = (
            f"🤖 *AI Forecast - {ticker}*\n"
            f"Current: {current:.2f}\n"
            f"Predicted: {forecast_price:.2f}\n"
            f"Trend: {trend}\n"
            f"Confidence: {confidence:.2f}%"
        )
        log_message(msg_text)

        # ==================== Google Sheets ====================
        try:
            try:
                ws = sheet.worksheet("AI_Forecast")
            except Exception:
                ws = sheet.add_worksheet(title="AI_Forecast", rows="1000", cols="10")

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ws.append_row([
                now_str, ticker, round(current, 2),
                round(forecast_price, 2), trend, round(confidence, 2)
            ])
            log_message(f"📊 Logged in Google Sheets for {ticker}")
        except Exception as e:
            log_message(f"⚠️ Google Sheet update failed for {ticker}: {e}")


        # ==================== Output ====================
        return {
            "ticker": ticker,
            "trend": trend,
            "confidence": round(confidence, 2),
            "current": round(current, 2),
            "forecast": round(forecast_price, 2)
        }

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
    ai_forecasts = []
    for ticker in tickers:
        try:
            bars = api.get_bars(ticker, tradeapi.TimeFrame.Minute,
                                start=start_date, end=end_date, adjustment="raw", feed="iex").df
            if not bars.empty:
                analyze_ticker(ticker, bars)
                result = deep_learning_forecast(ticker, bars, sheet)
                if result:
                    ai_forecasts.append(result)
            else:
                log_message(f"No bars found for {ticker}")
        except Exception as e:
            log_message(f"Error fetching bars for {ticker}: {e}")
      # Send AI forecast to Telegram
    if ai_forecasts:
        msg = "🤖 *AI Forecasts Summary — Current Day / Previous Day Analysis*\n\n"
        for f in ai_forecasts:
            # Determine reason based on predicted vs current price
            reason = "Predicted price > current price" if f['trend'] == "BULLISH" else "Predicted price < current price"
            
            msg += (
                f"{f['ticker']}: Current {f['current']:.2f} | "
                f"Predicted {f['forecast']:.2f} | "
                f"Trend {f['trend']} ({reason}) | "
                f"Conf {f['confidence']}%\n"
            )
        
        send_message(msg)
    log_message("📌 Full-day analysis completed.")

# ================= LIVE TRADING LOOP =================
def live_trading_loop():
    log_message("🚀 Starting live trading loop (runs every 10 mins)")
    now = datetime.now(EST)
    log_message(f"📈 Running live analysis at {now.strftime('%b %d %H:%M')}")

    summary_messages = []
    if datetime.now(EST).weekday() >= 5:  # 5=Sat, 6=Sun
        log_message("ℹ️ Weekend detected. Skipping live trading loop.")
        return
    
    for ticker in tickers:
        try:
            end = datetime.now(EST)
            start = end - timedelta(hours=6)
            bars = api.get_bars(
                ticker,
                tradeapi.TimeFrame.Minute,
                start=start.isoformat(),
                end=end.isoformat(),
                feed="iex"
            ).df

            if bars.empty:
                log_message(f"⚠️ No market data for {ticker}, skipping.")
                continue

            # --- Calculate latest indicators for message ---
            latest = bars.iloc[-1]
            sma20 = latest["close"].rolling(SMA_SHORT).mean() if "SMA20" not in bars else latest["SMA20"]
            sma50 = latest["close"].rolling(SMA_LONG).mean() if "SMA50" not in bars else latest["SMA50"]
            rsi = compute_rsi(bars["close"], 14).iloc[-1]
            atr = compute_atr(bars[["high","low","close"]], ATR_PERIOD).iloc[-1]
            macd, macd_signal = compute_macd(bars["close"])
            macd_latest = macd.iloc[-1]
            macd_signal_latest = macd_signal.iloc[-1]
            adx_indicator = ADXIndicator(bars["high"], bars["low"], bars["close"], window=14)
            adx = adx_indicator.adx().iloc[-1]

            # --- Technical summary string ---
            technical_summary = (
                f"• Indicators: Close ${latest['close']:.2f} | "
                f"SMA20 {sma20:.2f} / SMA50 {sma50:.2f} | "
                f"RSI {rsi:.1f} | ATR {atr:.2f} | "
                f"MACD {macd_latest:.2f}/{macd_signal_latest:.2f} | ADX {adx:.1f}"
            )

            # --- AI Deep Learning Forecast ---
            forecast_result = deep_learning_forecast(ticker, bars, sheet)
            if forecast_result:
                trend = forecast_result["trend"]
                reason = (
                    "Predicted ↑ (Bullish)" if trend == "BULLISH" else "Predicted ↓ (Bearish)"
                )
                forecast_summary = (
                    f"• AI Forecast: Current ${forecast_result['current']:.2f} → "
                    f"Forecast ${forecast_result['forecast']:.2f} | "
                    f"Trend {trend} | Conf {forecast_result['confidence']}% | {reason}"
                )
            else:
                forecast_summary = "• AI Forecast: N/A"

            # --- Combine into one message per ticker ---
            ticker_msg = f"📊 *{ticker}*\n{technical_summary}\n{forecast_summary}"
            summary_messages.append(ticker_msg)

        except Exception as e:
            log_message(f"❌ Error processing {ticker}: {e}")
            continue

    # --- Send consolidated Telegram summary ---
    if summary_messages:
        msg = f"📈 *AI + Technical Morning Scan — {now.strftime('%b %d %H:%M')}*\n\n"
        msg += "\n\n".join(summary_messages)
        send_message(msg)
        log_message("📩 Sent consolidated AI + Technical summary to Telegram.")
    else:
        log_message("ℹ️ No data or signals available this cycle.")

    log_message("⏳ Sleeping 10 minutes...")

        

# ================= MORNING SCAN =================
def morning_scan():
    """
    Runs morning scan:
    - Fetches last 5 days of minute bars
    - Detects bullish/bearish SMA crossovers
    - Sends AI forecast and technical signals to Telegram
    - Updates Google Sheets
    """
    log_message("🌅 Starting Morning Market Scanner...")
    potential_stocks = []
    ai_forecasts = []

    for ticker in tickers:
        try:
            # Fetch last 5 days of minute bars
            end_dt = datetime.now(EST)
            start_dt = end_dt - timedelta(days=5)
            bars = api.get_bars(ticker, tradeapi.TimeFrame.Minute,
                                start=start_dt.isoformat(),
                                end=end_dt.isoformat(),
                                feed="iex").df

            if bars.empty or len(bars) < 20:
                log_message(f"⚠️ Not enough data for {ticker}, skipping")
                continue

            # Localize timezone
            bars.index = bars.index.tz_localize("UTC").tz_convert("America/New_York") if bars.index.tz is None else bars.index.tz_convert("America/New_York")

            # AI Forecast
            if len(bars) >= 60:
                result = deep_learning_forecast(ticker, bars, sheet)
                if result:
                    ai_forecasts.append(result)

            # Compute indicators
            bars["SMA20"] = bars["close"].rolling(20).mean()
            bars["SMA50"] = bars["close"].rolling(50).mean()
            bars["RSI"] = compute_rsi(bars["close"], 14)
            bars["ADX"] = ta.trend.adx(bars["high"], bars["low"], bars["close"], window=14)
            bars["ATR"] = compute_atr(bars[["high", "low", "close"]], ATR_PERIOD)
            bars["MACD"], bars["MACD_Signal"] = compute_macd(bars["close"])
            bars["AvgVol"] = bars["volume"].rolling(20).mean()

            # Last two bars for crossover
            prev = bars.iloc[-2]
            curr = bars.iloc[-1]

            sma_gap = abs((curr["SMA20"] - curr["SMA50"]) / curr["SMA50"]) * 100
            macd_gap = abs(curr["MACD"] - curr["MACD_Signal"])
            atr_pct = (curr["ATR"] / curr["close"]) * 100
            vol_ratio = curr["volume"] / curr["AvgVol"]

            # Bullish/Bearish detection based on crossover
            bullish_ready = prev["SMA20"] <= prev["SMA50"] and curr["SMA20"] > curr["SMA50"]
            bearish_ready = prev["SMA20"] >= prev["SMA50"] and curr["SMA20"] < curr["SMA50"]

            if bullish_ready or bearish_ready:
                direction = "BULLISH" if bullish_ready else "BEARISH"
                score = sum([
                    bullish_ready or bearish_ready,
                    curr["ADX"] > 20,
                    vol_ratio > 1.0,
                    macd_gap < 1.0
                ])
                potential_stocks.append({
                    "Ticker": ticker,
                    "Direction": direction,
                    "Price": round(curr["close"], 2),
                    "RSI": round(curr["RSI"], 1),
                    "ADX": round(curr["ADX"], 1),
                    "VolSpike": round(vol_ratio, 2),
                    "SMA_Gap%": round(sma_gap, 2),
                    "Score": score
                })

        except Exception as e:
            log_message(f"⚠️ Error scanning {ticker}: {e}")

    # Send AI forecast to Telegram
    if ai_forecasts:
        msg = "🤖 *AI Forecasts Summary — Morning Scan*\n\n"
        for f in ai_forecasts:
            # Determine reason based on predicted vs current price
            reason = "Predicted price > current price" if f['trend'] == "BULLISH" else "Predicted price < current price"
            
            msg += (
                f"{f['ticker']}: Current {f['current']:.2f} | "
                f"Predicted {f['forecast']:.2f} | "
                f"Trend {f['trend']} ({reason}) | "
                f"Conf {f['confidence']}%\n"
            )
        
        send_message(msg)

    # No technical signals
    if not potential_stocks:
        send_message("🌄 Morning scan complete — no potential setups found.")
        return

    # Prepare top 10 signals
    df = pd.DataFrame(potential_stocks).sort_values("Score", ascending=False)
    top10 = df.head(10)

    # Update Google Sheet
    try:
        ws = sheet.worksheet("Morning_Scanner")
        ws.clear()
        ws.update([top10.columns.values.tolist()] + top10.values.tolist())
        log_message("📊 Morning scanner results updated to Google Sheets.")
    except Exception as e:
        log_message(f"⚠️ Failed to update Morning_Scanner sheet: {e}")

    # Send technical signals to Telegram
    msg = "🌅 *Morning Scanner — Potential Stocks for Today*\n\n"
    for _, row in top10.iterrows():
        msg += f"{row['Ticker']}: {row['Direction']} | Price: {row['Price']} | RSI: {row['RSI']} | ADX: {row['ADX']} | Vol x{row['VolSpike']} | SMA Gap: {row['SMA_Gap%']}%\n"
    send_message(msg)
    log_message("✅ Morning scan completed successfully.")


def fetch_top_movers():
    """Fetch top gainers and losers from FMP API."""
    base_url = "https://financialmodelingprep.com/stable"
    top_movers = {}

    try:
        gainers = requests.get(f"{base_url}/biggest-gainers?apikey={api_key}", timeout=10)
        losers = requests.get(f"{base_url}/biggest-losers?apikey={api_key}", timeout=10)
        gainers.raise_for_status()
        losers.raise_for_status()

        top_movers["gainers"] = gainers.json()
        top_movers["losers"] = losers.json()

        return top_movers["gainers"], top_movers["losers"]

    except requests.exceptions.RequestException as e:
        log_message(f"Error fetching top movers: {e}")
        return [], []

def send_top_movers_to_telegram(top_n=15):
    gainers, losers = fetch_top_movers()
    if not gainers and not losers:
        send_message("⚠️ Failed to fetch top movers from FMP.")
        return

    msg = "📈 *Top Gainers*\n"
    for stock in gainers[:top_n]:
        msg += f"{stock['symbol']} | {stock['price']} | {stock['changesPercentage']}\n"

    msg += "\n📉 *Top Losers*\n"
    for stock in losers[:top_n]:
        msg += f"{stock['symbol']} | {stock['price']} | {stock['changesPercentage']}\n"

    send_message(msg)
    print("✅ Top movers sent to Telegram.")


# ================= ENTRY POINT =================
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "analysis"

    if mode == "analysis":
        previous_day_analysis()
    elif mode == "live":
        live_trading_loop()
    elif mode == "morning":
        send_top_movers_to_telegram(top_n=15)
        morning_scan()
       
    else:
        log_message(f"Unknown mode: {mode}")


