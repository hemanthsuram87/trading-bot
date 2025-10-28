import os
import time
import pytz
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time as dt_time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import logging
# TA libs (some functions use ta)
import ta
from ta.trend import ADXIndicator

# Alpaca
import alpaca_trade_api as tradeapi


import matplotlib.pyplot as plt


import joblib
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ta.trend import ADXIndicator

# ================= CONFIG =================
ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# Telegram config
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
api_key = os.getenv("FMP_API_KEY")

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
        df = bars[["close"]].dropna().reset_index(drop=True)
        if len(df) < lookback + forecast_steps + 1:
            log_message(f"⚠️ Not enough data for LSTM {ticker}")
            return None

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df.values.reshape(-1, 1))

        X, y = [], []
        for i in range(lookback, len(scaled) - forecast_steps):
            X.append(scaled[i - lookback:i])
            y.append(scaled[i + forecast_steps][0])
        X, y = np.array(X), np.array(y)
        if X.size == 0:
            return None

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

        model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.h5")
        model = None
        if os.path.exists(model_path) and not retrain:
            try:
                model = load_model(model_path, compile=False)
                model.compile(optimizer="adam", loss="mse")
                log_message(f"✅ Loaded model for {ticker}")
            except Exception as e:
                log_message(f"⚠️ Model load failed ({ticker}): {e}")
                model = None

        if model is None:
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
                LSTM(32),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mse")
            es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=0, callbacks=[es])
            model.save(model_path)
            log_message(f"💾 Model trained & saved for {ticker}")

        last_seq = scaled[-lookback:].reshape((1, lookback, 1))
        forecast_scaled = model.predict(last_seq, verbose=0)[0][0]
        forecast_price = scaler.inverse_transform([[forecast_scaled]])[0][0]
        current = df["close"].iloc[-1]

        # validation rmse -> confidence
        y_pred_val = model.predict(X_val, verbose=0)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        avg_val_price = np.mean(scaler.inverse_transform(y_val.reshape(-1, 1)))
        confidence = max(0, 100 - (rmse / (avg_val_price + 1e-9) * 100))

        trend = "BULLISH" if forecast_price > current else "BEARISH"

        # Save to sheet if available
        if sheet:
            try:
                try:
                    ws = sheet.worksheet("AI_Forecast")
                except Exception:
                    ws = sheet.add_worksheet(title="AI_Forecast", rows="1000", cols="10")
                ws.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ticker, round(current, 2), round(forecast_price, 2), trend, round(confidence, 2)])
            except Exception as e:
                log_message(f"⚠️ Sheets update failed for AI forecast: {e}")

        return {"ticker": ticker, "trend": trend, "confidence": round(confidence, 2), "current": round(current, 2), "forecast": round(forecast_price, 2)}

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

# ================= TOP MOVERS =================
def fetch_top_movers():
    if not FMP_API_KEY:
        log_message("⚠️ FMP API key not set.")
        return [], []
    base_url = "https://financialmodelingprep.com/api/v3"
    try:
        g = requests.get(f"{base_url}/stock/gainers?apikey={FMP_API_KEY}", timeout=10)
        l = requests.get(f"{base_url}/stock/losers?apikey={FMP_API_KEY}", timeout=10)
        g.raise_for_status(); l.raise_for_status()
        return g.json(), l.json()
    except Exception as e:
        log_message(f"Error fetching top movers: {e}")
        return [], []

def send_top_movers_to_telegram(top_n=15):
    gainers, losers = fetch_top_movers()
    if not gainers and not losers:
        send_message("⚠️ Failed to fetch top movers from FMP.")
        return
    msg = "📈 *Top Gainers*\n"
    for s in gainers[:top_n]:
        msg += f"{s.get('ticker', s.get('symbol',''))} | {s.get('price','')} | {s.get('changesPercentage','')}\n"
    msg += "\n📉 *Top Losers*\n"
    for s in losers[:top_n]:
        msg += f"{s.get('ticker', s.get('symbol',''))} | {s.get('price','')} | {s.get('changesPercentage','')}\n"
    send_message(msg)
    log_message("✅ Top movers sent")

# ================= POSITION MANAGEMENT (auto-close) =================
def manage_positions():
    try:
        positions = api.list_positions()
        for pos in positions:
            try:
                symbol = pos.symbol
                qty = float(pos.qty)
                unreal_plpc = float(pos.unrealized_plpc) if pos.unrealized_plpc is not None else 0.0
                # unrealized_plpc is in fraction (e.g. 0.02), convert to percent:
                unreal_pct = unreal_plpc * 100
                # compute hold time from position's 'side' not always available; fallback skip time check
                # Alpaca doesn't provide opened_at in Position object; we'll do a safe check via account activities (optional)
                # For simplicity, auto-close if unreal_pct <= CLOSE_LOSS_PCT
                if unreal_pct <= CLOSE_LOSS_PCT:
                    log_message(f"⚠️ Auto-closing {symbol} due to unrealized pct {unreal_pct:.2f}% <= {CLOSE_LOSS_PCT}%")
                    api.close_position(symbol)
                    send_message(f"⚠️ Closed {symbol} (auto-close) due to loss threshold {CLOSE_LOSS_PCT}%")
                   
            except Exception as e:
                log_message(f"⚠️ manage_positions per-position error: {e}")
    except Exception as e:
        log_message(f"⚠️ manage_positions error: {e}")

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

# ================= LIVE TRADING LOOP (single-run job) =================
def live_trading_loop(backtest_days=0):
    """
    Runs the live trading loop and optionally backtests the last N days.
    
    Parameters:
        backtest_days (int): Number of previous days to simulate for backtesting.
                             0 = live only.
    """
    now = datetime.now(EST)
    log_message(f"🚀 Starting trading loop at {now.strftime('%Y-%m-%d %H:%M:%S')}")

    if now.weekday() >= 5:
        log_message("ℹ️ Weekend — skipping live trading.")
        return

    summary_messages = []

    for ticker in tickers:
        try:
            # ================= BACKTEST =================
            if backtest_days > 0:
                log_message(f"🧪 Running backtest for {ticker} ({backtest_days} days)")
                df_trades = backtest_ticker(
                    ticker,
                    start_date=(now - timedelta(days=backtest_days)).isoformat(),
                    end_date=now.isoformat()
                )
                if df_trades is not None and not df_trades.empty:
                    total_trades = len(df_trades)
                    pnl_total = df_trades["pnl"].sum()
                    win_rate = len(df_trades[df_trades["pnl"] > 0]) / total_trades * 100
                    summary_messages.append(
                        f"📊 Backtest {ticker}: Trades={total_trades}, "
                        f"P&L=${pnl_total:.2f}, Win Rate={win_rate:.1f}%"
                    )

            # ================= LIVE DATA =================
            log_message(f"🔍 Fetching live data for {ticker}...")
            end = now
            start = end - timedelta(hours=6)
            bars = api.get_bars(
                ticker, tradeapi.TimeFrame.Minute,
                start=start.isoformat(), end=end.isoformat(), feed="iex"
            ).df

            if bars.empty:
                log_message(f"⚠️ No market data for {ticker}")
                continue

            bars = safe_tz_localize_and_convert(bars)

            # ================= FEATURE ENGINEERING =================
            bars["SMA20"] = bars["close"].rolling(SMA_SHORT).mean()
            bars["SMA50"] = bars["close"].rolling(SMA_LONG).mean()
            bars["RSI"] = compute_rsi(bars["close"], 14)
            bars["ATR"] = compute_atr(bars[["high", "low", "close"]], ATR_PERIOD)
            macd, macd_signal = compute_macd(bars["close"])
            adx_indicator = ADXIndicator(bars["high"], bars["low"], bars["close"], window=14)
            adx = adx_indicator.adx().iloc[-1] if len(bars) >= 14 else np.nan

            latest = bars.iloc[-1]
            sma20, sma50 = bars["SMA20"].iloc[-1], bars["SMA50"].iloc[-1]
            rsi, atr = bars["RSI"].iloc[-1], bars["ATR"].iloc[-1]
            macd_latest, macd_signal_latest = macd.iloc[-1], macd_signal.iloc[-1]

            # ================= AI FORECAST =================
            forecast_result = deep_learning_forecast(ticker, bars, sheet if 'sheet' in globals() else None)
            if forecast_result:
                forecast_price = forecast_result["forecast"]
                current_price = forecast_result["current"]
                trend = forecast_result["trend"]
                confidence = forecast_result["confidence"]
            else:
                forecast_price = current_price = trend = confidence = None

            # ================= SIGNAL GENERATION =================
            signal_reason = []
            action = "HOLD"

            if sma20 > sma50 and macd_latest > macd_signal_latest and rsi < 70 and adx > 20:
                action = "BUY"
                signal_reason.extend([
                    "✅ SMA20 above SMA50 → bullish trend",
                    "✅ MACD bullish crossover",
                    f"✅ RSI healthy at {rsi:.1f}",
                    f"✅ ADX {adx:.1f} indicates strong trend"
                ])
                if trend == "Up" and confidence >= 60:
                    signal_reason.append(f"🤖 AI confirms upward momentum ({confidence}% confidence)")

            elif sma20 < sma50 and macd_latest < macd_signal_latest and rsi > 30 and adx > 20:
                action = "SELL"
                signal_reason.extend([
                    "❌ SMA20 below SMA50 → bearish trend",
                    "❌ MACD bearish crossover",
                    f"❌ RSI weakening at {rsi:.1f}",
                    f"❌ ADX {adx:.1f} indicates strong downtrend"
                ])
                if trend == "Down" and confidence >= 60:
                    signal_reason.append(f"🤖 AI confirms downward momentum ({confidence}% confidence)")
            else:
                action = "HOLD"
                signal_reason.append("⚖️ Mixed or neutral signals — waiting for confirmation.")

            # ================= TECHNICAL & FORECAST SUMMARY =================
            technical_summary = (
                f"• Close ${latest['close']:.2f} | SMA20 {sma20:.2f} / SMA50 {sma50:.2f}\n"
                f"• RSI {rsi:.1f} | ATR {atr:.2f} | MACD {macd_latest:.2f}/{macd_signal_latest:.2f} | ADX {adx:.1f}"
            )

            forecast_summary = (
                f"• AI Forecast: ${current_price:.2f} → ${forecast_price:.2f} ({trend}, {confidence}% confidence)"
                if forecast_result else "• AI Forecast: N/A"
            )

            signal_summary = f"🔔 *Signal: {action}*\n" + "\n".join(f"→ {r}" for r in signal_reason)

            ticker_msg = (
                f"📊 *{ticker}*\n"
                f"{technical_summary}\n"
                f"{forecast_summary}\n\n"
                f"{signal_summary}"
            )

            summary_messages.append(ticker_msg)
            log_message(f"{ticker}: {action} — {', '.join(signal_reason)}")

        except Exception as e:
            log_message(f"❌ Error processing {ticker}: {e}")
            continue

    # ================= SEND SUMMARY TO TELEGRAM =================
    if summary_messages:
        msg = f"📈 *AI + Technical Live Scan & Backtest — {now.strftime('%b %d %H:%M')}*\n\n" + "\n\n".join(summary_messages)
        send_message(msg)
        log_message("📩 Sent consolidated summary to Telegram")
    else:
        log_message("ℹ️ No actionable signals this run.")

    # ================= AUTO POSITION MANAGEMENT =================
    try:
        manage_positions()
        log_message("✅ Position management complete")
    except Exception as e:
        log_message(f"⚠️ manage_positions failed: {e}")


# -------------------- FULL INTEGRATION: REGIME / FEATURES / MODEL / RISK / EXEC --------------------
import os
import joblib
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ta.trend import ADXIndicator

# Config (override via environment)
MODEL_DIR = os.getenv("MODEL_DIR", "models/sklearn")
os.makedirs(MODEL_DIR, exist_ok=True)

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

# ---------------- Auto-train pipeline ----------------
def auto_train_models():
    """
    Train models for all tickers using recent historical data and save them.
    Intended to run nightly (GitHub Action or cron).
    """
    log_message("🧠 Starting auto_train_models pipeline...")
    trained = 0
    for ticker in tickers:
        try:
            # fetch historical minute bars for TRAIN_DAYS days; use daily if minute not available
            end = datetime.now(EST)
            start = end - timedelta(days=TRAIN_DAYS)
            try:
                bars = api.get_bars(ticker, tradeapi.TimeFrame.Minute, start=start.isoformat(), end=end.isoformat(), feed="iex").df
            except Exception:
                bars = api.get_bars(ticker, tradeapi.TimeFrame.Day, limit=TRAIN_DAYS).df

            if bars.empty or len(bars) < TRAIN_MIN_SAMPLES:
                log_message(f"⚠️ Not enough data for training {ticker} (have {len(bars)} rows)")
                continue

            feat_df, full_df = create_features(bars)
            labels = make_labels_for_direction(full_df, horizon=5, threshold=0.001)
            model = train_and_save_model(ticker, feat_df, labels, force_retrain=True)
            if model:
                trained += 1
        except Exception as e:
            log_message(f"❌ auto_train_models failed for {ticker}: {e}")
            continue

    log_message(f"🏁 auto_train_models complete — models trained: {trained}/{len(tickers)}")

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
def live_trading_loop(backtest_days=0):
    """
    Runs the live trading loop and optionally backtests the last N days.
    
    Parameters:
        backtest_days (int): Number of previous days to simulate for backtesting.
                             0 = live only.
    """
    now = datetime.now(EST)
    log_message(f"🚀 Starting trading loop at {now.strftime('%Y-%m-%d %H:%M:%S')}")

    if now.weekday() >= 5:
        log_message("ℹ️ Weekend — skipping live trading.")
        return

    summary_messages = []

    for ticker in tickers:
        try:
            # ================= BACKTEST =================
            if backtest_days > 0:
                log_message(f"🧪 Running backtest for {ticker} ({backtest_days} days)")
                df_trades = backtest_ticker(
                    ticker,
                    start_date=(now - timedelta(days=backtest_days)).isoformat(),
                    end_date=now.isoformat()
                )
                if df_trades is not None and not df_trades.empty:
                    total_trades = len(df_trades)
                    pnl_total = df_trades["pnl"].sum()
                    win_rate = len(df_trades[df_trades["pnl"] > 0]) / total_trades * 100
                    summary_messages.append(
                        f"📊 Backtest {ticker}: Trades={total_trades}, "
                        f"P&L=${pnl_total:.2f}, Win Rate={win_rate:.1f}%"
                    )

            # ================= LIVE DATA =================
            log_message(f"🔍 Fetching live data for {ticker}...")
            end = now
            start = end - timedelta(hours=6)
            bars = api.get_bars(
                ticker, tradeapi.TimeFrame.Minute,
                start=start.isoformat(), end=end.isoformat(), feed="iex"
            ).df

            if bars.empty:
                log_message(f"⚠️ No market data for {ticker}")
                continue

            bars = safe_tz_localize_and_convert(bars)

            # ================= FEATURE ENGINEERING =================
            bars["SMA20"] = bars["close"].rolling(SMA_SHORT).mean()
            bars["SMA50"] = bars["close"].rolling(SMA_LONG).mean()
            bars["RSI"] = compute_rsi(bars["close"], 14)
            bars["ATR"] = compute_atr(bars[["high", "low", "close"]], ATR_PERIOD)
            macd, macd_signal = compute_macd(bars["close"])
            adx_indicator = ADXIndicator(bars["high"], bars["low"], bars["close"], window=14)
            adx = adx_indicator.adx().iloc[-1] if len(bars) >= 14 else np.nan

            latest = bars.iloc[-1]
            sma20, sma50 = bars["SMA20"].iloc[-1], bars["SMA50"].iloc[-1]
            rsi, atr = bars["RSI"].iloc[-1], bars["ATR"].iloc[-1]
            macd_latest, macd_signal_latest = macd.iloc[-1], macd_signal.iloc[-1]

            # ================= AI FORECAST =================
            forecast_result = deep_learning_forecast(ticker, bars, sheet if 'sheet' in globals() else None)
            if forecast_result:
                forecast_price = forecast_result["forecast"]
                current_price = forecast_result["current"]
                trend = forecast_result["trend"]
                confidence = forecast_result["confidence"]
            else:
                forecast_price = current_price = trend = confidence = None

            # ================= SIGNAL GENERATION =================
            signal_reason = []
            action = "HOLD"

            if sma20 > sma50 and macd_latest > macd_signal_latest and rsi < 70 and adx > 20:
                action = "BUY"
                signal_reason.extend([
                    "✅ SMA20 above SMA50 → bullish trend",
                    "✅ MACD bullish crossover",
                    f"✅ RSI healthy at {rsi:.1f}",
                    f"✅ ADX {adx:.1f} indicates strong trend"
                ])
                if trend == "Up" and confidence >= 60:
                    signal_reason.append(f"🤖 AI confirms upward momentum ({confidence}% confidence)")

            elif sma20 < sma50 and macd_latest < macd_signal_latest and rsi > 30 and adx > 20:
                action = "SELL"
                signal_reason.extend([
                    "❌ SMA20 below SMA50 → bearish trend",
                    "❌ MACD bearish crossover",
                    f"❌ RSI weakening at {rsi:.1f}",
                    f"❌ ADX {adx:.1f} indicates strong downtrend"
                ])
                if trend == "Down" and confidence >= 60:
                    signal_reason.append(f"🤖 AI confirms downward momentum ({confidence}% confidence)")
            else:
                action = "HOLD"
                signal_reason.append("⚖️ Mixed or neutral signals — waiting for confirmation.")

            # ================= TECHNICAL & FORECAST SUMMARY =================
            technical_summary = (
                f"• Close ${latest['close']:.2f} | SMA20 {sma20:.2f} / SMA50 {sma50:.2f}\n"
                f"• RSI {rsi:.1f} | ATR {atr:.2f} | MACD {macd_latest:.2f}/{macd_signal_latest:.2f} | ADX {adx:.1f}"
            )

            forecast_summary = (
                f"• AI Forecast: ${current_price:.2f} → ${forecast_price:.2f} ({trend}, {confidence}% confidence)"
                if forecast_result else "• AI Forecast: N/A"
            )

            signal_summary = f"🔔 *Signal: {action}*\n" + "\n".join(f"→ {r}" for r in signal_reason)

            ticker_msg = (
                f"📊 *{ticker}*\n"
                f"{technical_summary}\n"
                f"{forecast_summary}\n\n"
                f"{signal_summary}"
            )

            summary_messages.append(ticker_msg)
            log_message(f"{ticker}: {action} — {', '.join(signal_reason)}")

        except Exception as e:
            log_message(f"❌ Error processing {ticker}: {e}")
            continue

    # ================= SEND SUMMARY TO TELEGRAM =================
    if summary_messages:
        msg = f"📈 *AI + Technical Live Scan & Backtest — {now.strftime('%b %d %H:%M')}*\n\n" + "\n\n".join(summary_messages)
        send_message(msg)
        log_message("📩 Sent consolidated summary to Telegram")
    else:
        log_message("ℹ️ No actionable signals this run.")

    # ================= AUTO POSITION MANAGEMENT =================
    try:
        manage_positions()
        log_message("✅ Position management complete")
    except Exception as e:
        log_message(f"⚠️ manage_positions failed: {e}")




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


# ================= MORNING SCAN =================
def morning_scan():
    log_message("🌅 Starting Morning Scan")
    potential = []
    ai_forecasts = []
    for ticker in tickers:
        try:
            end_dt = datetime.now(EST)
            start_dt = end_dt - timedelta(days=5)
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
            msg += f"{f['ticker']}: Current {f['current']:.2f} | Predicted {f['forecast']:.2f} | {f['trend']} | {reason} | Conf {f['confidence']}%\n"
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

    msg = "🌅 *Morning Scanner — Potential Stocks for Today*\n\n"
    for _, r in top10.iterrows():
        msg += f"{r['Ticker']}: {r['Direction']} | Price: {r['Price']} | RSI: {r['RSI']} | ADX: {r['ADX']} | Vol x{r['VolSpike']} | SMA Gap: {r['SMA_Gap%']}%\n"
    send_message(msg)
    log_message("✅ Morning scan completed")


def auto_train_models():
    import joblib, os
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    os.makedirs("models/sklearn", exist_ok=True)
    log_message("🚀 Starting nightly auto-train pipeline...")

    for ticker in tickers:
        try:
            bars = api.get_bars(ticker, tradeapi.TimeFrame.Day, limit=200).df
            if bars.empty:
                continue

            features, target = create_features(bars)
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)

            joblib.dump(model, f"models/sklearn/{ticker}_model.pkl")
            log_message(f"✅ {ticker} model trained & saved (Accuracy: {score:.2f})")

        except Exception as e:
            log_message(f"❌ Error training model for {ticker}: {e}")


def retrain_all_models():
    log_message("🔁 Running full model retrain (weekly refresh)...")
    auto_train_models()
    log_message("🏁 Weekly retrain complete.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python moving_average_bot.py [morning|analysis|live|backtest|auto_train]")
        sys.exit(1)
    mode = sys.argv[1]
    if mode == "morning":
        live_trading_loop()
    elif mode == "analysis":
        previous_day_analysis()
    elif mode == "live":
        live_trading_loop()
    elif mode == "backtest":
        backtest()
    elif mode == "auto_train":
        auto_train_models()
