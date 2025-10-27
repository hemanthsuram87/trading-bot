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
def live_trading_loop():
    log_message("🚀 Starting live trading loop v2 (hybrid scoring + risk control)")
    now = datetime.now(EST)
    log_message(f"📈 Running live analysis at {now.strftime('%Y-%m-%d %H:%M:%S')}")

    if now.weekday() >= 5:
        log_message("ℹ️ Weekend detected — skipping.")
        return

    summary_messages = []

    # configuration for scoring & risk
    VOL_MULTIPLIER = 1.5      # volume surge multiplier
    BREAKOUT_LOOKBACK = 50    # lookback for breakout high/low
    ROC_PERIOD = 10           # % change period for momentum check
    MIN_BARS = max(SMA_LONG, 60, BREAKOUT_LOOKBACK, ROC_PERIOD)  # require enough bars

    for ticker in tickers:
        try:
            log_message(f"🔍 Analyzing {ticker}...")

            end = datetime.now(EST)
            start = end - timedelta(hours=6)
            bars = api.get_bars(
                ticker,
                tradeapi.TimeFrame.Minute,
                start=start.isoformat(),
                end=end.isoformat(),
                feed="iex"
            ).df

            if bars.empty or len(bars) < MIN_BARS:
                log_message(f"⚠️ Not enough market data for {ticker} (have {len(bars)} bars, need {MIN_BARS})")
                continue

            bars = safe_tz_localize_and_convert(bars.copy())

            # --- Indicators / features
            bars["SMA20"] = bars["close"].rolling(SMA_SHORT).mean()
            bars["SMA50"] = bars["close"].rolling(SMA_LONG).mean()
            bars["RSI"] = compute_rsi(bars["close"], 14)
            bars["ATR"] = compute_atr(bars[["high", "low", "close"]], ATR_PERIOD)
            bars["Vol_MA20"] = bars["volume"].rolling(20).mean()
            macd, macd_signal = compute_macd(bars["close"])
            bars["MACD"] = macd
            bars["MACD_SIGNAL"] = macd_signal
            bars["ROC"] = bars["close"].pct_change(ROC_PERIOD) * 100

            # ADX (safe)
            try:
                adx_indicator = ADXIndicator(bars["high"], bars["low"], bars["close"], window=14)
                bars["ADX"] = adx_indicator.adx()
            except Exception:
                bars["ADX"] = np.nan

            latest = bars.iloc[-1]

            # safe extraction (avoid scalar .rolling on single value)
            sma20 = float(bars["SMA20"].iloc[-1]) if not pd.isna(bars["SMA20"].iloc[-1]) else np.nan
            sma50 = float(bars["SMA50"].iloc[-1]) if not pd.isna(bars["SMA50"].iloc[-1]) else np.nan
            rsi = float(bars["RSI"].iloc[-1]) if not pd.isna(bars["RSI"].iloc[-1]) else np.nan
            atr = float(bars["ATR"].iloc[-1]) if not pd.isna(bars["ATR"].iloc[-1]) else np.nan
            vol_ma = float(bars["Vol_MA20"].iloc[-1]) if not pd.isna(bars["Vol_MA20"].iloc[-1]) else np.nan
            volume_surge = (latest["volume"] > VOL_MULTIPLIER * (vol_ma if vol_ma > 0 else 1))
            macd_latest = float(bars["MACD"].iloc[-1]) if not pd.isna(bars["MACD"].iloc[-1]) else np.nan
            macd_signal_latest = float(bars["MACD_SIGNAL"].iloc[-1]) if not pd.isna(bars["MACD_SIGNAL"].iloc[-1]) else np.nan
            roc = float(bars["ROC"].iloc[-1]) if not pd.isna(bars["ROC"].iloc[-1]) else 0.0
            adx = float(bars["ADX"].iloc[-1]) if not pd.isna(bars["ADX"].iloc[-1]) else np.nan

            # breakout detection (compare to previous high/low over lookback, exclude current bar)
            if len(bars) >= BREAKOUT_LOOKBACK + 1:
                lookback_high = bars["high"].rolling(BREAKOUT_LOOKBACK).max().iloc[-2]
                lookback_low = bars["low"].rolling(BREAKOUT_LOOKBACK).min().iloc[-2]
                breakout_up = latest["close"] > (lookback_high if not pd.isna(lookback_high) else float("inf"))
                breakout_down = latest["close"] < (lookback_low if not pd.isna(lookback_low) else float("-inf"))
            else:
                breakout_up = breakout_down = False

            # --- AI forecast (weighted) ---
            forecast_result = deep_learning_forecast(ticker, bars, sheet if 'sheet' in globals() else None)
            ai_trend = None
            ai_conf = 0.0
            if forecast_result:
                ai_trend = forecast_result.get("trend")
                ai_conf = float(forecast_result.get("confidence", 0.0))

            # --- Composite scoring (weighted) ---
            score = 0.0
            reasons = []

            # Feature: SMA alignment
            if sma20 > sma50:
                score += 1.0
                reasons.append("SMA20 > SMA50 (trend up) [+1]")
            else:
                reasons.append("SMA20 <= SMA50 (no uptrend) [+0]")

            # Feature: MACD momentum
            if macd_latest > macd_signal_latest:
                score += 1.0
                reasons.append("MACD > Signal (momentum up) [+1]")
            else:
                reasons.append("MACD <= Signal (momentum weak) [+0]")

            # Feature: ADX strength
            if not pd.isna(adx) and adx > 25:
                score += 0.5
                reasons.append(f"ADX {adx:.1f} (trend strong) [+0.5]")
            else:
                reasons.append(f"ADX {adx if not pd.isna(adx) else 'n/a'} (trend weak) [+0]")

            # Feature: Volume confirmation
            if volume_surge:
                score += 0.5
                reasons.append(f"Volume surge (x{latest['volume'] / (vol_ma if vol_ma>0 else 1):.2f}) [+0.5]")
            else:
                reasons.append("No strong volume confirmation [+0]")

            # Feature: ROC momentum
            if roc > 0.5:
                score += 0.3
                reasons.append(f"ROC +{roc:.2f}% (momentum up) [+0.3]")
            elif roc < -0.5:
                score -= 0.3
                reasons.append(f"ROC {roc:.2f}% (momentum down) [-0.3]")
            else:
                reasons.append(f"ROC {roc:.2f}% (neutral) [+0]")

            # Feature: Breakout
            if breakout_up:
                score += 0.7
                reasons.append(f"Price broke above {BREAKOUT_LOOKBACK}-bar high (breakout) [+0.7]")
            if breakout_down:
                score -= 0.7
                reasons.append(f"Price broke below {BREAKOUT_LOOKBACK}-bar low (down breakout) [-0.7]")

            # Feature: RSI guard
            if not pd.isna(rsi):
                if rsi > 70:
                    score -= 0.5
                    reasons.append(f"RSI {rsi:.1f} (overbought) [-0.5]")
                elif rsi < 30:
                    score += 0.4
                    reasons.append(f"RSI {rsi:.1f} (oversold) [+0.4]")
                else:
                    reasons.append(f"RSI {rsi:.1f} (neutral) [+0]")

            # Feature: AI weighting (probabilistic)
            if ai_trend == "BULLISH" or ai_trend == "Up":
                ai_weight = (ai_conf / 100.0) * 1.0  # up to +1.0
                score += ai_weight
                reasons.append(f"AI bullish {ai_conf:.1f}% [+{ai_weight:.2f}]")
            elif ai_trend == "BEARISH" or ai_trend == "Down":
                ai_weight = (ai_conf / 100.0) * 1.0
                score -= ai_weight
                reasons.append(f"AI bearish {ai_conf:.1f}% [-{ai_weight:.2f}]")
            else:
                reasons.append("AI neutral or unavailable [+0]")

            # Normalize / clamp score for readability
            score = float(score)
            reasons.append(f"Composite score = {score:.2f}")

            # --- Decision thresholds (tunable) ---
            # These thresholds are conservative by default
            BUY_THRESHOLD = 2.0
            STRONG_BUY_THRESHOLD = 3.0
            SELL_THRESHOLD = -1.5

            action = "HOLD"
            if score >= STRONG_BUY_THRESHOLD:
                action = "STRONG BUY"
            elif score >= BUY_THRESHOLD:
                action = "BUY"
            elif score <= SELL_THRESHOLD:
                action = "SELL"
            else:
                action = "HOLD"

            # --- ATR-based SL/TP (risk management) ---
            if not pd.isna(atr) and atr > 0:
                sl = latest["close"] - 1.5 * atr if action in ("BUY", "STRONG BUY") else latest["close"] + 1.5 * atr
                tp = latest["close"] + 3.0 * atr if action in ("BUY", "STRONG BUY") else latest["close"] - 3.0 * atr
                rr = (tp - latest["close"]) / (latest["close"] - sl) if (latest["close"] - sl) != 0 else float("inf")
                reasons.append(f"SL ${sl:.2f} | TP ${tp:.2f} | R:R {rr:.2f}")
            else:
                sl = tp = rr = None
                reasons.append("ATR not available — no SL/TP calculated")

            # --- Compose human-friendly message ---
            technical_summary = (
                f"• Close ${latest['close']:.2f} | SMA20 {sma20:.2f} / SMA50 {sma50:.2f}\n"
                f"• RSI {rsi:.1f} | ATR {atr:.2f} | MACD {macd_latest:.3f}/{macd_signal_latest:.3f} | ADX {adx:.1f if not pd.isna(adx) else 'n/a'}\n"
                f"• Volume {latest['volume']:,} (MA20 {vol_ma:,.0f}) | ROC {roc:.2f}%"
            )

            ai_line = f"• AI Forecast: {ai_trend or 'N/A'} ({ai_conf:.1f}% confidence)" if forecast_result else "• AI Forecast: N/A"

            # Build reason text with bullets
            reasons_text = "\n".join(f"→ {r}" for r in reasons)

            ticker_msg = (
                f"📊 *{ticker}*\n"
                f"{technical_summary}\n"
                f"{ai_line}\n\n"
                f"🔔 *Decision: {action}*  (score {score:.2f})\n\n"
                f"🧠 *Why:*\n{reasons_text}\n"
            )

            # Optionally include SL/TP in top line if present
            if sl and tp:
                ticker_msg += f"\n⛑️ SL: ${sl:.2f} | 🎯 TP: ${tp:.2f} | R:R: {rr:.2f}\n"

            # Append & log
            summary_messages.append(ticker_msg)
            log_message(f"{ticker} decision={action} score={score:.2f} | {', '.join(reasons[:3])}...")

        except Exception as e:
            log_message(f"❌ Error processing {ticker}: {e}")
            continue

    # --- Send consolidated message (Telegram + Email optional) ---
    if summary_messages:
        header = f"📈 *Hybrid AI+Technical Live Scan — {now.strftime('%b %d %H:%M')}*\n\n"
        msg = header + "\n\n".join(summary_messages)
        send_message(msg)
        try:
            # optional email too (only if configured)
            if EMAIL_SMTP and EMAIL_FROM and EMAIL_TO:
                send_email(f"Live Scan — {now.strftime('%Y-%m-%d %H:%M')}", msg)
        except Exception as e:
            log_message(f"⚠️ Email send failed: {e}")
        log_message("📩 Sent consolidated summary to Telegram (and email if configured)")
    else:
        log_message("ℹ️ No actionable signals this run.")

    # --- Auto Position Management (close losers etc.) ---
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

# ================= ENTRY POINT =================
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "analysis"
    log_message(f"Starting script in mode: {mode}")
    if mode == "analysis":
        previous_day_analysis()
    elif mode == "live":
        live_trading_loop()
    elif mode == "morning":
        send_top_movers_to_telegram(top_n=15)
        morning_scan()
    else:
        log_message(f"Unknown mode: {mode}")
