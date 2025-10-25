import os
import time
import pandas as pd
import numpy as np
import requests
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import sys
import ta
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange

# ================= CONFIG =================
ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GOOGLE_CREDS_FILE = "google_creds.json"

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

def log_message(msg):
    log_file = os.path.join(LOG_DIR, f"trade_signals_{datetime.now().strftime('%Y-%m-%d')}.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(f"[{timestamp}] {msg}")

api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL, api_version="v2")
clock = api.get_clock()
LIVE_RUN = clock.is_open

if LIVE_RUN:
    log_message("🟢 Market is open — starting live trading monitoring.")
else:
    log_message("🔴 Market is closed — running previous-day analysis.")

# Google Sheets setup
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDS_FILE, scope)
client = gspread.authorize(creds)
sheet = client.open("Daily_stocks")
tickers_ws = sheet.worksheet("Tickers")
tickers = tickers_ws.col_values(1)

# ================= FILE STORAGE =================
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

def refresh_signal_file_daily():
    global sent_signals
    current_file = get_signal_file()
    if not os.path.exists(current_file):
        sent_signals = set()
        with open(current_file, "w") as f:
            f.write(f"# Signals for {datetime.now().strftime('%Y-%m-%d')}\n")

# ================= HELPERS =================
def send_message(msg):
    print(msg)
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                          data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
        except Exception as e:
            print(f"Telegram error: {e}")

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

def market_open():
    clock = api.get_clock()
    return clock.is_open

# ================= ANALYSIS =================
def analyze_ticker(ticker, bars):
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC").tz_convert("America/New_York")
    else:
        bars.index = bars.index.tz_convert("America/New_York")

    bars["SMA20"] = bars["close"].rolling(SMA_SHORT).mean()
    bars["SMA50"] = bars["close"].rolling(SMA_LONG).mean()
    bars["RSI"] = compute_rsi(bars["close"], 14)
    bars["ATR"] = compute_atr(bars[["high", "low", "close"]], ATR_PERIOD)
    bars["ADX"] = ta.trend.adx(bars["high"], bars["low"], bars["close"], window=14)
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

    for i in range(1, len(bars)):
        prev, curr = bars.iloc[i - 1], bars.iloc[i]
        atr_value = curr["ATR"]
        stop_distance = atr_value * ATR_MULTIPLIER
        volatility_pct = (atr_value / curr["close"]) * 100
        volume_spike = curr["volume"] > 1.5 * curr["AvgVol"]

        # ================= CONDITIONS =================
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
                log_message(msg)
                send_message(msg)
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
                log_message(msg)
                send_message(msg)
                sent_signals.add(signal_id)
                save_sent_signal(signal_id)

# ================= EXECUTION =================
def previous_day_analysis():
    today = datetime.now().date()
    log_message(f"📊 Starting previous-day analysis for {today}")

    for ticker in tickers:
        try:
            start_date = f"{today}T09:30:00-04:00"
            end_date = f"{today}T16:00:00-04:00"
            bars = api.get_bars(ticker, tradeapi.TimeFrame.Minute, start=start_date, end=end_date, feed="iex").df
            if not bars.empty:
                analyze_ticker(ticker, bars)
            else:
                log_message(f"No bars for {ticker}")
        except Exception as e:
            log_message(f"⚠️ Error analyzing {ticker}: {e}")

def live_trading_loop():
    log_message("🚀 Starting live trading monitoring...")
    while market_open():
        refresh_signal_file_daily()
        for ticker in tickers:
            try:
                bars = api.get_bars(ticker, tradeapi.TimeFrame.Minute, limit=SMA_LONG * 2, feed="iex").df
                if not bars.empty:
                    analyze_ticker(ticker, bars)
            except Exception as e:
                log_message(f"⚠️ Error analyzing {ticker}: {e}")
        time.sleep(CHECK_INTERVAL)
    log_message("⏰ Market closed. Live monitoring ended.")

def morning_scan():
    """
    Runs early-morning scan before market opens.
    Identifies potential breakout or crossover stocks for today's trading.
    """
    log_message("🌅 Starting Morning Market Scanner...")
    today = datetime.now().date()
    potential_stocks = []

    for ticker in tickers:
        try:
            # Pull last 2 days of hourly data (covers pre-market)
            bars = api.get_bars(ticker, tradeapi.TimeFrame.Hour, limit=48, feed="iex").df
            if bars.empty:
                continue

            bars["SMA20"] = bars["close"].rolling(20).mean()
            bars["SMA50"] = bars["close"].rolling(50).mean()
            bars["RSI"] = compute_rsi(bars["close"], 14)
            bars["ADX"] = ta.trend.adx(bars["high"], bars["low"], bars["close"], window=14)
            bars["ATR"] = compute_atr(bars[["high", "low", "close"]], ATR_PERIOD)
            bars["MACD"], bars["MACD_Signal"] = compute_macd(bars["close"])
            bars["AvgVol"] = bars["volume"].rolling(20).mean()

            curr = bars.iloc[-1]
            prev = bars.iloc[-2]

            # Calculate proximity to crossover
            sma_gap = abs((curr["SMA20"] - curr["SMA50"]) / curr["SMA50"]) * 100
            macd_gap = abs(curr["MACD"] - curr["MACD_Signal"])
            atr_pct = (curr["ATR"] / curr["close"]) * 100
            vol_ratio = curr["volume"] / curr["AvgVol"]

            # Evaluate readiness
            bullish_ready = (
                curr["SMA20"] > curr["SMA50"] * 0.99 and
                curr["RSI"] > 50 and
                macd_gap < 0.2 and
                curr["ADX"] > 20 and
                vol_ratio > 1.2 and
                0.5 < atr_pct < 3
            )

            bearish_ready = (
                curr["SMA20"] < curr["SMA50"] * 1.01 and
                curr["RSI"] < 50 and
                macd_gap < 0.2 and
                curr["ADX"] > 20 and
                vol_ratio > 1.2 and
                0.5 < atr_pct < 3
            )

            if bullish_ready or bearish_ready:
                direction = "BULLISH" if bullish_ready else "BEARISH"
                score = sum([
                    bullish_ready or bearish_ready,
                    curr["ADX"] > 25,
                    vol_ratio > 1.5,
                    macd_gap < 0.15
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

    if not potential_stocks:
        send_message("🌄 Morning scan complete — no potential setups found.")
        return

    df = pd.DataFrame(potential_stocks).sort_values("Score", ascending=False)
    top10 = df.head(10)

    # Save to Google Sheet
    try:
        ws = sheet.worksheet("Morning_Scanner")
        ws.clear()
        ws.update([top10.columns.values.tolist()] + top10.values.tolist())
        log_message("📊 Morning scanner results updated to Google Sheets.")
    except Exception as e:
        log_message(f"⚠️ Failed to update Morning_Scanner sheet: {e}")

    # Telegram Summary
    msg = "🌅 *Morning Scanner — Potential Stocks for Today*\n\n"
    for _, row in top10.iterrows():
        msg += (f"{row['Ticker']}: {row['Direction']} | Price: {row['Price']} | "
                f"RSI: {row['RSI']} | ADX: {row['ADX']} | "
                f"Vol x{row['VolSpike']} | SMA Gap: {row['SMA_Gap%']}%\n")
    send_message(msg)
    log_message("✅ Morning scan completed successfully.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "analysis"

    if mode == "morning":
        morning_scan()
    elif mode == "live":
        live_trading_loop(interval=5)
    elif mode == "analysis":
        previous_day_analysis()
    else:
        previous_day_analysis()


