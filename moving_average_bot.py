import pandas as pd
import yfinance as yf
import schedule
from email.mime.text import MIMEText
import time
import datetime
import os
import requests
import pywhatkit
# ========================
#  Configuration
# ========================
GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRT4_URyoGfO4JMnHf7GJ3be442ItRj9zh--Cr904jmuwsybqna1cT6KauoaCryefurCnf18HBniFqh/pub?output=csv"
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, f"trading_log_{datetime.date.today()}.txt")

INTERVAL_MINUTES = 10  # Run every 10 minutes

# ========================
#  Logging helper
# ========================
def log_message(message):
    """Log messages to daily log file with timestamp"""
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {message}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    print(line.strip())

# ========================
#  Moving Average Strategy
# ========================
def moving_average_strategy(symbol):
    """Calculate 20-day & 200-day moving averages and generate signal."""
    try:
        data = yf.download(symbol, period="1y", progress=False)
        if data.empty:
            return "HOLD", f"{symbol}: No price data"

        # Calculate moving averages
        data["MA20"] = data["Close"].rolling(window=20).mean()
        data["MA200"] = data["Close"].rolling(window=200).mean()

        # Drop incomplete data
        data = data.dropna(subset=["MA20", "MA200"])
        if data.empty:
            return "HOLD", f"{symbol}: Not enough data for MA200"

        # Latest values (guaranteed to be scalar)
        latest = data.iloc[-1]
        ma20 = float(latest["MA20"])
        ma200 = float(latest["MA200"])

        # Trading signals
        if ma20 > ma200:
            return "BUY", f"{symbol}: 20-day MA above 200-day MA (Golden Cross)"
        elif ma20 < ma200:
            return "SELL", f"{symbol}: 20-day MA below 200-day MA (Death Cross)"
        else:
            return "HOLD", f"{symbol}: No crossover signal"

    except Exception as e:
        return "ERROR", f"{symbol}: {e}"

# ========================
#  SMS Notification
# Replace these with your actual values
BOT_TOKEN = "8475528816:AAFntgwGkp9jW5mVVnaX1MHGtM4kjPfnvC8"
CHAT_ID = "7862318105"


def send_message(message):
    """
    Sends a message to your Telegram chat using the Bot API.
    """
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message}
        response = requests.post(url, data=payload)
        
        if response.status_code == 200:
            print(" Message sent successfully!")
        else:
            print(f" Failed to send message: {response.text}")
    except Exception as e:
        print(f" Error sending message: {e}")


# ========================
#  Load stock symbols
# ========================
def load_stocks_from_sheet():
    """Fetch stock symbols from a published Google Sheet (CSV link)."""
    try:
        df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
        # Expecting stock symbols in first column
        stocks = [str(s).strip().upper() for s in df.iloc[:, 0].dropna()]
        log_message(f"Loaded {len(stocks)} symbols from Google Sheet.")
        return stocks
    except Exception as e:
        log_message(f" Failed to fetch Google Sheet: {e}")
        return []

# ========================
#  Main trading logic
# ========================
def run_trading_bot():
    """Run trading strategy for all stocks."""
    log_message(" Trading bot started.")
    stocks = load_stocks_from_sheet()
    if not stocks:
        log_message("No stocks to process — exiting.")
        return

    for symbol in stocks:
        signal, message = moving_average_strategy(symbol)
        log_message(message)
        


        if signal in ("BUY", "SELL"):
            send_message(f"{signal} ALERT for {symbol}: {message}")

    log_message(" Trading bot completed.\n")



def is_market_open():
    """Check if current time is within market hours (Mon-Fri, 9:30-16:00)"""
    now = datetime.datetime.now()
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    market_open = datetime.time(9, 30)
    market_close = datetime.time(16,00)
    return market_open <= now.time() <= market_close

# ========================
# Continuous scheduler
# ========================
def start_scheduler():
    log_message(" Scheduler started. Bot will run every 10 minutes during market hours.")
    send_message("Trading bot started and running.")
    while True:
        if is_market_open():
            run_trading_bot()
            log_message(f" Sleeping for {INTERVAL_MINUTES} minutes...")
            time.sleep(INTERVAL_MINUTES * 60)
        else:
            # Sleep until next market open
            now = datetime.datetime.now()
            if now.weekday() >= 5:  # Weekend
                days_until_monday = 7 - now.weekday()
                next_open = datetime.datetime.combine(
                    now.date() + datetime.timedelta(days=days_until_monday),
                    datetime.time(9, 30)
                )
            elif now.time() < datetime.time(9, 30):
                next_open = datetime.datetime.combine(now.date(), datetime.time(9, 30))
            else:  # After 16:00
                next_open = datetime.datetime.combine(now.date() + datetime.timedelta(days=1), datetime.time(9, 30))

            sleep_seconds = (next_open - now).total_seconds()
            log_message(f"Market closed. Sleeping until next open at {next_open}...")
            time.sleep(sleep_seconds)
# ========================
#  Entry Point
# ========================
if __name__ == "__main__":
    start_scheduler()
