import os
import pandas as pd
import numpy as np
import requests
import alpaca_trade_api as tradeapi
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time

# ================= CONFIG =================
# Read keys and settings from environment variables (GitHub Actions safe)
ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GOOGLE_CREDS_FILE = "service_account.json"

# Trading strategy parameters
SMA_SHORT = 20
SMA_LONG = 50
ATR_PERIOD = 14
RISK_PER_TRADE = 0.01
ATR_MULTIPLIER = 1

# Dry-run mode: True = simulate trades, False = place live orders
DRY_RUN = os.getenv("DRY_RUN", "True").lower() == "true"

# ================= INIT =================
print("Initializing Alpaca API and Google Sheets...")
api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL, api_version='v2')

# Google Sheets setup
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDS_FILE, scope)
client = gspread.authorize(creds)

# Connect to Google Sheet
sheet = client.open("Daily_stocks")
tickers_ws = sheet.worksheet("Tickers")
positions_ws = sheet.worksheet("Positions")
equity_ws = sheet.worksheet("Equity")

tickers = tickers_ws.col_values(1)
send_message = lambda msg: print(msg) or (
    TELEGRAM_TOKEN and TELEGRAM_CHAT_ID and requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    )
)
send_message(f"✅ Bot started. Loaded {len(tickers)} tickers: {tickers}")

# ================= HELPERS =================
def compute_atr(df, period=14):
    df['H-L'] = df['High'] - df['Low']
    df['H-Cp'] = abs(df['High'] - df['Close'].shift(1))
    df['L-Cp'] = abs(df['Low'] - df['Close'].shift(1))
    tr = df[['H-L', 'H-Cp', 'L-Cp']].max(axis=1)
    return tr.rolling(period).mean()

def compute_shares_by_atr(equity, atr_value, price):
    stop_distance = atr_value * ATR_MULTIPLIER
    if stop_distance <= 0 or np.isnan(stop_distance):
        return 0
    dollar_risk = equity * RISK_PER_TRADE
    return max(int(dollar_risk / stop_distance), 0)

def market_open():
    clock = api.get_clock()
    return clock.is_open

# ================= LOAD ACCOUNT =================
account = api.get_account()
cash = float(account.cash)
positions_list = api.list_positions()
positions_dict = {p.symbol.upper(): int(p.qty) for p in positions_list}
avg_price_dict = {p.symbol.upper(): float(p.avg_entry_price) for p in positions_list}

# Prepare positions sheet
positions_df = pd.DataFrame(positions_ws.get_all_records())
if positions_df.empty or len(positions_df) != len(tickers):
    positions_df = pd.DataFrame({
        'Ticker': tickers,
        'Shares': [positions_dict.get(t, 0) for t in tickers],
        'AvgPrice': [avg_price_dict.get(t, 0) for t in tickers],
        'CurrentPrice': [0]*len(tickers),
        'P&L': [0]*len(tickers)
    })

# ================= MAIN LOOP =================
if not market_open():
    send_message("❌ Market is closed. Exiting bot.")
    exit()

total_unrealized = 0
positions_updates = []

for ticker in tickers:
    ticker = ticker.strip().upper()
    try:
        bars = api.get_bars(ticker, tradeapi.TimeFrame.Minute, limit=60).df.tail(60)
        if len(bars) < SMA_LONG:
            send_message(f"Skipping {ticker}: not enough data.")
            continue

        close = bars['close']
        high = bars['high']
        low = bars['low']

        sma_short = close.rolling(SMA_SHORT).mean().iloc[-1]
        sma_long = close.rolling(SMA_LONG).mean().iloc[-1]
        atr = compute_atr(pd.DataFrame({'High': high, 'Low': low, 'Close': close}), ATR_PERIOD).iloc[-1]
        current_price = close.iloc[-1]

        current_shares = positions_dict.get(ticker, 0)
        current_avg_price = avg_price_dict.get(ticker, 0)

        # BUY
        if sma_short > sma_long and current_shares == 0:
            shares_to_trade = compute_shares_by_atr(cash, atr, current_price)
            if shares_to_trade > 0:
                if DRY_RUN:
                    send_message(f"[DRY-RUN] BUY {shares_to_trade} {ticker} at ${current_price:.2f}")
                else:
                    api.submit_order(symbol=ticker, qty=shares_to_trade, side='buy', type='market', time_in_force='day')
                    cash -= shares_to_trade * current_price
                    send_message(f"[LIVE] ✅ BUY {shares_to_trade} {ticker} at ${current_price:.2f}")
                positions_dict[ticker] = shares_to_trade
                avg_price_dict[ticker] = current_price

        # SELL
        elif sma_short < sma_long and current_shares > 0:
            shares_to_trade = current_shares
            if DRY_RUN:
                send_message(f"[DRY-RUN] SELL {shares_to_trade} {ticker} at ${current_price:.2f}")
            else:
                api.submit_order(symbol=ticker, qty=shares_to_trade, side='sell', type='market', time_in_force='day')
                cash += shares_to_trade * current_price
                send_message(f"[LIVE] ❌ SELL {shares_to_trade} {ticker} at ${current_price:.2f}")
            positions_dict[ticker] = 0
            avg_price_dict[ticker] = 0

        # Update P&L
        unrealized = positions_dict[ticker] * (current_price - avg_price_dict[ticker])
        total_unrealized += unrealized
        positions_updates.append([positions_dict[ticker], avg_price_dict[ticker], current_price, unrealized])

    except Exception as e:
        send_message(f"⚠️ Error processing {ticker}: {e}")
        continue

# ================= UPDATE GOOGLE SHEETS =================
try:
    positions_ws.update(positions_updates, range_name=f'B2:E{len(tickers)+1}')
    equity_ws.update('A2', [[cash]])
    equity_ws.update('B2', [[cash + total_unrealized]])
    send_message(f"✅ Bot completed successfully. Total Equity: ${cash + total_unrealized:.2f}")
except Exception as e:
    send_message(f"⚠️ Google Sheets update failed: {e}")
