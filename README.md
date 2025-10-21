GitHub Actions Live Trading Bot Workflow
          ┌───────────────────┐
          │ GitHub Repository │
          │ (Python Bot Code) │
          └────────┬──────────┘
                   │
                   │ Push / Schedule / Manual Run
                   ▼
          ┌─────────────────────────┐
          │ GitHub Actions Workflow │
          │ .github/workflows/...   │
          └────────┬────────────────┘
                   │
                   │ Sets up environment, Python, dependencies
                   ▼
          ┌─────────────────────────┐
          │ Python Bot Execution    │
          │ sma20_50_atr_backtest  │
          └────────┬────────────────┘
                   │
                   │ Reads environment variables:
                   │ - ALPACA_KEY / SECRET / BASE_URL
                   │ - TELEGRAM_TOKEN / CHAT_ID
                   │ - DRY_RUN flag
                   │ - GOOGLE_CREDS_JSON
                   ▼
          ┌─────────────────────────┐
          │ Alpaca API (Live/Paper)│
          │ - Fetch account info   │
          │ - Fetch positions      │
          │ - Fetch market data    │
          │ - Place buy/sell orders│
          └────────┬────────────────┘
                   │
                   │ Market data + signals processed
                   ▼
          ┌─────────────────────────┐
          │ Trading Logic & Signals │
          │ - SMA 20 / SMA 50      │
          │ - ATR sizing           │
          │ - Buy / Sell decision  │
          └────────┬────────────────┘
                   │
                   │ Trade executed or simulated
                   ▼
          ┌─────────────────────────┐
          │ Google Sheets Update    │
          │ - Tickers & positions   │
          │ - P&L, current price    │
          │ - Total equity          │
          └────────┬────────────────┘
                   │
                   │ Send alert
                   ▼
          ┌─────────────────────────┐
          │ Telegram Notification   │
          │ - Buy/Sell trades       │
          │ - Errors or info logs   │
          └─────────────────────────┘

Flow Summary

GitHub Actions triggers the bot (schedule or manual).

The workflow sets up Python and installs dependencies.

The bot reads API keys & environment variables securely.

It connects to Alpaca API to fetch account, positions, and market data.

It runs the trading logic (SMA + ATR strategy) and decides whether to buy/sell.

Trades are executed or simulated depending on DRY_RUN.

Positions, equity, and P&L are updated in Google Sheets.

Alerts and logs are sent via Telegram and Actions logs.
