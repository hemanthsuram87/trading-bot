# 📈 Python Stock Trading Bot with Telegram Alerts

An automated trading signal bot that calculates **20-day** and **200-day moving averages** for selected stocks and sends **Buy/Sell alerts** via **Telegram** — scheduled to run automatically during U.S. market hours.

---

## 🚀 Features

✅ Fetches live stock data using [yfinance](https://pypi.org/project/yfinance/)  
✅ Calculates **20-day** and **200-day** moving averages  
✅ Generates **Buy/Sell** signals based on crossover strategy  
✅ Sends instant alerts to **Telegram** via Bot API  
✅ Runs automatically every **10 minutes** between **9:30 AM–4:00 PM (ET)** using **GitHub Actions**  
✅ Logs all actions and signals with timestamps  

---

## 🧰 Requirements

- Python **3.9+**
- A [Telegram Bot](https://core.telegram.org/bots#6-botfather) (via [@BotFather](https://t.me/botfather))
- A **GitHub account** for cloud scheduling
- Required Python packages:

```bash
pip install pandas yfinance requests schedule
