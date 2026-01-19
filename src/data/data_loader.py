import yfinance as yf
from datetime import datetime, timedelta

def fetch_data(symbol, days=3650, interval='1d'):
    end = datetime.now()
    start = end - timedelta(days=days)
    return yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False)
