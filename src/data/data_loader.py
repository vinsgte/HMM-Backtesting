import yfinance as yf
from datetime import datetime, timedelta

def fetch_data(symbol, days=3650, interval='1d'):
    # Set end date to current date and time
    end = datetime.now()
    # Calculate start date by subtracting the specified number of days
    # Default: 3650 days = approximately 10 years of data
    start = end - timedelta(days=days)
    # Download data from Yahoo Finance
    # auto_adjust=False: Returns raw prices without dividend/split adjustments
    return yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False)
