def calculate_roc(data, window=12):
    return data['Close'].pct_change(window)
