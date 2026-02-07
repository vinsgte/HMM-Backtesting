# Calculate percentage change from 'window' periods ago
# ROC = (Close_today - Close_window_ago) / Close_window_ago
def calculate_roc(data, window=12):
    return data['Close'].pct_change(window)
