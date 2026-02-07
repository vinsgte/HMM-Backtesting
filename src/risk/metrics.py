import numpy as np

# Annualized Sharpe = sqrt(periods) * mean(returns) / std(returns)
# Higher values indicate better risk-adjusted performance
def sharpe_ratio(returns, periods=252):
    return np.sqrt(periods) * returns.mean() / returns.std()
