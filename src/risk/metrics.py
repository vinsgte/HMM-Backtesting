import numpy as np

def sharpe_ratio(returns, periods=252):
    return np.sqrt(periods) * returns.mean() / returns.std()
