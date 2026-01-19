import numpy as np

def monte_carlo_simulation(prices, n_simulations=50, n_days=252):
    last_price = prices.iloc[-1]
    returns = prices.pct_change().dropna()

    mu = returns.mean()
    sigma = returns.std()

    paths = []
    for _ in range(n_simulations):
        path = [last_price]
        for _ in range(n_days):
            shock = np.random.normal(mu, sigma)
            path.append(path[-1] * (1 + shock))
        paths.append(path)

    paths = np.array(paths)
    return paths, paths.mean(axis=0), np.median(paths, axis=0)
