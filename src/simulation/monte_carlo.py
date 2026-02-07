import numpy as np

def monte_carlo_simulation(prices, n_simulations=50, n_days=252):
    # Starting point: last known price
    last_price = prices.iloc[-1]

    # Calculate historical return statistics
    returns = prices.pct_change().dropna()
    mu = returns.mean()# Mean daily return
    sigma = returns.std()# Volatility

    # Generate multiple random price paths
    paths = []
    for _ in range(n_simulations):
        path = [last_price]
        for _ in range(n_days):
            # Random shock based on historical distribution
            shock = np.random.normal(mu, sigma)
            # Compound the price: P_next = P_current * (1 + return)
            path.append(path[-1] * (1 + shock))
        paths.append(path)

    # Convert to numpy array for efficient calculations
    paths = np.array(paths)

    # Return all paths plus statistical aggregates
    return paths, paths.mean(axis=0), np.median(paths, axis=0)
