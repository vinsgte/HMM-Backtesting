import pandas as pd

def generate_signals(roc: pd.Series, states: pd.Series) -> pd.Series:
    #Generate trading signals based on HMM regimes.
    # Ensure both series have exactly the same index
    # Uses inner join to keep only matching timestamps
    roc, states = roc.align(states, join="inner", axis=0)

    # In case roc accidentally comes as a DataFrame, extract the first column
    # This prevents downstream errors in groupby operations
    if isinstance(roc, pd.DataFrame):
        roc = roc.iloc[:, 0]

    # Calculate the average ROC for each regime → Series
    # This helps identify which regimes are bullish (high ROC) vs bearish (low ROC)
    regime_means = roc.groupby(states).mean()

    # Lowest mean = bearish regime, highest mean = bullish regime
    regime_means = regime_means.sort_values()

    bear_regime = regime_means.index[0] # Identify the bearish regime (lowest average ROC)
    bull_regime = regime_means.index[-1] # Identify the bullish regime (highest average ROC)
    signal = pd.Series(0, index=roc.index) # Initialize all signals to 0 (neutral/hold position)
    signal.loc[states == bull_regime] = 1 # Set signal to 1 (long/buy) when in bullish regime
    signal.loc[states == bear_regime] = -1 # Set signal to -1 (short/sell) when in bearish regime

    return signal
