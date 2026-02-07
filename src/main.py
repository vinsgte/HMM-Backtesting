# IMPORTS
import pandas as pd

from data.data_loader import fetch_data
from features.indicators import calculate_roc
from models.hmm_model import fit_hmm
from strategy.signals import generate_signals
from backtest.backtest_daily import run_daily_backtest
from backtest.backtest_trades import run_trade_backtest
from risk.metrics import sharpe_ratio
from simulation.monte_carlo import monte_carlo_simulation
from visualization.plots import plot_all


# MAIN FUNCTION
def main():
    # INITIAL PARAMETERS
    initial_capital = 10_000
    fee_rate = 0.001
    # Prompt user to choose the symbol to analyze
    symbol = input("Choose symbol like you want (ex: BTC-USD, EURUSD=X) : ")
    # Download historical data for the chosen symbol
    data = fetch_data(symbol)
    # Calculate Rate of Change (ROC) and remove NaN values
    # ROC measures the percentage change in price over a given period
    roc = calculate_roc(data).dropna()
    # Use Hidden Markov Model to identify different regimes
    states = fit_hmm(roc)
    # Generate buy (1), sell (-1), or hold (0) signals
    # based on ROC and detected regimes
    signals = generate_signals(roc, states)

    # Create a DataFrame with all necessary information
    df = pd.DataFrame(index=roc.index)
    df['Close'] = data.loc[roc.index, 'Close']      # Closing price
    df['ROC'] = roc                                 # ROC indicator
    df['Regime'] = states                           # Regimes identified by HMM
    df['Signal'] = signals                          # Trading signals
    
    # Daily backtest: calculates capital evolution day by day
    # Compares strategy with Buy & Hold approach
    df = run_daily_backtest(df, initial_capital, fee_rate)
    # Trade-by-trade backtest: detailed analysis of each transaction
    df_trades = run_trade_backtest(df, initial_capital, fee_rate)

    # Calculate Sharpe ratio for the strategy
    # Measures risk-adjusted return
    sharpe_strat = sharpe_ratio(df['Strategy_Returns'].dropna())
    # Calculate Sharpe ratio for Buy & Hold
    sharpe_bh = sharpe_ratio(df['Returns'].dropna())

    # RESULTS DISPLAY
    print("BACKTEST RESULTS")
    print(f"final strategy capital : {df['Equity_Strategy'].iloc[-1]:.2f}")
    print(f"final Buy & Holdcapital : {df['Equity_BuyHold'].iloc[-1]:.2f}")
    print(f"Sharpe Strategy : {sharpe_strat:.2f}")
    print(f"Sharpe Buy & Hold : {sharpe_bh:.2f}")

    # MONTE CARLO SIMULATION
    # Generate possible future price trajectories based on historical data
    mc_paths, mc_mean, mc_median = monte_carlo_simulation(df['Close'])

    # VISUALIZATION
    # Generate all plots
    plot_all(df, df_trades, mc_paths, mc_mean, mc_median, symbol)


# STARTT SCRIPT
if __name__ == "__main__":
    main()

