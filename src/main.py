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

def main():
    initial_capital = 10_000
    fee_rate = 0.001
    symbol = input("Choisir le symbol (ex: BTC-USD, EURUSD=X) : ")

    data = fetch_data(symbol)
    roc = calculate_roc(data).dropna()
    states = fit_hmm(roc)
    signals = generate_signals(roc, states)

    df = pd.DataFrame(index=roc.index)
    df['Close'] = data.loc[roc.index, 'Close']
    df['ROC'] = roc
    df['Regime'] = states
    df['Signal'] = signals

    df = run_daily_backtest(df, initial_capital, fee_rate)
    df_trades = run_trade_backtest(df, initial_capital, fee_rate)

    sharpe_strat = sharpe_ratio(df['Strategy_Returns'].dropna())
    sharpe_bh = sharpe_ratio(df['Returns'].dropna())

    print("\n===== BACKTEST RESULTS =====")
    print(f"Capital final stratégie : {df['Equity_Strategy'].iloc[-1]:.2f}")
    print(f"Capital final Buy & Hold : {df['Equity_BuyHold'].iloc[-1]:.2f}")
    print(f"Sharpe Strategy : {sharpe_strat:.2f}")
    print(f"Sharpe Buy & Hold : {sharpe_bh:.2f}")

    mc_paths, mc_mean, mc_median = monte_carlo_simulation(df['Close'])

    plot_all(df, df_trades, mc_paths, mc_mean, mc_median, symbol)

if __name__ == "__main__":
    main()
