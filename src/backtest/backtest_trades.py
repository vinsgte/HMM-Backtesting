import pandas as pd

def run_trade_backtest(df, initial_capital, fee_rate):
    capital = initial_capital
    portfolio_curve, portfolio_index = [], []

    entry_price = None
    entry_position = 0

    for i in range(1, len(df)):
        signal = df['Signal'].iloc[i]
        price = df['Close'].iloc[i]
        date = df.index[i]

        if entry_position == 0 and signal != 0:
            entry_price = price
            entry_position = signal

        elif entry_position != 0 and signal != entry_position:
            trade_return = entry_position * ((price - entry_price) / entry_price)
            capital *= (1 + trade_return - fee_rate)

            portfolio_curve.append(capital)
            portfolio_index.append(date)

            entry_price = None
            entry_position = 0

    return pd.DataFrame(
        {'Portfolio_Value': portfolio_curve},
        index=portfolio_index
    )
