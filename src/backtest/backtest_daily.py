def run_daily_backtest(df, initial_capital, fee_rate):
    df['Returns'] = df['Close'].pct_change()
    df['Position'] = df['Signal'].shift(1).fillna(0)
    df['Trade'] = df['Position'].diff().abs()
    df['Fees'] = df['Trade'] * fee_rate

    df['Strategy_Returns'] = (df['Position'] * df['Returns']) - df['Fees']
    df['Equity_Strategy'] = initial_capital * (1 + df['Strategy_Returns']).cumprod()
    df['Equity_BuyHold'] = initial_capital * (1 + df['Returns']).cumprod()

    return df
