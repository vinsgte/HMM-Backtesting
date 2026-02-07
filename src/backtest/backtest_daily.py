def run_daily_backtest(df, initial_capital, fee_rate):
    # Calculate percentage change in closing price from one day to the next
    # This represents the market's daily return
    df['Returns'] = df['Close'].pct_change()
    
    '''
    Shift signals by 1 day to avoid look-ahead bias
    Position taken today is based on yesterday's signal
    Fill NaN (first row) with 0 (no position initially)
    '''
    df['Position'] = df['Signal'].shift(1).fillna(0)

    '''
    Detect when position changes (trade occurs)
    diff() calculates change in position: 0→1, 1→-1, etc.
    abs() converts to binary: 0 (no trade) or non-zero (trade occurred)
    '''
    df['Trade'] = df['Position'].diff().abs()
    '''
    Calculate fees incurred on each trade
    '''
    df['Fees'] = df['Trade'] * fee_rate

    '''
    Calculate net strategy returns after fees
    Position * Returns: gain/loss from holding the position
    Fees: subtract transaction costs
    Example: If position=1 and Returns=2%, Strategy_Returns ≈ 2% - fees
             If position=0, Strategy_Returns = -fees (if traded) or 0
    '''
    df['Strategy_Returns'] = (df['Position'] * df['Returns']) - df['Fees']

    '''
    Calculate cumulative equity for the strategy
    (1 + Strategy_Returns).cumprod() compounds returns over time
    Multiply by initial_capital to get dollar value
    '''
    df['Equity_Strategy'] = initial_capital * (1 + df['Strategy_Returns']).cumprod()

    # Calculate cumulative equity for Buy & Hold
    # Same calculation but using raw market returns (no position or fees)
    df['Equity_BuyHold'] = initial_capital * (1 + df['Returns']).cumprod()

    return df
