import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots


# ===============================
# 1️⃣ DATA
# ===============================
def fetch_data(symbol, days=3650, interval='1d'):
    end = datetime.now()
    start = end - timedelta(days=days)
    return yf.download(symbol, start=start, end=end, interval=interval)

# ===============================
# 2️⃣ ROC
# ===============================
def calculate_roc(data, window=12):
    return data['Close'].pct_change(window)

# ===============================
# 3️⃣ HMM
# ===============================
def fit_hmm(roc, n_states=3):
    scaler = StandardScaler()
    roc_scaled = scaler.fit_transform(roc.values.reshape(-1, 1))

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=100,
        random_state=42
    )
    model.fit(roc_scaled)
    states = model.predict(roc_scaled)

    return pd.Series(states, index=roc.index)

# ===============================
# 4️⃣ SIGNALS
# ===============================
def generate_signals(roc, states):
    means = roc.groupby(states).mean().iloc[:, 0].sort_values()

    bear = means.index[0]
    bull = means.index[-1]

    signal = pd.Series(0, index=states.index)
    signal[states == bull] = 1
    signal[states == bear] = -1

    return signal


# ===============================
# 5️⃣ SHARPE
# ===============================
def sharpe_ratio(returns, periods=252):
    return np.sqrt(periods) * returns.mean() / returns.std()

# ===============================
# 6️⃣ MAIN
# ===============================
def main():
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
        
    # ===============================
    # BACKTEST
    # ===============================
    df['Returns'] = df['Close'].pct_change()
    df['Position'] = df['Signal'].shift(1).fillna(0)
    df['Strategy_Returns'] = df['Position'] * df['Returns']

    df['Equity_Strategy'] = (1 + df['Strategy_Returns']).cumprod()
    df['Equity_BuyHold'] = (1 + df['Returns']).cumprod()

    sharpe_strat = sharpe_ratio(df['Strategy_Returns'].dropna())
    sharpe_bh = sharpe_ratio(df['Returns'].dropna())

    print("\n===== BACKTEST RESULTS =====")
    print(f"Sharpe Strategy : {sharpe_strat:.2f}")
    print(f"Sharpe Buy & Hold : {sharpe_bh:.2f}")
    print(f"Return Strategy : {(df['Equity_Strategy'].iloc[-1]-1)*100:.2f}%")
    print(f"Return Buy & Hold : {(df['Equity_BuyHold'].iloc[-1]-1)*100:.2f}%")

    # ===============================
    # GRAPHS CÔTE À CÔTE
    # ===============================
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"{symbol} – Price & HMM Signals",
            "Equity Curve"
        )
    )

    # -------- GRAPH 1 : PRICE + SIGNALS --------
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name='Price'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df[df['Signal'] == 1].index,
            y=df[df['Signal'] == 1]['Close'],
            mode='markers',
            marker=dict(symbol='triangle-up', color='green', size=10),
            name='Buy'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df[df['Signal'] == -1].index,
            y=df[df['Signal'] == -1]['Close'],
            mode='markers',
            marker=dict(symbol='triangle-down', color='red', size=10),
            name='Sell'
        ),
        row=1, col=1
    )

    # -------- GRAPH 2 : EQUITY CURVE --------
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Equity_Strategy'],
            name='HMM Strategy'
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Equity_BuyHold'],
            name='Buy & Hold',
            line=dict(dash='dash')
        ),
        row=1, col=2
    )

    # -------- LAYOUT --------
    fig.update_layout(
        template="plotly_dark",
        height=600,
        title_text=f"{symbol} – HMM Strategy Overview",
        legend=dict(orientation="h", y=-0.15)
    )

    fig.show()


if __name__ == "__main__":
    main()
