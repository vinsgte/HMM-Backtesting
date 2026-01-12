import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

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
    # ROC = Rate of Change
    # Unité : pourcentage (ratio)
    # Pourquoi : permet de comparer les variations indépendamment du prix absolu
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
    # -------- Paramètres --------
    initial_capital = 10_000
    fee_rate = 0.001  # 0.1% par trade
    symbol = input("Choisir le symbol (ex: BTC-USD, EURUSD=X) : ")

    # -------- Data --------
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
    # BACKTEST JOURNALIER
    # ===============================
    df['Returns'] = df['Close'].pct_change()
    df['Position'] = df['Signal'].shift(1).fillna(0)
    df['Trade'] = df['Position'].diff().abs()
    df['Fees'] = df['Trade'] * fee_rate
    df['Strategy_Returns'] = (df['Position'] * df['Returns']) - df['Fees']

    df['Equity_Strategy'] = initial_capital * (1 + df['Strategy_Returns']).cumprod()
    df['Equity_BuyHold'] = initial_capital * (1 + df['Returns']).cumprod()

    sharpe_strat = sharpe_ratio(df['Strategy_Returns'].dropna())
    sharpe_bh = sharpe_ratio(df['Returns'].dropna())

    print("\n===== BACKTEST RESULTS =====")
    print(f"Capital initial : {initial_capital:.0f}")
    print(f"Capital final stratégie : {df['Equity_Strategy'].iloc[-1]:.2f}")
    print(f"Capital final Buy & Hold : {df['Equity_BuyHold'].iloc[-1]:.2f}")
    print(f"Sharpe Strategy : {sharpe_strat:.2f}")
    print(f"Sharpe Buy & Hold : {sharpe_bh:.2f}")

    # ===============================
    # BACKTEST PAR TRADES (Portfolio Value)
    # ===============================
    capital = initial_capital
    portfolio_curve = []
    portfolio_index = []
    entry_price = None
    entry_position = 0

    for i in range(1, len(df)):
        signal = df['Signal'].iloc[i]
        price = df['Close'].iloc[i]
        date = df.index[i]

        # Entrée en position
        if entry_position == 0 and signal != 0:
            entry_price = price
            entry_position = signal

        # Sortie de position
        elif entry_position != 0 and signal != entry_position:
            trade_return = entry_position * ((price - entry_price) / entry_price)
            capital *= (1 + trade_return - fee_rate)
            portfolio_curve.append(capital)
            portfolio_index.append(date)
            entry_price = None
            entry_position = 0

    df_portfolio = pd.DataFrame(
        {'Portfolio_Value': portfolio_curve},
        index=portfolio_index
    )

    # ===============================
    # VOLATILITÉ HISTORIQUE 3D
    # ===============================
    windows = [5, 10, 20, 60, 120]
    vol_matrix = []
    for w in windows:
        # Volatilité annualisée
        # Unité : % par an
        # Pourquoi : standard en finance pour comparer des actifs sur différentes périodes
        vol = df['Returns'].rolling(w).std() * np.sqrt(252)
        vol_matrix.append(vol.values)
    vol_matrix = np.array(vol_matrix)

    # ===============================
    # MONTE CARLO PRICE SIMULATION
    # ===============================
    n_simulations = 50
    n_days = 252
    last_price = df['Close'].iloc[-1]
    mu = df['Returns'].mean()      # Rendement moyen journalier (ratio)
    sigma = df['Returns'].std()   # Volatilité journalière (ratio)

    # Pourquoi :
    # mu et sigma sont adimensionnels car basés sur des rendements
    # Le prix simulé reste dans l'unité monétaire du prix initial


    mc_paths = []
    for _ in range(n_simulations):
        prices = [last_price]
        for _ in range(n_days):
            shock = np.random.normal(loc=mu, scale=sigma)
            price_next = prices[-1] * (1 + shock)
            prices.append(price_next)
        mc_paths.append(prices)
    mc_paths = np.array(mc_paths)

    # ===============================
    # MONTE CARLO – TRAJECTOIRE MOYENNE
    # ===============================
    mc_mean = mc_paths.mean(axis=0)
    mc_median = np.median(mc_paths, axis=0)

    # ===============================
    # GRAPHIQUES
    # ===============================
    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{}, {}, {"type": "surface"}],  # ligne 1 : vol 3D
            [{"colspan": 2}, None, {}]      # ligne 2 : portfolio colspan 2, MC dans col3
        ],
        subplot_titles=(
            f"{symbol} – Price & HMM Signals",
            "Equity Curve (Normalisée)",
            "Volatilité 3D",
            "Évolution du portefeuille (par trades)",
            "Monte Carlo Price Simulation"
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

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(
        title_text=f"Price ({symbol.split('-')[-1] if '-' in symbol else 'Currency'})",
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

    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(
        title_text="Portfolio Value (Monetary Units)",
        row=1, col=2
    )


    # -------- GRAPH 3 : VOLATILITÉ 3D --------
    fig.add_trace(
        go.Surface(
            z=vol_matrix,
            x=np.arange(len(df)),
            y=windows,
            colorscale='Viridis',
            name='Volatilité'
        ),
        row=1, col=3
    )

    fig.update_scenes(
        dict(
            xaxis_title="Time Index (Days)",
            yaxis_title="Rolling Window (Days)",
            zaxis_title="Annualized Volatility (%)"
        )
    )




    # -------- GRAPH 4 : PORTEFEUILLE (TRADES) --------
    fig.add_trace(
        go.Scatter(
            x=df_portfolio.index,
            y=df_portfolio['Portfolio_Value'],
            name='Portfolio (Trades)',
            mode='lines+markers',
            line=dict(color='gold', width=2),
            marker=dict(size=6)
        ),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Trade Exit Date", row=2, col=1)
    fig.update_yaxes(
        title_text="Portfolio Value (Monetary Units)",
        row=2, col=1
    )


    # -------- GRAPH 5 : MONTE CARLO --------
    for i in range(n_simulations):
        fig.add_trace(
            go.Scatter(
                y=mc_paths[i],
                x=np.arange(n_days + 1),
                mode='lines',
                line=dict(width=1, color='cyan'),
                opacity=0.4,
                showlegend=False
            ),
            row=2, col=3
        )

    # Courbe moyenne (espérance)
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_days + 1),
            y=mc_mean,
            mode='lines',
            line=dict(color='yellow', width=3),
            name='Expected Price (Mean)'
        ),
        row=2, col=3
    )

    # Courbe médiane (scénario le plus probable)
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_days + 1),
            y=mc_median,
            mode='lines',
            line=dict(color='white', width=2, dash='dash'),
            name='Median Scenario'
        ),
        row=2, col=3
    )

    
    fig.update_xaxes(title_text="Time (Days)", row=2, col=3)
    fig.update_yaxes(
        title_text="Simulated Price (Monetary Units)",

    )

    # -------- LAYOUT --------
    fig.update_layout(
        template="plotly_dark",
        height=900,
        title_text=f"{symbol} – HMM Strategy Overview",
        legend=dict(orientation="h", y=-0.15)
    )


    fig.show()

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    main()





