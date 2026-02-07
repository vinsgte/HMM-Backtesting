import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_all(
    df,
    df_portfolio,
    mc_paths,
    mc_mean,
    mc_median,
    symbol
):
    # Calculate rolling volatility across multiple windows
    windows = [5, 10, 20, 60, 120] #days
    vol_matrix = []

    for w in windows:
        # Annualized volatility: rolling std * sqrt(252)
        vol = df['Returns'].rolling(w).std() * np.sqrt(252)
        vol_matrix.append(vol.values)

    vol_matrix = np.array(vol_matrix)


    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{}, {}, {"type": "surface"}],# Row 1: 2D, 2D, 3D
            [{"colspan": 2}, None, {}]# Row 2: Wide plot, empty, normal
        ],
        subplot_titles=(
            f"{symbol} – Price & HMM Signals",
            "Equity Curve (Normalisée)",
            "Historical Volatility 3D",
            "Capital Evolution (Trades)",
            "Monte Carlo Price Simulation"
        )
    )


    # GRAPH 1 : PRICE + SIGNALS
    # Price line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            name="Price",
            line=dict(width=2)
        ),
        row=1, col=1
    )
    # Buy signals (green triangles up)
    fig.add_trace(
        go.Scatter(
            x=df[df['Signal'] == 1].index,
            y=df[df['Signal'] == 1]['Close'],
            mode="markers",
            marker=dict(symbol="triangle-up", size=10, color="green"),
            name="Buy Signal"
        ),
        row=1, col=1
    )
    # Sell signals (red triangles down)
    fig.add_trace(
        go.Scatter(
            x=df[df['Signal'] == -1].index,
            y=df[df['Signal'] == -1]['Close'],
            mode="markers",
            marker=dict(symbol="triangle-down", size=10, color="red"),
            name="Sell Signal"
        ),
        row=1, col=1
    )
    # Axis labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(
        title_text=f"Price ({symbol.split('-')[-1] if '-' in symbol else 'Currency'})",
        row=1, col=1
    )

    # GRAPH 2 : EQUITY CURVES
    # Strategy equity curve
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Equity_Strategy'],
            name="HMM Strategy",
            line=dict(width=2)
        ),
        row=1, col=2
    )
    # Buy & Hold benchmark
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Equity_BuyHold'],
            name="Buy & Hold",
            line=dict(width=2, dash="dash")
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(
        title_text="Portfolio Value",
        row=1, col=2
    )


    # GRAPH 3 : VOLATILITÉ 3D
    fig.add_trace(
        go.Surface(
            z=vol_matrix,
            x=np.arange(len(df)),# Time axis
            y=windows, # Window sizes
            colorscale="Viridis",
            showscale=True,
            name="Volatility"
        ),
        row=1, col=3
    )

    fig.update_scenes(
        dict(
            xaxis_title="Time Index (Days)",
            yaxis_title="Rolling Window (Days)",
            zaxis_title="Annualized Volatility"
        )
    )

    # GRAPH 4 : PORTFOLIO (TRADES)
    # Shows portfolio value at each trade exit
    fig.add_trace(
        go.Scatter(
            x=df_portfolio.index,
            y=df_portfolio['Portfolio_Value'],
            name="Portfolio (Trades)",
            mode="lines+markers",
            line=dict(width=2, color="gold"),
            marker=dict(size=6)
        ),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Trade Exit Date", row=2, col=1)
    fig.update_yaxes(
        title_text="Portfolio Value",
        row=2, col=1
    )

    # GRAPH 5 : MONTE CARLO
    n_simulations = mc_paths.shape[0]
    n_days = mc_paths.shape[1] - 1
    # Plot all simulated paths (semi-transparent)
    for i in range(n_simulations):
        fig.add_trace(
            go.Scatter(
                x=np.arange(n_days + 1),
                y=mc_paths[i],
                mode="lines",
                line=dict(width=1, color="cyan"),
                opacity=0.3,
                showlegend=False
            ),
            row=2, col=3
        )
    # Mean trajectory (yellow, thick)
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_days + 1),
            y=mc_mean,
            mode="lines",
            line=dict(width=3, color="yellow"),
            name="Expected Price (Mean)"
        ),
        row=2, col=3
    )
    # Median trajectory (white, dashed)
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_days + 1),
            y=mc_median,
            mode="lines",
            line=dict(width=2, dash="dash", color="white"),
            name="Median Scenario"
        ),
        row=2, col=3
    )

    fig.update_xaxes(title_text="Time (Days)", row=2, col=3)
    fig.update_yaxes(title_text="Simulated Price", row=2, col=3)

    # LAYOUT FINAL
    fig.update_layout(
        template="plotly_dark",
        height=900,
        title=f"{symbol} – HMM Strategy Overview",
        legend=dict(
            orientation="h",# Horizontal legend
            y=-0.15,# Below plots
            x=0.5,
            xanchor="center"
        )
    )

    fig.show()
