#Created by Vincent GAUTHEREAU

import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta


#Data function

def data(symbol, days=3650, interval='1d'):
    end = datetime.now()
    start=end-timedelta(days=days)
    return yf.download(symbol, start=start, end=end, interval=interval)


#ROC function

def calculate_roc(data, window=12):
    return data['Close'].pct_change(window)

#HMM Function

def hmm(roc, n_states=3):
    scaler=StandardScaler()
    roc_scaled=scaler.fit_transform(roc.values.reshape(-1, 1))

    model=hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=100,
        random_state=42
    )
    model.fit(roc_scaled)
    states=model.predict(roc_scaled)

    return pd.Series(states, index=roc.index)

#Signals Function



