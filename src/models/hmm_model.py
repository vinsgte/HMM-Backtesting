import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

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
