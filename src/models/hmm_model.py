import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

def fit_hmm(roc, n_states=3):
    # Normalize ROC values (mean=0, std=1) for better HMM convergence
    scaler = StandardScaler()
    roc_scaled = scaler.fit_transform(roc.values.reshape(-1, 1))
    # Configure and train Gaussian HMM
    model = hmm.GaussianHMM(
        n_components=n_states, # Number of hidden regimes
        covariance_type="full", # Full covariance matrix
        n_iter=100, # Maximum training iterations
        random_state=42 # Reproducibility
    )
    model.fit(roc_scaled)
    # Predict the most likely regime for each time step
    states = model.predict(roc_scaled)
    # Return as Series with original index
    return pd.Series(states, index=roc.index)
