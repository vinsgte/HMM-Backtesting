import pandas as pd

def generate_signals(roc: pd.Series, states: pd.Series) -> pd.Series:
    """
    Génère les signaux de trading à partir des régimes HMM
    """

    # 🔒 ALIGNEMENT STRICT DES INDEX
    roc, states = roc.align(states, join="inner", axis=0)

    # 🔒 SÉCURITÉ : forcer roc en Series
    if isinstance(roc, pd.DataFrame):
        roc = roc.iloc[:, 0]

    # Moyenne du ROC par régime → Series
    regime_means = roc.groupby(states).mean()

    # Tri du plus baissier au plus haussier
    regime_means = regime_means.sort_values()

    bear_regime = regime_means.index[0]
    bull_regime = regime_means.index[-1]

    # Création des signaux
    signal = pd.Series(0, index=roc.index)
    signal.loc[states == bull_regime] = 1
    signal.loc[states == bear_regime] = -1

    return signal
