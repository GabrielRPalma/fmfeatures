import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
import numpy as np


def fit_discrete_hmm(state_series):
    """Extrate state' statistics with Hidden Markov Models (HMM)

    Fits a Discrete Hidden Markov Model to
    a time series of states that represent
    trader's decisions of (i) buying, (ii)
    selling, or (iii) holding.

    Parameters
    ----------
    state_series : list
                A list of states (i) buying, (ii)
                selling, or (iii) holding for a
                given financial time series.

    Returns
    -------
    type
        tuple:
            Transition matrix and Emission matrix
            as pandas DataFrames.

    Examples
    --------
    # Example state series
    states = ['Buy', 'Buy', 'Sell', 'Buy',
              'Buy', 'Sell', 'Sell', 'Hold',
              'Hold']

    # Fit the HMM and get the transition and emission matrices
    transition_matrix, emission_matrix = fit_discrete_hmm(states)

    print("Transition Matrix:")
    print(transition_matrix)

    print("\nEmission Matrix:")
    print(emission_matrix)
    """
    # Encode the states as integers
    encoder = LabelEncoder()
    encoded_states = encoder.fit_transform(state_series)
    X = encoded_states.reshape(-1, 1)

    # Define the number of hidden states
    n_hidden = len(encoder.classes_)

    # Initialize and fit the HMM
    model = hmm.MultinomialHMM(n_components=n_hidden,
                               random_state=42, n_iter=100)
    model.fit(X)

    # Extract the transition matrix
    transition_matrix = model.transmat_

    # Extract emission probabilities
    emission_matrix = model.emissionprob_

    # Ensure the emission matrix has the correct number of symbols
    n_symbols = len(encoder.classes_)
    if emission_matrix.shape[1] < n_symbols:
        # Pad the emission_matrix with zeros for missing symbols
        emission_matrix_padded = np.zeros((n_hidden, n_symbols))
        emission_matrix_padded[:, :emission_matrix.shape[1]] = emission_matrix
        emission_matrix = emission_matrix_padded

    # Create a DataFrame for the transition matrix
    transition_df = pd.DataFrame(
        transition_matrix,
        index=encoder.classes_,
        columns=encoder.classes_
    )

    # Create a DataFrame for the emission probabilities
    emission_df = pd.DataFrame(
        emission_matrix,
        index=[f'Hidden_State_{i}' for i in range(n_hidden)],
        columns=encoder.classes_
    )

    return transition_df, emission_df
