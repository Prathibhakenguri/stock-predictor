# utils.py

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
import joblib
import os

def scale_features(df, feature_cols, scaler_path=None, scaler_type='minmax', save=True, load=False):
    """
    Scales feature columns using MinMaxScaler or StandardScaler.
    - If `load=True`, loads an existing scaler from `scaler_path`.
    - If `save=True`, saves the fitted scaler to `scaler_path`.

    Args:
        df (pd.DataFrame): Input data.
        feature_cols (list): Columns to scale.
        scaler_path (str): Path to save/load the scaler.
        scaler_type (str): 'minmax' (default) or 'standard'.
        save (bool): Whether to save fitted scaler.
        load (bool): Whether to load existing scaler instead of fitting.

    Returns:
        df_scaled (pd.DataFrame): DataFrame with scaled feature columns.
        scaler (MinMaxScaler or StandardScaler): The scaler object.
    """
    df = df.copy()
    df.dropna(subset=feature_cols, inplace=True)

    if load and scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
        scaler.fit(df[feature_cols])
        if save and scaler_path:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(scaler, scaler_path)

    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])

    return df_scaled, scaler


def create_lstm_sequences(df, feature_cols, sequence_length=60):
    """
    Generates LSTM sequences of fixed window size from the dataframe.

    Args:
        df (pd.DataFrame): Scaled DataFrame with required features.
        feature_cols (list): List of feature columns used as inputs.
        sequence_length (int): Number of time steps in each sequence.

    Returns:
        X (np.ndarray): Input sequences with shape (samples, sequence_length, features)
        y (np.ndarray): Target values with shape (samples,)
    """
    X, y = [], []
    if len(df) < sequence_length:
        raise ValueError(f"Not enough rows ({len(df)}) to create sequences of length {sequence_length}.")

    for i in range(sequence_length, len(df)):
        X_seq = df[feature_cols].iloc[i - sequence_length:i].values
        y_target = df['Close'].iloc[i]
        X.append(X_seq)
        y.append(y_target)

    return np.array(X), np.array(y)
