import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

# === Load Models ===
def load_models(ticker):
    try:
        lstm_model = load_model(f"models/{ticker}/lstm_model.h5")
        xgb_model = joblib.load(f"models/{ticker}/xgb_model.pkl")
        scaler = joblib.load(f"models/{ticker}/scaler.pkl")
    except Exception as e:
        raise ValueError(f"Failed to load models for {ticker}: {e}")
    return lstm_model, xgb_model, scaler

# === Scaling ===
def scale_features(df, feature_cols, scaler=None, include_close=False):
    cols = feature_cols + (['Close'] if include_close else [])
    if not all(col in df.columns for col in cols):
        raise ValueError(f"Missing columns: {[c for c in cols if c not in df.columns]}")

    values = df[cols].values
    if scaler is None:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(values)
    else:
        scaled = scaler.transform(values)
    return scaled, scaler

# === Create Sequences for LSTM ===
def create_lstm_sequences(data, sequence_length=60):
    if len(data) < sequence_length:
        return np.array([])
    return np.array([data[i:i + sequence_length] for i in range(len(data) - sequence_length)])

# === Predict with LSTM ===
def predict_next_lstm(df, feature_cols, lstm_model, scaler):
    try:
        X_scaled, _ = scale_features(df, feature_cols, scaler, include_close=True)
        sequence_length = 60
        X_seq = create_lstm_sequences(X_scaled, sequence_length=sequence_length)

        if len(X_seq) == 0:
            raise ValueError("Not enough data to form LSTM sequence.")

        X_input = X_seq[-1].reshape(1, sequence_length, X_scaled.shape[1])
        pred_scaled = lstm_model.predict(X_input, verbose=0)[0][0]

        # Inverse scale the prediction
        close_min = scaler.data_min_[-1]
        close_max = scaler.data_max_[-1]
        pred_close = pred_scaled * (close_max - close_min) + close_min
        return pred_close

    except Exception as e:
        raise RuntimeError(f"LSTM Prediction failed: {e}")

# === Predict with XGBoost ===
def predict_next_xgb(df, feature_cols, xgb_model, scaler):
    try:
        X_scaled, _ = scale_features(df, feature_cols, scaler, include_close=True)
        X_input = X_scaled[-1][:-1].reshape(1, -1)  # exclude 'Close'
        pred_scaled = xgb_model.predict(X_input)[0]

        # Replace last close with prediction to inverse transform
        last_row = X_scaled[-1].copy()
        last_row[-1] = pred_scaled
        pred_close = scaler.inverse_transform([last_row])[0][-1]
        return pred_close

    except Exception as e:
        raise RuntimeError(f"XGBoost Prediction failed: {e}")

# === SHAP Explanation ===
def explain_xgb_prediction(model, input_df, feature_cols):
    try:
        if not isinstance(input_df, pd.DataFrame):
            input_df = pd.DataFrame(input_df, columns=feature_cols)

        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)

        shap.plots.bar(shap_values[0], show=False)
        plt.tight_layout()
        return plt.gcf()

    except Exception as e:
        raise RuntimeError(f"SHAP explanation failed: {e}")
