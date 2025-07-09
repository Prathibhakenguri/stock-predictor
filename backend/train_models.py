import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ========================== Step 1: Data Download & Feature Engineering ==========================
def get_stock_data(ticker):
    print(f"📥 Downloading data for: {ticker}")
    df = yf.download(ticker, period="10y", auto_adjust=True)

    if df.empty:
        raise ValueError("No data fetched for ticker.")

    print(f"📊 Raw rows: {len(df)}")

    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

    # Ensure 'Close' column exists
    if 'Close' not in df.columns:
        raise ValueError("❌ 'Close' column not found after download.")

    # Technical Indicators
    try:
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
        df.ta.mom(length=10, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.sma(length=5, append=True)
        df.ta.ema(length=20, append=True)
    except Exception as e:
        print("⚠️ Indicator Error:", e)

    # Financial returns
    df['Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Rename indicators
    df.rename(columns={
        'RSI_14': 'RSI',
        'MACD_12_26_9': 'MACD',
        'MACDs_12_26_9': 'MACD_signal',
        'MOM_10': 'MOM',
        'ADX_14': 'ADX',
        'SMA_5': 'SMA_5',
        'EMA_20': 'EMA_20'
    }, inplace=True)

    df.dropna(inplace=True)
    print(f"✅ Cleaned rows: {len(df)}")
    return df

# ========================== Step 2: LSTM Sequence Builder ==========================
def create_lstm_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len][0])  # 0th column = Close (after scaling)
    return np.array(X), np.array(y)

# ========================== Step 3: Model Training ==========================
def train_and_save_models(ticker='AAPL'):
    df = get_stock_data(ticker)

    # Features to use
    features = ['RSI', 'MACD', 'MACD_signal', 'MOM', 'ADX', 'SMA_5', 'EMA_20', 'Return', 'Log_Return']
    for col in features + ['Close']:
        if col not in df.columns:
            raise ValueError(f"❌ Required column '{col}' missing in dataframe.")

    data = df[features + ['Close']]

    # ================= Scaling =================
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    model_dir = f"models/{ticker}"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")

    # =============== XGBoost ==================
    X_xgb = scaled_data[:, :-1]
    y_xgb = scaled_data[:, -1]
    split = int(len(X_xgb) * 0.8)
    X_train_xgb, X_test_xgb = X_xgb[:split], X_xgb[split:]
    y_train_xgb, y_test_xgb = y_xgb[:split], y_xgb[split:]

    xgb_model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.95,
        colsample_bytree=0.95,
        reg_alpha=0.3,
        reg_lambda=1,
        random_state=42
    )
    xgb_model.fit(X_train_xgb, y_train_xgb)
    joblib.dump(xgb_model, f"{model_dir}/xgb_model.pkl")
    print(f"✅ XGBoost model saved for {ticker}")

    # XGBoost Evaluation
    y_pred_xgb = xgb_model.predict(X_test_xgb)
    y_pred_xgb_real = scaler.inverse_transform(np.hstack([X_test_xgb, y_pred_xgb.reshape(-1, 1)]))[:, -1]
    y_test_xgb_real = scaler.inverse_transform(np.hstack([X_test_xgb, y_test_xgb.reshape(-1, 1)]))[:, -1]
    print(f"📊 XGBoost Evaluation:")
    print(f"  MAE  : ₹{mean_absolute_error(y_test_xgb_real, y_pred_xgb_real):.2f}")
    print(f"  RMSE : ₹{mean_squared_error(y_test_xgb_real, y_pred_xgb_real, squared=False):.2f}")
    print(f"  R²   : {r2_score(y_test_xgb_real, y_pred_xgb_real):.4f}")

    # SHAP
    try:
        explainer = shap.Explainer(xgb_model)
        shap_values = explainer(X_test_xgb)
        shap.summary_plot(shap_values, features=X_test_xgb, feature_names=features, show=False)
        plt.savefig(f"{model_dir}/xgb_shap_summary.png")
        plt.close()
    except Exception as e:
        print("⚠️ SHAP plot error:", e)

    # =============== LSTM ==================
    seq_len = 60
    X_lstm, y_lstm = create_lstm_sequences(scaled_data, seq_len)
    split = int(len(X_lstm) * 0.8)
    X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
    y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]

    lstm_model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_len, X_lstm.shape[2])),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

    lstm_model.fit(
        X_train_lstm, y_train_lstm,
        epochs=80,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
        callbacks=[early_stop, reduce_lr]
    )
    lstm_model.save(f"{model_dir}/lstm_model.h5")
    print(f"✅ LSTM model saved for {ticker}")

    # LSTM Evaluation
    y_pred_lstm = lstm_model.predict(X_test_lstm)
    y_pred_lstm_real = scaler.inverse_transform(
        np.hstack([X_test_lstm[:, -1, :-1], y_pred_lstm])
    )[:, -1]
    y_test_lstm_real = scaler.inverse_transform(
        np.hstack([X_test_lstm[:, -1, :-1], y_test_lstm.reshape(-1, 1)])
    )[:, -1]

    print(f"📉 LSTM Evaluation:")
    print(f"  MAE  : ₹{mean_absolute_error(y_test_lstm_real, y_pred_lstm_real):.2f}")
    print(f"  RMSE : ₹{mean_squared_error(y_test_lstm_real, y_pred_lstm_real, squared=False):.2f}")
    print(f"  MAPE : {np.mean(np.abs((y_test_lstm_real - y_pred_lstm_real) / y_test_lstm_real)) * 100:.2f}%")

# ========================== Run For All Tickers ==========================
if __name__ == "__main__":
    tickers = ['AAPL', 'TSLA', 'RELIANCE.NS', 'TCS.NS']
    for ticker in tickers:
        print(f"\n================ TRAINING: {ticker} ================\n")
        try:
            train_and_save_models(ticker)
        except Exception as e:
            print(f"❌ Failed to train {ticker}: {e}")
