# app/streamlit_app.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backend.data_pipeline import get_stock_data
from backend.predictor import (
    load_models,
    predict_next_lstm,
    predict_next_xgb,
    explain_xgb_prediction,
    scale_features
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === Page Setup ===
st.set_page_config(layout="wide")
st.title("📈 Stock Market Predictor – LSTM + XGBoost")

# === Stock Selector ===
TICKERS = ['AAPL', 'TSLA', 'RELIANCE.NS', 'TCS.NS']
ticker = st.selectbox("📌 Choose a Stock", TICKERS)

# === Predict Button ===
if st.button("🔮 Predict Next Closing Price"):
    with st.spinner("🔄 Fetching and preparing data..."):
        try:
            df = get_stock_data(ticker)
        except Exception as e:
            st.error(f"⚠️ Data fetch error: {e}")
            st.stop()

        st.subheader(f"🧾 Processed Rows: {len(df)}")
        st.dataframe(df.tail(5), use_container_width=True)

        if df.empty or len(df) < 60:
            st.warning("Not enough data for prediction.")
            st.stop()

        st.subheader("💰 Latest Close Price")
        st.metric("Today’s Price", f"₹ {df['Close'].iloc[-1]:.2f}")

        # === Load Models ===
        try:
            lstm_model, xgb_model, scaler = load_models(ticker)
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            st.stop()

        feature_cols = ['RSI', 'MACD', 'MACD_signal', 'MOM', 'ADX', 'SMA_5', 'EMA_20']

        # === Make Predictions ===
        try:
            pred_lstm = predict_next_lstm(df, feature_cols, lstm_model, scaler)
            pred_xgb = predict_next_xgb(df, feature_cols, xgb_model, scaler)
        except Exception as e:
            st.error(f"⚠️ Prediction error: {e}")
            st.stop()

        col1, col2 = st.columns(2)
        col1.metric("📈 LSTM Prediction", f"₹ {pred_lstm:.2f}")
        col2.metric("🤖 XGBoost Prediction", f"₹ {pred_xgb:.2f}")

        st.line_chart(df['Close'][-90:])

        # === Accuracy Metrics on Last 7 Days (Optional) ===
        with st.expander("📊 Show XGBoost Accuracy (Last 7 Days)", expanded=False):
            try:
                recent_df = df.copy()
                X_scaled, _ = scale_features(recent_df, feature_cols, scaler, include_close=True)
                X_recent = X_scaled[-7:]
                y_actual = recent_df['Close'].values[-7:]

                preds_recent = xgb_model.predict(X_recent[:, :-1])
                y_pred_rescaled = scaler.inverse_transform(
                    np.hstack([X_recent[:, :-1], preds_recent.reshape(-1, 1)])
                )[:, -1]

                mae = mean_absolute_error(y_actual, y_pred_rescaled)
                rmse = mean_squared_error(y_actual, y_pred_rescaled, squared=False)
                r2 = r2_score(y_actual, y_pred_rescaled)
                mape = np.mean(np.abs((y_actual - y_pred_rescaled) / y_actual)) * 100

                st.write(f"✅ MAE: ₹{mae:.2f}, RMSE: ₹{rmse:.2f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")

                fig, ax = plt.subplots()
                ax.plot(y_actual, label="Actual", marker='o')
                ax.plot(y_pred_rescaled, label="Predicted", marker='x')
                ax.set_title("📊 XGBoost: Actual vs Predicted (7 Days)")
                ax.set_ylabel("Price (₹)")
                ax.set_xlabel("Days")
                ax.legend()
                st.pyplot(fig)

            except Exception as e:
                st.warning(f"Could not calculate accuracy: {e}")

        # === SHAP Feature Explanation ===
        st.subheader("🧠 Feature Contribution (XGBoost)")

        try:
            X_input_df = pd.DataFrame([X_scaled[-1][:-1]], columns=feature_cols)
            shap_fig = explain_xgb_prediction(xgb_model, X_input_df, feature_cols)
            st.pyplot(shap_fig)
        except Exception as plot_err:
            st.warning(f"SHAP plot failed: {plot_err}")
            st.info("📋 Showing raw SHAP values instead.")

            import shap
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_input_df)

            shap_df = pd.DataFrame(shap_values, columns=feature_cols).T
            shap_df.columns = ['SHAP Value']
            shap_df['Absolute Impact'] = shap_df['SHAP Value'].abs()
            shap_df = shap_df.sort_values(by='Absolute Impact', ascending=False)
            st.dataframe(shap_df[['SHAP Value']])
