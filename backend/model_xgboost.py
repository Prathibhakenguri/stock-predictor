import xgboost as xgb
import shap
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_xgboost(df, feature_cols, model_path, scaler_path, shap_plot_path=None):
    # === Step 1: Scale Data ===
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[feature_cols])
    y = df['Close'].values

    joblib.dump(scaler, scaler_path)

    # === Step 2: Train/Test Split ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # === Step 3: Initialize and Train Model ===
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    model.fit(X_train, y_train)

    # === Step 4: Evaluation ===
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ XGBoost MAE: ₹{mae:.2f}, RMSE: ₹{rmse:.2f}, R²: {r2:.4f}")

    # === Step 5: Save Model ===
    joblib.dump(model, model_path)

    # === Step 6: SHAP Plot ===
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)

        if shap_plot_path:
            shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
            plt.tight_layout()
            plt.savefig(shap_plot_path)
            plt.close()
            print(f"📊 SHAP summary plot saved to {shap_plot_path}")
        else:
            shap.summary_plot(shap_values, X_test, feature_names=feature_cols)
    except Exception as e:
        print("⚠️ SHAP explanation failed:", e)

    return model, scaler
