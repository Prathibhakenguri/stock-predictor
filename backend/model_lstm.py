from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils import scale_features, create_lstm_sequences

def train_lstm_model(df, feature_cols, model_path, scaler_path, epochs=60):
    # === Step 1: Scale and Create Sequences ===
    df_scaled, scaler = scale_features(df, feature_cols, scaler_path, include_close=True)
    X, y = create_lstm_sequences(df_scaled, feature_cols)

    # === Step 2: Define Model ===
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # === Step 3: Define Callbacks ===
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

    # === Step 4: Train Model ===
    model.fit(
        X, y,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
        callbacks=[early_stop, reduce_lr]
    )

    # === Step 5: Save Model & Return ===
    model.save(model_path)
    return model, scaler
