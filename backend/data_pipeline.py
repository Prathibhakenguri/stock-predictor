# backend/data_pipeline.py

import yfinance as yf
import pandas as pd
import pandas_ta as ta

def get_stock_data(ticker):
    print(f"📥 Downloading data for: {ticker}")
    df = yf.download(ticker, period="10y", auto_adjust=False)
    if df.empty:
        raise ValueError("No data fetched from yfinance for ticker: " + ticker)

    print(f"📊 Raw data rows: {len(df)}")
    print(f"📋 Columns fetched: {df.columns.tolist()}")  # Add this for debug

    df.columns = [col if isinstance(col, str) else col[0] for col in df.columns]

    # === Indicators
    try:
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
        df.ta.mom(length=10, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.sma(length=5, append=True)
        df.ta.ema(length=20, append=True)
        df.ta.wma(length=14, append=True)
        df.ta.cci(length=14, append=True)
        df.ta.cmo(length=14, append=True)
        df.ta.willr(length=14, append=True)
        df.ta.roc(length=10, append=True)
        df.ta.obv(append=True)
        df.ta.stoch(length=14, smooth_k=3, append=True)
    except Exception as e:
        print("⚠️ Indicator error:", e)

    rename_map = {
        'RSI_14': 'RSI',
        'MACD_12_26_9': 'MACD',
        'MACDs_12_26_9': 'MACD_signal',
        'MOM_10': 'MOM',
        'ADX_14': 'ADX',
        'SMA_5': 'SMA_5',
        'EMA_20': 'EMA_20',
        'WMA_14': 'WMA',
        'CCI_14': 'CCI',
        'CMO_14': 'CMO',
        'WILLR_14': 'WILLR',
        'ROC_10': 'ROC',
        'OBV': 'OBV',
        'STOCHk_14_3_3': 'STOCH_k',
        'STOCHd_14_3_3': 'STOCH_d'
    }
    df.rename(columns=rename_map, inplace=True)

    df.dropna(inplace=True)

    print(f"✅ Rows after indicators & cleanup: {len(df)}")

    if df.empty:
        raise ValueError("Data is empty after computing indicators and cleaning.")

    return df


# Test mode for development
if __name__ == "__main__":
    test_df = get_stock_data("AAPL")
    print("\n📋 Final Columns:\n", test_df.columns.tolist())
    print(test_df.tail())
