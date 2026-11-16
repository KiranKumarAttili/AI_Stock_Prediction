from paths import PROJECT_DIR
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the same dataset used for LSTM training
df = pd.read_csv(
    PROJECT_DIR / "tcs_with_indicators.csv",
    index_col=0,
    parse_dates=True
)

# Recreate the original training features
df["Return_1d"] = df["Close"].pct_change()
df["Return_5d"] = df["Close"].pct_change(5)

df["SMA_Distance"] = df["Close"] / df["SMA_20"] - 1
df["EMA_Distance"] = df["Close"] / df["EMA_20"] - 1

df["Volatility_5d"] = df["Return_1d"].rolling(5).std()
df["Volume_Change"] = df["Volume"].pct_change()

df["Target_Return"] = (
    df["Close"].shift(-1) / df["Close"] - 1
)
df["Actual_Next_Close"] = df["Close"].shift(-1)

features = [
    "Return_1d",
    "Return_5d",
    "SMA_Distance",
    "EMA_Distance",
    "RSI_14",
    "Volatility_5d",
    "Volume_Change"
]

# Match the original training data cleaning
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(
    subset=features + ["Target_Return", "Actual_Next_Close"]
)

# Fit only on the original chronological training portion
train_end = int(len(df) * 0.6)

scaler = StandardScaler()
scaler.fit(df.iloc[:train_end][features])

# Save the fitted scaler
joblib.dump(scaler, PROJECT_DIR / "lstm_scaler.pkl")

print("LSTM scaler saved successfully!")
print("Training rows:", train_end)
print("Features:", scaler.feature_names_in_.tolist())