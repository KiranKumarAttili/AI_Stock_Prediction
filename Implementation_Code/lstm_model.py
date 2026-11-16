import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from paths import PROJECT_DIR

print("Starting return-based LSTM training...")

# 1. Load data
df = pd.read_csv(
    PROJECT_DIR / "tcs_with_indicators.csv",
    index_col="Date",
    parse_dates=True
)

df = df.sort_index()

# 2. Create features
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

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(
    subset=features + [
        "Target_Return",
        "Actual_Next_Close"
    ]
)

# 3. Same chronological split as Ridge
train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)

print("\nDATASET SPLIT")
print("Training rows:", train_end)
print("Validation rows:", val_end - train_end)
print("Testing rows:", len(df) - val_end)

# 4. Fit scaler on training data only
scaler = StandardScaler()

scaler.fit(
    df.iloc[:train_end][features]
)

scaled_features = scaler.transform(
    df[features]
)

# 5. Create 20-day sequences
sequence_length = 20

X = []
y = []
dates = []
current_prices = []
actual_prices = []
positions = []

for i in range(sequence_length - 1, len(df)):

    X.append(
        scaled_features[
            i - sequence_length + 1:i + 1
        ]
    )

    y.append(
        df["Target_Return"].iloc[i]
    )

    dates.append(df.index[i])

    current_prices.append(
        df["Close"].iloc[i]
    )

    actual_prices.append(
        df["Actual_Next_Close"].iloc[i]
    )

    positions.append(i)

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.float32)

dates = pd.DatetimeIndex(dates)

current_prices = np.asarray(current_prices)
actual_prices = np.asarray(actual_prices)
positions = np.asarray(positions)

train_mask = positions < train_end

val_mask = (
    (positions >= train_end)
    & (positions < val_end)
)

test_mask = positions >= val_end

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]

test_dates = dates[test_mask]

print("\nSEQUENCES")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))
print("Features:", len(features))
print("Sequence length:", sequence_length)

# 6. Build LSTM
model = Sequential([
    Input(
        shape=(sequence_length, len(features))
    ),
    LSTM(32),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse"
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

print("\nTraining LSTM...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    shuffle=False,
    verbose=1
)

# 7. Predict on test data
predicted_returns = model.predict(
    X_test,
    verbose=0
).flatten()

test_current = current_prices[test_mask]
test_actual = actual_prices[test_mask]

predicted_prices = (
    test_current * (1 + predicted_returns)
)

baseline_prices = test_current.copy()

# 8. Evaluate
model_mae = mean_absolute_error(
    test_actual,
    predicted_prices
)

model_rmse = np.sqrt(
    mean_squared_error(
        test_actual,
        predicted_prices
    )
)

model_mape = (
    mean_absolute_percentage_error(
        test_actual,
        predicted_prices
    ) * 100
)

model_r2 = r2_score(
    test_actual,
    predicted_prices
)

baseline_mae = mean_absolute_error(
    test_actual,
    baseline_prices
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        test_actual,
        baseline_prices
    )
)

actual_direction = np.sign(
    test_actual - test_current
)

predicted_direction = np.sign(
    predicted_prices - test_current
)

directional_accuracy = (
    np.mean(
        actual_direction == predicted_direction
    ) * 100
)

print("\nLSTM TEST RESULTS")
print("Model MAE:", round(model_mae, 2))
print("Model RMSE:", round(model_rmse, 2))
print("Model MAPE:", round(model_mape, 2), "%")
print("Model R2:", round(model_r2, 4))

print("\nBASELINE RESULTS")
print("Baseline MAE:", round(baseline_mae, 2))
print("Baseline RMSE:", round(baseline_rmse, 2))

print(
    "Directional accuracy:",
    round(directional_accuracy, 2),
    "%"
)

# 9. Save predictions
results = pd.DataFrame(
    {
        "Actual": test_actual,
        "Predicted": predicted_prices,
        "Baseline": baseline_prices,
        "Current_Close": test_current,
        "Actual_Return": (
            test_actual / test_current - 1
        ),
        "Predicted_Return": predicted_returns
    },
    index=test_dates
)

results.index.name = "Date"

results.to_csv(
    PROJECT_DIR / "lstm_results.csv"
)

# 10. Save metrics
metrics = pd.DataFrame(
    [{
        "Model": "LSTM",
        "MAE": model_mae,
        "RMSE": model_rmse,
        "MAPE": model_mape,
        "R2": model_r2,
        "Baseline_MAE": baseline_mae,
        "Baseline_RMSE": baseline_rmse,
        "Directional_Accuracy": directional_accuracy,
        "Train_Sequences": len(X_train),
        "Validation_Sequences": len(X_val),
        "Test_Sequences": len(X_test),
        "Sequence_Length": sequence_length,
        "Epochs_Trained": len(history.history["loss"])
    }]
)

metrics.to_csv(
    PROJECT_DIR / "lstm_metrics.csv",
    index=False
)

# 11. Save model and scaler
model.save(
    PROJECT_DIR / "lstm_model.keras"
)

joblib.dump(
    scaler,
    PROJECT_DIR / "lstm_scaler.pkl"
)

# Save feature order for future predictions
joblib.dump(
    {
        "features": features,
        "sequence_length": sequence_length
    },
    PROJECT_DIR / "lstm_config.pkl"
)

# 12. Compare test dates with other models
for name, filename in [
    ("Random Forest", "random_forest_results.csv"),
    ("Ridge", "ridge_regression_results.csv")
]:

    file_path = PROJECT_DIR / filename

    if file_path.exists():

        other = pd.read_csv(
            file_path,
            index_col=0,
            parse_dates=True
        )

        other_dates = pd.DatetimeIndex(
            other.index
        ).normalize()

        lstm_dates = test_dates.normalize()

        print(
            f"\n{name} comparison:"
        )

        print(
            "Identical test dates:",
            other_dates.equals(lstm_dates)
        )

# 13. Save prediction graph
plt.figure(figsize=(12, 6))

plt.plot(
    results.index,
    results["Actual"],
    label="Actual Price",
    color="black",
    linewidth=2
)

plt.plot(
    results.index,
    results["Predicted"],
    label="LSTM Prediction",
    color="red",
    linewidth=1.5
)

plt.plot(
    results.index,
    results["Baseline"],
    label="No-change Baseline",
    color="#C9A227",
    alpha=0.7
)

plt.title(
    "LSTM: Actual vs Predicted TCS Prices"
)

plt.xlabel("Date")
plt.ylabel("Next Trading Day Price (INR)")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    PROJECT_DIR / "lstm_predictions.png",
    dpi=300
)

plt.close()

# 14. Save training history graph
plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss",
    color="red"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss",
    color="#C9A227"
)

plt.title("LSTM Training History")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    PROJECT_DIR / "lstm_training_history.png",
    dpi=300
)

plt.close()

print("\nLSTM experiment completed!")
print("Saved: lstm_results.csv")
print("Saved: lstm_metrics.csv")
print("Saved: lstm_model.keras")
print("Saved: lstm_scaler.pkl")
print("Saved: lstm_config.pkl")
print("Saved: lstm_predictions.png")
print("Saved: lstm_training_history.png")