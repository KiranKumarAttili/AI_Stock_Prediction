import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

print("Starting Ridge Regression training...")

# 1. Load data
from paths import PROJECT_DIR

df = pd.read_csv(
    PROJECT_DIR / "tcs_with_indicators.csv",
    index_col="Date",
    parse_dates=True
)

df = df.sort_index()

# 2. Create return-based features
df["Return_1d"] = df["Close"].pct_change()
df["Return_5d"] = df["Close"].pct_change(5)

df["SMA_Distance"] = df["Close"] / df["SMA_20"] - 1
df["EMA_Distance"] = df["Close"] / df["EMA_20"] - 1

df["Volatility_5d"] = df["Return_1d"].rolling(5).std()
df["Volume_Change"] = df["Volume"].pct_change()

# Tomorrow's return and closing price
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

# 3. Chronological data split
train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)

train = df.iloc[:train_end]
validation = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

print("\nDATASET SPLIT")
print("Training rows:", len(train))
print("Validation rows:", len(validation))
print("Testing rows:", len(test))
print("Number of features:", len(features))

# 4. Select Ridge alpha using validation data
alphas = [0.1, 1.0, 10.0, 100.0]

best_alpha = None
best_val_mae = float("inf")

for alpha in alphas:

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha)
    )

    model.fit(
        train[features],
        train["Target_Return"]
    )

    predicted_returns = model.predict(
        validation[features]
    )

    predicted_prices = (
        validation["Close"].to_numpy()
        * (1 + predicted_returns)
    )

    mae = mean_absolute_error(
        validation["Actual_Next_Close"],
        predicted_prices
    )

    print(
        f"Alpha {alpha}: "
        f"Validation MAE = {mae:.2f}"
    )

    if mae < best_val_mae:
        best_val_mae = mae
        best_alpha = alpha

print("\nSELECTED RIDGE SETTINGS")
print("Best alpha:", best_alpha)
print("Validation MAE:", round(best_val_mae, 2))

# 5. Train final model
train_val = df.iloc[:val_end]

final_model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=best_alpha)
)

final_model.fit(
    train_val[features],
    train_val["Target_Return"]
)

# 6. Evaluate on test data
predicted_returns = final_model.predict(
    test[features]
)

current_prices = test["Close"].to_numpy()

actual_prices = (
    test["Actual_Next_Close"].to_numpy()
)

predicted_prices = (
    current_prices * (1 + predicted_returns)
)

baseline_prices = current_prices.copy()

model_mae = mean_absolute_error(
    actual_prices,
    predicted_prices
)

model_rmse = np.sqrt(
    mean_squared_error(
        actual_prices,
        predicted_prices
    )
)

model_mape = (
    mean_absolute_percentage_error(
        actual_prices,
        predicted_prices
    ) * 100
)

model_r2 = r2_score(
    actual_prices,
    predicted_prices
)

baseline_mae = mean_absolute_error(
    actual_prices,
    baseline_prices
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        actual_prices,
        baseline_prices
    )
)

actual_direction = np.sign(
    actual_prices - current_prices
)

predicted_direction = np.sign(
    predicted_prices - current_prices
)

directional_accuracy = (
    np.mean(
        actual_direction == predicted_direction
    ) * 100
)

print("\nRIDGE REGRESSION RESULTS")
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

# 7. Save results
results = pd.DataFrame(
    {
        "Actual": actual_prices,
        "Predicted": predicted_prices,
        "Baseline": baseline_prices,
        "Current_Close": current_prices,
        "Actual_Return": (
            actual_prices / current_prices - 1
        ),
        "Predicted_Return": predicted_returns
    },
    index=test.index
)

results.index.name = "Date"

results.to_csv(
    PROJECT_DIR / "ridge_regression_results.csv"
)

metrics = pd.DataFrame(
    [{
        "Model": "Ridge Regression",
        "Alpha": best_alpha,
        "Validation_MAE": best_val_mae,
        "MAE": model_mae,
        "RMSE": model_rmse,
        "MAPE": model_mape,
        "R2": model_r2,
        "Baseline_MAE": baseline_mae,
        "Baseline_RMSE": baseline_rmse,
        "Directional_Accuracy": directional_accuracy,
        "Train_Validation_Rows": len(train_val),
        "Test_Rows": len(test)
    }]
)

metrics.to_csv(
    PROJECT_DIR / "ridge_regression_metrics.csv",
    index=False
)

joblib.dump(
    {
        "model": final_model,
        "features": features,
        "alpha": best_alpha
    },
    PROJECT_DIR / "ridge_regression_model.pkl"
)

# 8. Check whether RF test dates match
rf_file = PROJECT_DIR / "random_forest_results.csv"

if rf_file.exists():

    rf_results = pd.read_csv(
        rf_file,
        index_col=0,
        parse_dates=True
    )

    rf_dates = pd.DatetimeIndex(
        rf_results.index
    ).normalize()

    ridge_dates = pd.DatetimeIndex(
        results.index
    ).normalize()

    same_dates = rf_dates.equals(ridge_dates)

    print("\nRANDOM FOREST COMPARISON")
    print("RF test rows:", len(rf_dates))
    print("Ridge test rows:", len(ridge_dates))
    print("Identical test dates:", same_dates)

    if not same_dates:
        print(
            "WARNING: Test dates differ. "
            "Do not directly rank the models "
            "until they are evaluated on "
            "the same dates."
        )

# 9. Save prediction graph
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
    label="Ridge Prediction",
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
    "Ridge Regression: Actual vs Predicted TCS Prices"
)

plt.xlabel("Date")
plt.ylabel("Next Trading Day Price (INR)")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    PROJECT_DIR / "ridge_regression_predictions.png",
    dpi=300
)

plt.close()

print("\nRidge Regression experiment completed!")
print("Saved: ridge_regression_results.csv")
print("Saved: ridge_regression_metrics.csv")
print("Saved: ridge_regression_model.pkl")
print("Saved: ridge_regression_predictions.png")