import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

from paths import PROJECT_DIR as PROJECT

print("Starting XGBoost experiment...", flush=True)

# 1. Load data
df = pd.read_csv(
    PROJECT / "tcs_with_indicators.csv",
    index_col="Date",
    parse_dates=True
).sort_index()

# 2. Recreate the same seven features as our other models
df["Return_1d"] = df["Close"].pct_change()
df["Return_5d"] = df["Close"].pct_change(5)
df["SMA_Distance"] = df["Close"] / df["SMA_20"] - 1
df["EMA_Distance"] = df["Close"] / df["EMA_20"] - 1
df["Volatility_5d"] = df["Return_1d"].rolling(5).std()
df["Volume_Change"] = df["Volume"].pct_change()

base_features = [
    "Return_1d",
    "Return_5d",
    "SMA_Distance",
    "EMA_Distance",
    "RSI_14",
    "Volatility_5d",
    "Volume_Change"
]

# 3. Additional features using historical information only
df["Return_2d"] = df["Close"].pct_change(2)
df["Return_3d"] = df["Close"].pct_change(3)
df["Return_10d"] = df["Close"].pct_change(10)

df["Volatility_10d"] = (
    df["Return_1d"].rolling(10).std()
)

df["Volume_Ratio"] = (
    df["Volume"]
    / df["Volume"].rolling(10).mean()
)

df["Daily_Range"] = (
    (df["High"] - df["Low"]) / df["Close"]
)

df["MACD_Relative"] = (
    df["MACD"] / df["Close"]
)

extra_features = [
    "Return_2d",
    "Return_3d",
    "Return_10d",
    "Volatility_10d",
    "Volume_Ratio",
    "Daily_Range",
    "MACD_Relative"
]

# 4. Predict tomorrow's return
df["Target_Return"] = (
    df["Close"].shift(-1) / df["Close"] - 1
)

df["Actual_Next_Close"] = df["Close"].shift(-1)

df = df.replace([np.inf, -np.inf], np.nan)

# Keep the SAME rows as RF, Ridge and LSTM.
# Extra features should not remove additional test dates.
df = df.dropna(
    subset=base_features
    + ["Target_Return", "Actual_Next_Close"]
).copy()

# 5. Chronological split
train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)

development = df.iloc[:val_end].copy()
test = df.iloc[val_end:].copy()

print("\nDATASET")
print("Total rows:", len(df))
print("Development rows:", len(development))
print("Test rows:", len(test))

# 6. Walk-forward validation
# Each fold trains on earlier dates and validates on later dates.
# The final 15% test period is never used to choose settings.

n = len(development)

folds = [
    (int(n * 0.55), int(n * 0.65)),
    (int(n * 0.65), int(n * 0.75)),
    (int(n * 0.75), n)
]

configurations = [
    {
        "n_estimators": 100,
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 10
    },
    {
        "n_estimators": 200,
        "max_depth": 2,
        "learning_rate": 0.02,
        "min_child_weight": 15
    },
    {
        "n_estimators": 150,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 20
    }
]

feature_sets = {
    "Base": base_features,
    "Expanded": base_features + extra_features
}

def create_model(settings):
    return XGBRegressor(
        **settings,
        objective="reg:squarederror",
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

validation_records = []

print("\nWALK-FORWARD VALIDATION", flush=True)

for feature_name, features in feature_sets.items():

    for settings in configurations:

        fold_maes = []
        fold_baselines = []

        for fold_number, (end_train, end_val) in enumerate(
            folds, start=1
        ):

            # Remove the last training row to avoid
            # using a target from the validation boundary.
            fold_train = development.iloc[:end_train - 1]
            fold_val = development.iloc[end_train:end_val]

            # Extra features may be missing in early rows.
            fold_train = fold_train.dropna(
                subset=features
            )

            fold_val = fold_val.dropna(
                subset=features
            )

            model = create_model(settings)

            model.fit(
                fold_train[features],
                fold_train["Target_Return"]
            )

            predicted_returns = model.predict(
                fold_val[features]
            )

            current = fold_val["Close"].to_numpy()
            actual = (
                fold_val["Actual_Next_Close"].to_numpy()
            )

            predicted = current * (1 + predicted_returns)

            fold_maes.append(
                mean_absolute_error(actual, predicted)
            )

            fold_baselines.append(
                mean_absolute_error(actual, current)
            )

        average_mae = float(np.mean(fold_maes))
        average_baseline = float(np.mean(fold_baselines))

        record = {
            "Feature_Set": feature_name,
            **settings,
            "Validation_MAE": average_mae,
            "Baseline_MAE": average_baseline
        }

        validation_records.append(record)

        print(
            f"{feature_name} | "
            f"Depth={settings['max_depth']} | "
            f"Trees={settings['n_estimators']} | "
            f"MAE={average_mae:.2f} | "
            f"Baseline={average_baseline:.2f}",
            flush=True
        )

# 7. Select settings using validation results only
validation_df = pd.DataFrame(validation_records)

validation_df.to_csv(
    PROJECT / "xgboost_validation.csv",
    index=False
)

best = validation_df.loc[
    validation_df["Validation_MAE"].idxmin()
]

best_features = feature_sets[best["Feature_Set"]]

best_settings = {
    "n_estimators": int(best["n_estimators"]),
    "max_depth": int(best["max_depth"]),
    "learning_rate": float(best["learning_rate"]),
    "min_child_weight": int(best["min_child_weight"])
}

print("\nSELECTED SETTINGS")
print("Features:", best["Feature_Set"])
print("Settings:", best_settings)
print("Validation MAE:", round(best["Validation_MAE"], 2))
print("Validation baseline:", round(best["Baseline_MAE"], 2))

# 8. Fit the selected model on development data
final_train = development.iloc[:-1].dropna(
    subset=best_features
)

final_model = create_model(best_settings)

final_model.fit(
    final_train[best_features],
    final_train["Target_Return"]
)

# 9. Test on the same dates as RF and Ridge
if test[best_features].isna().any().any():
    raise ValueError(
        "Missing test features. Check the dataset."
    )

predicted_returns = final_model.predict(
    test[best_features]
)

current = test["Close"].to_numpy()
actual = test["Actual_Next_Close"].to_numpy()

predicted = current * (1 + predicted_returns)

mae = mean_absolute_error(actual, predicted)

rmse = np.sqrt(
    mean_squared_error(actual, predicted)
)

mape = (
    mean_absolute_percentage_error(actual, predicted)
    * 100
)

r2 = r2_score(actual, predicted)

baseline_mae = mean_absolute_error(actual, current)

baseline_rmse = np.sqrt(
    mean_squared_error(actual, current)
)

directional_accuracy = (
    np.mean(
        np.sign(actual - current)
        == np.sign(predicted - current)
    ) * 100
)

print("\nXGBOOST TEST RESULTS")
print("Model MAE:", round(mae, 2))
print("Model RMSE:", round(rmse, 2))
print("Model MAPE:", round(mape, 2), "%")
print("Model R2:", round(r2, 4))
print("Baseline MAE:", round(baseline_mae, 2))
print("Baseline RMSE:", round(baseline_rmse, 2))
print(
    "Directional accuracy:",
    round(directional_accuracy, 2),
    "%"
)

if mae < baseline_mae:
    print("XGBoost beat the baseline on this test period.")
else:
    print("XGBoost did not beat the baseline.")

# 10. Check dates against existing RF results
rf_path = PROJECT / "random_forest_results.csv"

if rf_path.exists():

    rf = pd.read_csv(
        rf_path,
        index_col=0,
        parse_dates=True
    )

    same_dates = (
        pd.DatetimeIndex(rf.index).normalize().equals(
            pd.DatetimeIndex(test.index).normalize()
        )
    )

    print("\nIdentical RF test dates:", same_dates)

# 11. Save results and trained model
results = pd.DataFrame(
    {
        "Actual": actual,
        "Predicted": predicted,
        "Baseline": current,
        "Current_Close": current,
        "Predicted_Return": predicted_returns
    },
    index=test.index
)

results.index.name = "Date"

results.to_csv(
    PROJECT / "xgboost_results.csv"
)

metrics = pd.DataFrame([{
    "Model": "XGBoost",
    "MAE": mae,
    "RMSE": rmse,
    "MAPE": mape,
    "R2": r2,
    "Baseline_MAE": baseline_mae,
    "Baseline_RMSE": baseline_rmse,
    "Directional_Accuracy": directional_accuracy,
    "Test_Rows": len(test)
}])

metrics.to_csv(
    PROJECT / "xgboost_metrics.csv",
    index=False
)

joblib.dump(
    {
        "model": final_model,
        "features": best_features,
        "settings": best_settings
    },
    PROJECT / "xgboost_model.pkl"
)

# 12. Prediction graph: Red, Black and Gold
plt.figure(figsize=(12, 6))

plt.plot(
    results.index,
    results["Actual"],
    color="black",
    label="Actual",
    linewidth=2
)

plt.plot(
    results.index,
    results["Predicted"],
    color="red",
    label="XGBoost",
    linewidth=1.5
)

plt.plot(
    results.index,
    results["Baseline"],
    color="#C9A227",
    label="Baseline",
    alpha=0.7
)

plt.title("XGBoost: Actual vs Predicted TCS Prices")
plt.xlabel("Date")
plt.ylabel("Next Trading Day Price (INR)")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    PROJECT / "xgboost_predictions.png",
    dpi=300
)

plt.close()

print("\nXGBoost experiment completed!")