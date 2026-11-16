from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# AI-BASED STOCK PREDICTION ENGINE
# Return-Based Random Forest Model
# ==========================================

from paths import PROJECT_DIR

INPUT_FILE = PROJECT_DIR / "tcs_with_indicators.csv"

RESULTS_FILE = PROJECT_DIR / "random_forest_results.csv"
METRICS_FILE = PROJECT_DIR / "random_forest_metrics.csv"
MODEL_FILE = PROJECT_DIR / "random_forest_model.pkl"
GRAPH_FILE = PROJECT_DIR / "random_forest_predictions.png"

RED = "#A1272F"
BLACK = "#171717"
GOLD = "#D4AF37"


def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mape = np.mean(
        np.abs(
            (actual - predicted) / actual
        )
    ) * 100

    r2 = r2_score(
        actual,
        predicted
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }


def train_random_forest():

    print(
        "Starting return-based Random Forest training..."
    )

    # ======================================
    # 1. LOAD DATA
    # ======================================

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Run technical_indicators.py first."
        )

    df = pd.read_csv(
        INPUT_FILE,
        index_col="Date",
        parse_dates=True
    )

    df = df.sort_index()

    # ======================================
    # 2. FEATURE ENGINEERING
    # ======================================

    df["Return_1d"] = (
        df["Close"].pct_change()
    )

    df["Return_5d"] = (
        df["Close"].pct_change(5)
    )

    df["SMA_Distance"] = (
        df["Close"] / df["SMA_20"] - 1
    )

    df["EMA_Distance"] = (
        df["Close"] / df["EMA_20"] - 1
    )

    df["Volatility_5d"] = (
        df["Return_1d"]
        .rolling(5)
        .std()
    )

    df["Volume_Change"] = (
        df["Volume"].pct_change()
    )

    df["MACD_Relative"] = (
        df["MACD"] / df["Close"]
    )

    df["MACD_Signal_Relative"] = (
        df["MACD_Signal"] / df["Close"]
    )

    df["BB_Width"] = (
        (df["BB_Upper"] - df["BB_Lower"])
        / df["BB_Middle"]
    )

    df["BB_Position"] = (
        (df["Close"] - df["BB_Lower"])
        / (df["BB_Upper"] - df["BB_Lower"])
    )

    df["Volume_Ratio"] = (
        df["Volume"]
        / df["Volume"].rolling(20).mean()
    )

    # Predict the NEXT trading day's return
    df["Target_Return"] = (
        df["Close"].shift(-1) / df["Close"] - 1
    )

    df["Target_Close"] = (
        df["Close"].shift(-1)
    )

    features = [
        "Return_1d",
        "Return_5d",
        "SMA_Distance",
        "EMA_Distance",
        "RSI_14",
        "Volatility_5d",
        "Volume_Change",
        "MACD_Relative",
        "MACD_Signal_Relative",
        "BB_Width",
        "BB_Position",
        "Volume_Ratio"
    ]

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=features + [
            "Target_Return",
            "Target_Close"
        ]
    ).copy()

    if len(df) < 200:
        raise ValueError(
            "Not enough data for training."
        )

    X = df[features]
    y = df["Target_Return"]

    # ======================================
    # 3. CHRONOLOGICAL SPLIT
    # ======================================

    train_end = int(len(df) * 0.70)
    val_end = int(len(df) * 0.85)

    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]

    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end:val_end]

    print("\nDATASET SPLIT")
    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Testing rows:", len(X_test))

    # ======================================
    # 4. MODEL SELECTION
    # ======================================

    configurations = [
        {
            "n_estimators": 200,
            "max_depth": 4,
            "min_samples_leaf": 5
        },
        {
            "n_estimators": 300,
            "max_depth": 6,
            "min_samples_leaf": 10
        },
        {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 15
        }
    ]

    best_mae = float("inf")
    best_config = None

    for config in configurations:

        candidate = RandomForestRegressor(
            **config,
            random_state=42,
            n_jobs=-1
        )

        candidate.fit(
            X_train,
            y_train
        )

        predicted_returns = candidate.predict(
            X_val
        )

        # Convert predicted returns into prices
        current_prices = df.loc[
            X_val.index,
            "Close"
        ].to_numpy()

        predicted_prices = (
            current_prices
            * (1 + predicted_returns)
        )

        actual_prices = df.loc[
            X_val.index,
            "Target_Close"
        ].to_numpy()

        val_mae = mean_absolute_error(
            actual_prices,
            predicted_prices
        )

        print(
            f"Depth {config['max_depth']}: "
            f"Validation MAE = {val_mae:.2f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            best_config = config

    print("\nSelected configuration:")
    print(best_config)

    # ======================================
    # 5. TRAIN FINAL MODEL
    # ======================================

    model = RandomForestRegressor(
        **best_config,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X.iloc[:val_end],
        y.iloc[:val_end]
    )

    # ======================================
    # 6. TEST PREDICTIONS
    # ======================================

    predicted_returns = model.predict(
        X_test
    )

    current_close = df.loc[
        X_test.index,
        "Close"
    ].to_numpy()

    actual_close = df.loc[
        X_test.index,
        "Target_Close"
    ].to_numpy()

    predicted_close = (
        current_close
        * (1 + predicted_returns)
    )

    # No-change baseline
    baseline = current_close.copy()

    model_metrics = calculate_metrics(
        actual_close,
        predicted_close
    )

    baseline_metrics = calculate_metrics(
        actual_close,
        baseline
    )

    print("\nRETURN-BASED RANDOM FOREST RESULTS")

    print(
        "Model MAE:",
        round(model_metrics["MAE"], 2)
    )

    print(
        "Model RMSE:",
        round(model_metrics["RMSE"], 2)
    )

    print(
        "Model MAPE:",
        round(model_metrics["MAPE"], 2),
        "%"
    )

    print(
        "Model R2:",
        round(model_metrics["R2"], 4)
    )

    print("\nBASELINE RESULTS")

    print(
        "Baseline MAE:",
        round(baseline_metrics["MAE"], 2)
    )

    print(
        "Baseline RMSE:",
        round(baseline_metrics["RMSE"], 2)
    )

    # ======================================
    # 7. DIRECTIONAL ACCURACY
    # ======================================

    actual_direction = (
        actual_close > current_close
    )

    predicted_direction = (
        predicted_close > current_close
    )

    directional_accuracy = np.mean(
        actual_direction == predicted_direction
    ) * 100

    print(
        "\nDirectional accuracy:",
        round(directional_accuracy, 2),
        "%"
    )

    # ======================================
    # 8. SAVE RESULTS
    # ======================================

    target_dates = (
        df.index.to_series()
        .shift(-1)
        .loc[X_test.index]
    )

    results = pd.DataFrame(
        {
            "Target_Date": target_dates.to_numpy(),
            "Current_Close": current_close,
            "Actual": actual_close,
            "Predicted": predicted_close,
            "Baseline": baseline,
            "Predicted_Return": predicted_returns
        },
        index=X_test.index
    )

    results.index.name = "Prediction_Date"

    results.to_csv(
        RESULTS_FILE
    )

    metrics_df = pd.DataFrame(
        [
            {
                "Model": "Return-Based Random Forest",
                **model_metrics,
                "Directional_Accuracy": directional_accuracy
            },
            {
                "Model": "Naive Baseline",
                **baseline_metrics,
                "Directional_Accuracy": np.nan
            }
        ]
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False
    )

    joblib.dump(
        {
            "model": model,
            "features": features,
            "target": "Next-day percentage return",
            "training_end": str(
                X.index[val_end - 1].date()
            ),
            "test_start": str(
                X_test.index.min().date()
            )
        },
        MODEL_FILE
    )

    # ======================================
    # 9. CREATE GRAPH
    # ======================================

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    ax.plot(
        results["Target_Date"],
        results["Actual"],
        color=BLACK,
        label="Actual Price",
        linewidth=1.8
    )

    ax.plot(
        results["Target_Date"],
        results["Predicted"],
        color=RED,
        label="Predicted Price",
        linewidth=1.5
    )

    ax.plot(
        results["Target_Date"],
        results["Baseline"],
        color=GOLD,
        label="Baseline",
        linewidth=1.2,
        alpha=0.8
    )

    ax.set_title(
        "Return-Based Random Forest: Actual vs Predicted",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel("Target Trading Date")
    ax.set_ylabel("TCS Closing Price (INR)")

    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        GRAPH_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nFiles saved successfully!")
    print(RESULTS_FILE.name)
    print(METRICS_FILE.name)
    print(MODEL_FILE.name)
    print(GRAPH_FILE.name)

    return model


if __name__ == "__main__":
    train_random_forest()