from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# ==========================================
# 1. PROJECT SETTINGS
# ==========================================

from paths import PROJECT_DIR

BASE_FEATURES = [
    "Return_1d",
    "Return_5d",
    "SMA_Distance",
    "EMA_Distance",
    "RSI_14",
    "Volatility_5d",
    "Volume_Change"
]


# ==========================================
# 2. LOAD AND PREPARE STOCK DATA
# ==========================================

def prepare_latest_data():

    file_path = PROJECT_DIR / "tcs_with_indicators.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            "tcs_with_indicators.csv was not found."
        )

    df = pd.read_csv(
        file_path,
        index_col="Date",
        parse_dates=True
    )

    df = df.sort_index()

    if df.empty:
        raise ValueError("Stock dataset is empty.")

    required_columns = [
        "Close",
        "High",
        "Low",
        "Volume",
        "SMA_20",
        "EMA_20",
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "BB_Middle",
        "BB_Upper",
        "BB_Lower"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing CSV columns: {missing_columns}"
        )

    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------
    # Price returns
    # --------------------------------------

    df["Return_1d"] = df["Close"].pct_change(
        fill_method=None
    )

    df["Return_2d"] = df["Close"].pct_change(
        2,
        fill_method=None
    )

    df["Return_3d"] = df["Close"].pct_change(
        3,
        fill_method=None
    )

    df["Return_5d"] = df["Close"].pct_change(
        5,
        fill_method=None
    )

    df["Return_10d"] = df["Close"].pct_change(
        10,
        fill_method=None
    )

    # --------------------------------------
    # Moving-average features
    # --------------------------------------

    df["SMA_Distance"] = (
        df["Close"] / df["SMA_20"] - 1
    )

    df["EMA_Distance"] = (
        df["Close"] / df["EMA_20"] - 1
    )

    # --------------------------------------
    # Volatility features
    # --------------------------------------

    df["Volatility_5d"] = (
        df["Return_1d"].rolling(5).std()
    )

    df["Volatility_10d"] = (
        df["Return_1d"].rolling(10).std()
    )

    # --------------------------------------
    # Volume features
    # --------------------------------------

    # The CSV contains some zero-volume records.
    # Treat those records as missing observations
    # for feature calculation only.
    #
    # The original Volume column is unchanged.

    valid_volume = df["Volume"].replace(
        0,
        np.nan
    )

    filled_volume = valid_volume.ffill()

    # This avoids division by zero.
    df["Volume_Change"] = (
        filled_volume.pct_change(
            fill_method=None
        )
    )

    df["Volume_Ratio"] = (
        filled_volume
        / filled_volume.rolling(10).mean()
    )

    # --------------------------------------
    # Additional model features
    # --------------------------------------

    df["Daily_Range"] = (
        (df["High"] - df["Low"])
        / df["Close"]
    )

    df["MACD_Relative"] = (
        df["MACD"] / df["Close"]
    )

    df["MACD_Signal_Relative"] = (
        df["MACD_Signal"] / df["Close"]
    )

    bb_range = (
        df["BB_Upper"] - df["BB_Lower"]
    )

    df["BB_Width"] = (
        bb_range
        / df["BB_Middle"].replace(
            0,
            np.nan
        )
    )

    df["BB_Position"] = (
        (df["Close"] - df["BB_Lower"])
        / bb_range.replace(
            0,
            np.nan
        )
    )

    # --------------------------------------
    # Remove invalid calculated values
    # --------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


# ==========================================
# 3. LOAD SAVED TABULAR MODELS
# ==========================================

def load_tabular_model(filename):

    model_path = PROJECT_DIR / filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found: {filename}"
        )

    saved = joblib.load(model_path)

    if isinstance(saved, dict):

        model = saved["model"]
        features = saved.get("features")

    else:

        model = saved
        features = None

    if features is None:

        if hasattr(model, "feature_names_in_"):

            features = list(
                model.feature_names_in_
            )

        else:

            features = BASE_FEATURES

    return model, list(features)


# ==========================================
# 4. PREDICT WITH TABULAR MODELS
# ==========================================

def predict_tabular_model(
    filename,
    latest_row
):

    model, features = load_tabular_model(
        filename
    )

    missing_columns = [
        feature
        for feature in features
        if feature not in latest_row.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing model features: {missing_columns}"
        )

    values = latest_row[features]

    missing_values = [
        feature
        for feature in features
        if values[feature].isna().any()
    ]

    if missing_values:
        raise ValueError(
            f"Missing latest values: {missing_values}"
        )

    predicted_return = float(
        model.predict(values)[0]
    )

    if not np.isfinite(predicted_return):
        raise ValueError(
            "Model produced an invalid prediction."
        )

    latest_close = float(
        latest_row["Close"].iloc[-1]
    )

    predicted_close = (
        latest_close * (1 + predicted_return)
    )

    return {
        "predicted_close": predicted_close,
        "predicted_return": predicted_return * 100
    }


# ==========================================
# 5. PREDICT WITH LSTM
# ==========================================

def predict_lstm(df):

    model_path = (
        PROJECT_DIR / "lstm_model.keras"
    )

    scaler_path = (
        PROJECT_DIR / "lstm_scaler.pkl"
    )

    config_path = (
        PROJECT_DIR / "lstm_config.pkl"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "lstm_model.keras was not found."
        )

    if not scaler_path.exists():
        raise FileNotFoundError(
            "lstm_scaler.pkl was not found."
        )

    # Load the original training configuration.

    if config_path.exists():

        config = joblib.load(config_path)

        features = list(
            config["features"]
        )

        sequence_length = int(
            config["sequence_length"]
        )

    else:

        features = BASE_FEATURES
        sequence_length = 20

    # Use the latest consecutive trading days.
    # Do not drop missing rows and silently
    # switch to older stock prices.

    latest_sequence = df.tail(
        sequence_length
    ).copy()

    if len(latest_sequence) < sequence_length:
        raise ValueError(
            "Not enough trading days for LSTM."
        )

    missing_columns = [
        feature
        for feature in features
        if feature not in latest_sequence.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing LSTM columns: {missing_columns}"
        )

    # Check for missing features after
    # applying the zero-volume correction.

    missing_mask = (
        latest_sequence[features].isna()
    )

    if missing_mask.any().any():

        affected = [
            feature
            for feature in features
            if missing_mask[feature].any()
        ]

        raise ValueError(
            "Missing LSTM features: "
            + ", ".join(affected)
        )

    # Load the original scaler.
    # Never fit a new scaler during prediction.

    scaler = joblib.load(
        scaler_path
    )

    if hasattr(
        scaler,
        "feature_names_in_"
    ):

        expected_features = list(
            scaler.feature_names_in_
        )

        if expected_features != features:
            raise ValueError(
                "LSTM scaler feature order mismatch."
            )

    scaled = scaler.transform(
        latest_sequence[features]
    )

    sequence = np.asarray(
        [scaled],
        dtype=np.float32
    )

    # Load the saved LSTM model.

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    predicted_return = float(
        model.predict(
            sequence,
            verbose=0
        ).reshape(-1)[0]
    )

    if not np.isfinite(predicted_return):
        raise ValueError(
            "LSTM produced an invalid prediction."
        )

    latest_close = float(
        latest_sequence["Close"].iloc[-1]
    )

    predicted_close = (
        latest_close * (1 + predicted_return)
    )

    return {
        "predicted_close": predicted_close,
        "predicted_return": predicted_return * 100
    }


# ==========================================
# 6. GENERATE ALL FOUR PREDICTIONS
# ==========================================

def generate_prediction():

    df = prepare_latest_data()

    latest_row = df.tail(1)

    latest_close = float(
        latest_row["Close"].iloc[-1]
    )

    if not np.isfinite(latest_close):
        raise ValueError(
            "Latest closing price is invalid."
        )

    latest_date = latest_row.index[-1]

    predictions = {}
    errors = {}

    model_files = {
        "Random Forest":
            "random_forest_model.pkl",

        "Ridge Regression":
            "ridge_regression_model.pkl",

        "XGBoost":
            "xgboost_model.pkl"
    }

    # Run the three tabular models.

    for name, filename in model_files.items():

        try:

            predictions[name] = (
                predict_tabular_model(
                    filename,
                    latest_row
                )
            )

        except Exception as exc:

            errors[name] = str(exc)

    # Run the saved LSTM.

    try:

        predictions["LSTM"] = (
            predict_lstm(df)
        )

    except Exception as exc:

        errors["LSTM"] = str(exc)

    # Preserve compatibility with the existing
    # dashboard's original LSTM fields.

    lstm = predictions.get(
        "LSTM",
        {}
    )

    return {
        "date": latest_date,

        "latest_close": latest_close,

        "baseline_close": latest_close,

        "predicted_close": lstm.get(
            "predicted_close"
        ),

        "predicted_return": lstm.get(
            "predicted_return"
        ),

        "predictions": predictions,

        "errors": errors
    }


# ==========================================
# 7. DISPLAY RESULTS IN TERMINAL
# ==========================================

def main():

    print(
        "Generating next-trading-day predictions...",
        flush=True
    )

    result = generate_prediction()

    print("\nLATEST STOCK DATA")

    print(
        "Date:",
        result["date"].strftime(
            "%d-%m-%Y"
        )
    )

    print(
        "Latest close:",
        f"₹{result['latest_close']:.2f}"
    )

    print(
        "No-change baseline:",
        f"₹{result['baseline_close']:.2f}"
    )

    print("\nMODEL PREDICTIONS")

    if result["predictions"]:

        for name, prediction in (
            result["predictions"].items()
        ):

            print(
                f"{name}: "
                f"₹{prediction['predicted_close']:.2f} "
                f"({prediction['predicted_return']:+.2f}%)"
            )

    else:

        print(
            "No models produced a prediction."
        )

    if result["errors"]:

        print(
            "\nMODELS THAT COULD NOT LOAD"
        )

        for name, error in (
            result["errors"].items()
        ):

            print(
                f"{name}: {error}"
            )

    print(
        "\nPrediction engine completed!"
    )


if __name__ == "__main__":
    main()