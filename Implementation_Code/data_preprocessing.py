from pathlib import Path

import numpy as np
import pandas as pd


# File paths
from paths import PROJECT_DIR

INPUT_FILE = PROJECT_DIR / "tcs_stock_data.csv"
OUTPUT_FILE = PROJECT_DIR / "tcs_cleaned_data.csv"

REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]


def preprocess_data():

    print("Starting data preprocessing...")

    # 1. Load the original dataset
    df = pd.read_csv(
        INPUT_FILE,
        index_col="Date",
        parse_dates=True
    )

    original_rows = len(df)

    # 2. Check required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    # 3. Convert values to numbers
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # 4. Check missing values
    print("\nMissing values before cleaning:")
    print(df.isna().sum())

    # 5. Replace infinite values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # 6. Sort by date
    df = df.sort_index()

    # 7. Remove duplicate dates
    duplicate_count = df.index.duplicated(
        keep="last"
    ).sum()

    df = df[
        ~df.index.duplicated(keep="last")
    ]

    # 8. Remove invalid price and volume records
    valid_prices = (
        (df["Open"] > 0)
        & (df["High"] > 0)
        & (df["Low"] > 0)
        & (df["Close"] > 0)
    )

    valid_volume = df["Volume"] >= 0

    invalid_count = (
        ~(valid_prices & valid_volume)
    ).sum()

    df = df[
        valid_prices & valid_volume
    ]

    # 9. Remove missing values
    df = df.dropna(
        subset=REQUIRED_COLUMNS
    )

    if df.empty:
        raise ValueError(
            "No valid stock data remains."
        )

    # 10. Correct the date column name
    df.index.name = "Date"

    # 11. Save the cleaned dataset
    df.to_csv(
        OUTPUT_FILE,
        index_label="Date"
    )

    # 12. Display results
    print("\nMissing values after cleaning:")
    print(df.isna().sum())

    print("\nData preprocessing completed successfully!")

    print("Original rows:", original_rows)
    print("Cleaned rows:", len(df))
    print("Rows removed:", original_rows - len(df))
    print("Duplicate dates:", duplicate_count)
    print("Invalid records:", invalid_count)

    print("First date:", df.index.min().date())
    print("Latest date:", df.index.max().date())

    print("Dataset saved:", OUTPUT_FILE.name)

    return df


if __name__ == "__main__":
    preprocess_data()