from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# AI-BASED STOCK PREDICTION ENGINE
# Technical Indicator Calculation
# ==========================================

from paths import PROJECT_DIR

INPUT_FILE = PROJECT_DIR / "tcs_cleaned_data.csv"
OUTPUT_FILE = PROJECT_DIR / "tcs_with_indicators.csv"

PRICE_GRAPH = PROJECT_DIR / "tcs_technical_indicators.png"
RSI_MACD_GRAPH = PROJECT_DIR / "tcs_rsi_macd.png"

# Red x Black x Gold theme
RED = "#A1272F"
BLACK = "#171717"
GOLD = "#D4AF37"


def calculate_rsi(close, period=14):
    """
    Calculate the Relative Strength Index
    using Wilder's smoothing method.
    """

    change = close.diff()

    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    # Initial averages from the first 14 changes
    avg_gain = pd.Series(
        np.nan,
        index=close.index,
        dtype=float
    )

    avg_loss = pd.Series(
        np.nan,
        index=close.index,
        dtype=float
    )

    if len(close) <= period:
        return avg_gain

    avg_gain.iloc[period] = (
        gain.iloc[1:period + 1].mean()
    )

    avg_loss.iloc[period] = (
        loss.iloc[1:period + 1].mean()
    )

    # Wilder's recursive smoothing
    for i in range(period + 1, len(close)):

        avg_gain.iloc[i] = (
            avg_gain.iloc[i - 1] * (period - 1)
            + gain.iloc[i]
        ) / period

        avg_loss.iloc[i] = (
            avg_loss.iloc[i - 1] * (period - 1)
            + loss.iloc[i]
        ) / period

    # Avoid division by zero
    rsi = pd.Series(
        np.nan,
        index=close.index,
        dtype=float
    )

    for i in range(period, len(close)):

        current_gain = avg_gain.iloc[i]
        current_loss = avg_loss.iloc[i]

        if current_gain == 0 and current_loss == 0:
            rsi.iloc[i] = 50.0

        elif current_loss == 0:
            rsi.iloc[i] = 100.0

        else:
            rs = current_gain / current_loss

            rsi.iloc[i] = (
                100 - (100 / (1 + rs))
            )

    return rsi


def calculate_indicators():

    print("Calculating technical indicators...")

    # ======================================
    # 1. LOAD CLEANED STOCK DATA
    # ======================================

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "tcs_cleaned_data.csv was not found. "
            "Run data_preprocessing.py first."
        )

    df = pd.read_csv(
        INPUT_FILE,
        index_col="Date",
        parse_dates=True
    )

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    df = df.sort_index()

    df.index.name = "Date"

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    if df["Close"].isna().any():
        raise ValueError(
            "The cleaned dataset contains invalid closing prices."
        )

    if len(df) < 35:
        raise ValueError(
            "At least 35 trading days are required."
        )

    close = df["Close"]

    # ======================================
    # 2. SIMPLE MOVING AVERAGES
    # ======================================

    df["SMA_20"] = close.rolling(
        window=20,
        min_periods=20
    ).mean()

    df["SMA_50"] = close.rolling(
        window=50,
        min_periods=50
    ).mean()

    # ======================================
    # 3. EXPONENTIAL MOVING AVERAGES
    # ======================================

    df["EMA_20"] = close.ewm(
        span=20,
        adjust=False
    ).mean()

    df["EMA_50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    # ======================================
    # 4. RELATIVE STRENGTH INDEX
    # ======================================

    df["RSI_14"] = calculate_rsi(
        close,
        period=14
    )

    # ======================================
    # 5. MACD
    # ======================================

    ema_12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema_26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema_12 - ema_26

    df["MACD_Signal"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD_Histogram"] = (
        df["MACD"] - df["MACD_Signal"]
    )

    # ======================================
    # 6. BOLLINGER BANDS
    # ======================================

    rolling_mean = close.rolling(
        window=20,
        min_periods=20
    ).mean()

    rolling_std = close.rolling(
        window=20,
        min_periods=20
    ).std()

    df["BB_Middle"] = rolling_mean

    df["BB_Upper"] = (
        rolling_mean + 2 * rolling_std
    )

    df["BB_Lower"] = (
        rolling_mean - 2 * rolling_std
    )

    # ======================================
    # 7. SAVE DATASET
    # ======================================

    # Keep initial NaN values because
    # rolling indicators need historical data.
    # Model scripts will remove warm-up rows
    # after selecting their required features.

    df.index.name = "Date"

    df.to_csv(
        OUTPUT_FILE,
        index_label="Date"
    )

    # ======================================
    # 8. PRICE AND INDICATOR GRAPH
    # ======================================

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    ax.plot(
        df.index,
        df["Close"],
        color=BLACK,
        linewidth=1.5,
        label="TCS Closing Price"
    )

    ax.plot(
        df.index,
        df["SMA_20"],
        color=RED,
        linewidth=1.3,
        label="20-Day SMA"
    )

    ax.plot(
        df.index,
        df["EMA_20"],
        color=GOLD,
        linewidth=1.3,
        label="20-Day EMA"
    )

    ax.fill_between(
        df.index,
        df["BB_Lower"].to_numpy(),
        df["BB_Upper"].to_numpy(),
        color=GOLD,
        alpha=0.12,
        label="Bollinger Bands"
    )

    ax.set_title(
        "TCS Stock Price and Technical Indicators",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price (INR)")

    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        PRICE_GRAPH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ======================================
    # 9. RSI AND MACD GRAPH
    # ======================================

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True
    )

    # RSI graph
    axes[0].plot(
        df.index,
        df["RSI_14"],
        color=RED,
        linewidth=1.5,
        label="RSI (14)"
    )

    axes[0].axhline(
        70,
        color=GOLD,
        linestyle="--",
        label="RSI 70"
    )

    axes[0].axhline(
        30,
        color=BLACK,
        linestyle="--",
        label="RSI 30"
    )

    axes[0].set_title(
        "Relative Strength Index"
    )

    axes[0].set_ylabel("RSI")

    axes[0].set_ylim(0, 100)

    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    # MACD graph
    axes[1].plot(
        df.index,
        df["MACD"],
        color=RED,
        label="MACD"
    )

    axes[1].plot(
        df.index,
        df["MACD_Signal"],
        color=GOLD,
        label="Signal Line"
    )

    axes[1].bar(
        df.index,
        df["MACD_Histogram"],
        color=BLACK,
        alpha=0.45,
        label="MACD Histogram"
    )

    axes[1].axhline(
        0,
        color=BLACK,
        linewidth=0.8
    )

    axes[1].set_title(
        "Moving Average Convergence Divergence"
    )

    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("MACD")

    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        RSI_MACD_GRAPH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ======================================
    # 10. DISPLAY RESULTS
    # ======================================

    display_columns = [
        "Close",
        "SMA_20",
        "EMA_20",
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "BB_Upper",
        "BB_Lower"
    ]

    print("\nLatest five trading days:")
    print(
        df[display_columns]
        .tail()
        .round(2)
        .to_string()
    )

    print(
        "\nTechnical indicators calculated successfully!"
    )

    print("Total records:", len(df))

    print(
        "Latest available date:",
        df.index.max().date()
    )

    print("Dataset saved:", OUTPUT_FILE.name)
    print("Price graph saved:", PRICE_GRAPH.name)
    print("RSI/MACD graph saved:", RSI_MACD_GRAPH.name)

    return df


if __name__ == "__main__":
    calculate_indicators()