from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# Project folder
from paths import PROJECT_DIR

# Stock symbol
STOCK_SYMBOL = "TCS.NS"


def collect_stock_data():

    print("Fetching the latest available TCS stock data...")

    # Download five years of historical stock data
    stock = yf.download(
        STOCK_SYMBOL,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        multi_level_index=False
    )

    # Check whether the download succeeded
    if stock.empty:
        print("Download failed. Check your internet connection.")
        return None

    # Keep the required columns
    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for column in required_columns:
        if column not in stock.columns:
            raise ValueError(
                f"Missing column: {column}"
            )

    stock = stock[required_columns].copy()

    # Clean the dataset
    stock.index = pd.to_datetime(stock.index)
    stock.index.name = "Date"

    stock = stock.sort_index()

    stock = stock[
        ~stock.index.duplicated(keep="last")
    ]

    stock = stock.replace(
        [np.inf, -np.inf],
        np.nan
    )

    stock = stock.dropna(
        subset=required_columns
    )

    if stock.empty:
        raise ValueError(
            "No valid stock data after cleaning."
        )

    # Save the dataset
    csv_path = PROJECT_DIR / "tcs_stock_data.csv"

    stock.to_csv(csv_path)

    # Create the historical stock-price graph
    plt.figure(figsize=(12, 6))

    plt.plot(
        stock.index,
        stock["Close"],
        color="#A1272F",
        label="TCS Closing Price"
    )

    plt.title("TCS Historical Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Stock Price (INR)")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the graph
    graph_path = PROJECT_DIR / "tcs_stock_graph.png"

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    # Display results
    print("\nData collection completed successfully!")

    print("Total trading days:", len(stock))

    print(
        "First date:",
        stock.index.min().date()
    )

    print(
        "Latest available date:",
        stock.index.max().date()
    )

    print(
        "Latest closing price:",
        round(float(stock["Close"].iloc[-1]), 2)
    )

    print("Dataset saved:", csv_path.name)
    print("Graph saved:", graph_path.name)

    return stock


# Run the data collection
if __name__ == "__main__":
    collect_stock_data()