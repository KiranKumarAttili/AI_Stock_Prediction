# AI-Based Stock Prediction Engine

An integrated AI framework for short-term stock prediction and performance analysis, built for **TCS.NS** (Tata Consultancy Services). It collects daily market data, computes technical indicators, trains four models, evaluates them against a no-change baseline, and presents everything in an interactive Streamlit dashboard.

> Academic research prototype (Special Project, Faculty of Science & Technology, IFHE Hyderabad). Outputs are **not** financial advice.

## Features

- Data collection (Yahoo Finance via `yfinance`), cleaning and validation
- Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands
- Models: Random Forest, Ridge Regression, XGBoost, LSTM (next-day return prediction, chronological 70/15/15 split)
- Evaluation: MAE, RMSE, MAPE, R², directional accuracy vs a no-change baseline
- Streamlit dashboard with interactive charts and a hover help panel
- Next-day prediction engine

## Results (183-day test period)

| Method | MAE (₹) | RMSE (₹) | R² | Directional acc. (%) |
|---|---|---|---|---|
| Random Forest | 34.63 | 49.36 | 0.9753 | 55.19 |
| Ridge Regression | 34.91 | 49.47 | 0.9752 | 46.99 |
| XGBoost | 34.67 | 49.12 | 0.9756 | 51.91 |
| LSTM | 73.26 | 105.68 | 0.8869 | 44.81 |
| No-change baseline | 34.20 | 48.65 | 0.9760 | - |

None of the models consistently beat the no-change baseline, which is why AI forecasts of daily prices must always be benchmarked against simple baselines.

## Repository structure

```
Dataset/                  tcs_stock_data.csv, tcs_cleaned_data.csv, tcs_with_indicators.csv
Implementation_Code/
  app.py                  Streamlit dashboard
  prediction_engine.py    next-day predictions from saved models
  data_collection.py      download TCS.NS data
  data_preprocessing.py   clean and validate data
  technical_indicators.py compute indicators and graphs
  random_forest_model.py  ridge_regression_model.py  xgboost_experiment.py  lstm_model.py
  paths.py                maps files to Dataset / Saved_Models / Results_and_Plots
  Saved_Models/           trained models
  Results_and_Plots/      metrics, predictions, graphs
docs/                     report, mid-term report, presentation
requirements.txt
```

## Setup

Requires Python 3.10 or later.

```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv venv
venv\Scripts\activate        # Windows  (use: source venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

## Run

Dashboard:

```bash
cd Implementation_Code
streamlit run app.py
```

Next-day prediction from the command line:

```bash
cd Implementation_Code
python prediction_engine.py
```

Re-run the full pipeline (from `Implementation_Code`, in this order):

```bash
python data_collection.py
python data_preprocessing.py
python technical_indicators.py
python random_forest_model.py
python ridge_regression_model.py
python xgboost_experiment.py
python lstm_model.py
```

`data_collection.py` needs an internet connection.

## Authors

- Attili V S S S Kiran Kumar (24STUCHH010745)
- Anjum Sultana (24STUCHH010681)

Supervisor: Dr. Sathya AR

## License

MIT - see [LICENSE](LICENSE).
