"""AI-Based Stock Prediction Engine — Streamlit research dashboard."""

from pathlib import Path
import altair as alt
import pandas as pd
import streamlit as st

# 1. Configuration and original dashboard design
st.set_page_config(
    page_title="AI-Based Stock Prediction Engine",
    page_icon="📈",
    layout="wide"
)

from paths import PROJECT_DIR
NAVY = "#102A56"
RED = "#A1272F"
GREEN = "#146B3A"
GOLD = "#D4A017"
BLACK = "#343A40"

st.markdown("""
<style>
.stApp {background:#F4F6FB;color:#111827}
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"] {
    background:transparent!important;box-shadow:none!important
}
[data-testid="stAppViewContainer"]>.stMain,
[data-testid="stMain"] {background:#F4F6FB}
[data-testid="stAppDeployButton"] {display:none!important}
[data-testid="stWidgetLabel"] p,
[data-testid="stSelectbox"] label p {
    color:#475569!important;font-weight:600;font-size:.82rem
}
.block-container {
    max-width:none;padding:2.5rem 340px 3rem 2.5rem
}
h1,h2,h3 {color:#102A56!important;letter-spacing:-.3px}
h2 {font-size:1.6rem!important;margin-bottom:.2rem!important}
h3 {font-size:1.15rem!important}
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#102A56 0%,#0B1E3F 100%)
}
[data-testid="stSidebar"] * {color:white!important}
[data-testid="stSidebar"] .stButton>button {
    background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.3);
    border-radius:10px;width:100%
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding:1.2rem .9rem
}
[data-testid="stSidebar"] h1 {
    font-size:1.35rem!important;line-height:1.3;font-weight:700;
    padding:.9rem 1rem;margin:0 0 .4rem;border-radius:14px;
    background:rgba(255,255,255,.08);
    border:1px solid rgba(255,255,255,.14);
    border-left:4px solid #D4A017
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color:#B8C4DA!important;font-size:.8rem;opacity:1;
    padding:.4rem .3rem 0
}
[data-testid="stSidebar"] hr {
    border-color:rgba(255,255,255,.14);margin:1.1rem 0
}
[data-testid="stSidebar"] [data-testid="stRadio"]>label {
    margin-bottom:.4rem
}
[data-testid="stSidebar"] [data-testid="stRadio"]>label p {
    font-size:.72rem;font-weight:700;letter-spacing:1.4px;
    color:#D4A017!important
}
[data-testid="stSidebar"] [role="radiogroup"] {gap:.45rem;width:100%}
[data-testid="stSidebar"] [role="radiogroup"]>div {width:100%}
[data-testid="stSidebar"] [data-testid="stRadioOption"] {
    display:flex;box-sizing:border-box
}
[data-testid="stSidebar"] [data-testid="stRadioOption"]>div {width:100%}
[data-testid="stSidebar"] [data-testid="stRadioOption"] {
    width:100%;padding:.65rem .9rem;border-radius:12px;
    background:rgba(255,255,255,.05);
    border:1px solid rgba(255,255,255,.10);
    cursor:pointer;transition:all .15s ease;margin:0
}
[data-testid="stSidebar"] [data-testid="stRadioOption"]>div>div:first-child {
    display:none
}
[data-testid="stSidebar"] [data-testid="stRadioOption"] p {
    font-size:.92rem;font-weight:500
}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {
    background:rgba(255,255,255,.14);transform:translateX(3px)
}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {
    background:#D4A017;border-color:#D4A017;
    box-shadow:0 4px 12px rgba(0,0,0,.25)
}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {
    color:#102A56!important;font-weight:700
}
[data-testid="stSidebar"] .stButton>button {
    padding:.6rem 1rem;font-weight:600;transition:all .15s ease
}
[data-testid="stSidebar"] .stButton>button:hover {
    background:rgba(255,255,255,.22);border-color:#D4A017
}
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background:rgba(255,255,255,.07);
    border:1px solid rgba(255,255,255,.14);
    border-left:4px solid #D4A017;border-radius:12px
}
[data-testid="stSidebar"] [data-testid="stAlert"] p {
    font-size:.82rem;line-height:1.5
}
[data-testid="stMetric"] {
    background:white;border:1px solid #E2E8F0;border-radius:14px;
    padding:18px 20px;box-shadow:0 3px 12px rgba(16,42,86,.05);
    border-top:3px solid #D4A017
}
[data-testid="stMetricLabel"] {color:#64748B}
[data-testid="stMetricValue"] {color:#102A56}
div.stButton>button[kind="primary"] {
    background:#102A56;border-color:#102A56;
    color:white;border-radius:10px;font-weight:600;padding:.5rem 1.4rem
}
div.stButton>button[kind="primary"]:hover {
    background:#1B3D72;border-color:#1B3D72
}
[data-testid="stDataFrame"] {
    border:1px solid #E2E8F0;border-radius:12px;overflow:hidden;
    background:white
}
[class*="st-key-tip_chart"] {
    background:white;border:1px solid #E2E8F0;border-radius:14px;
    padding:14px 16px;box-shadow:0 3px 12px rgba(16,42,86,.05)
}
[class*="st-key-tip_"] {transition:box-shadow .15s}
[class*="st-key-tip_"]:hover {
    box-shadow:0 0 0 2px rgba(212,160,23,.55);border-radius:12px
}
.project-banner {
    background:linear-gradient(135deg,#EAF1F8 0%,#F7F9FC 100%);
    border:1px solid #C9D9E8;border-left:5px solid #102A56;
    padding:24px 30px;border-radius:16px;margin-bottom:26px
}
.project-banner h1 {color:#102A56!important;font-size:1.9rem;margin:0}
.project-banner p {color:#475569;margin:8px 0 0}
.section-label {
    color:#A1272F;font-size:.78rem;font-weight:700;
    letter-spacing:1.4px;margin-bottom:4px
}
hr {border-color:#E2E8F0;margin:1.6rem 0}
[data-testid="stSelectbox"] [role="group"] {
    background:white!important;border:1px solid #CBD5E1!important;
    border-radius:10px!important
}
[data-testid="stSelectbox"] input,
[data-testid="stSelectbox"] [role="group"] * {color:#102A56!important}
#info-panel {
    position:fixed;top:3.5rem;right:1.5rem;width:290px;
    max-height:calc(100vh - 6rem);overflow:auto;z-index:999;
    background:white;border:1px solid #E2E8F0;border-top:4px solid #102A56;
    border-radius:14px;padding:18px 20px;
    box-shadow:0 6px 22px rgba(16,42,86,.10);font-family:inherit
}
#info-panel .ip-label {
    color:#A1272F;font-size:.72rem;font-weight:700;letter-spacing:1.3px
}
#info-panel h4 {color:#102A56;margin:.3rem 0 .6rem;font-size:1.05rem}
#info-panel p {color:#475569;font-size:.88rem;line-height:1.5;margin:0}
@media (max-width:1100px) {
    .block-container {padding-right:2rem}
    #info-panel {display:none}
}
</style>
""", unsafe_allow_html=True)


# Hover descriptions shown in the right-hand information panel
INFO = {
    "close": ("Closing Price", "The final price at which TCS traded at the end of a trading day. It is the main value the models try to predict."),
    "records": ("Historical Records", "The number of trading days stored in the dataset used to build indicators and train the models."),
    "symbol": ("Stock Symbol", "TCS.NS is the ticker of Tata Consultancy Services on the National Stock Exchange of India."),
    "chart_price": ("Closing Price Chart", "Shows how the closing price changed over the selected period. Hover over the line to read the exact date and price."),
    "chart_ma": ("Moving Averages Chart", "Compares the closing price with its SMA and EMA. When the price is above the averages the trend is generally upward."),
    "chart_rsi": ("RSI Chart", "Relative Strength Index on a 0-100 scale. Above 70 is often seen as overbought (may fall); below 30 as oversold (may rise)."),
    "chart_macd": ("MACD Chart", "Shows momentum as the gap between two EMAs (MACD) and its smoothed Signal line. MACD crossing above Signal hints at rising momentum."),
    "chart_bb": ("Bollinger Bands Chart", "Upper and lower bands placed around a moving average. Prices near the upper band are relatively high, near the lower band relatively low."),
    "chart_mae": ("MAE Comparison", "Bars show each method's Mean Absolute Error. Shorter bars mean smaller average mistakes."),
    "chart_pred": ("Actual vs Predicted", "Compares the real price with the model's prediction and the no-change baseline on historical test days."),
    "chart_models": ("Model Forecast Chart", "Next-day closing price predicted by each saved model."),
    "sma": ("SMA (Simple Moving Average)", "The plain average of the last 20 closing prices. It smooths out day-to-day noise to show the trend."),
    "ema": ("EMA (Exponential Moving Average)", "A 20-day average that gives more weight to recent prices, so it reacts faster than the SMA."),
    "rsi": ("RSI (14)", "Relative Strength Index over 14 days. It measures the speed of recent price changes, from 0 to 100."),
    "period": ("Time Period", "Chooses how many recent trading days are displayed in the chart."),
    "table_recent": ("Recent Stock Data", "The latest rows of the dataset: Open, High, Low, Close, Volume and the calculated indicators."),
    "table_indicator": ("Recent Indicator Data", "The most recent closing price along with its SMA, EMA and RSI values."),
    "table_compare": ("Model Comparison Table", "MAE and RMSE for each method on the same test period. Lower values mean better accuracy."),
    "mae": ("MAE (Mean Absolute Error)", "The average size of the prediction error in rupees, ignoring direction. Lower is better."),
    "rmse": ("RMSE (Root Mean Squared Error)", "Like MAE but penalises large mistakes more heavily. Lower is better."),
    "baseline": ("No-Change Baseline", "A simple benchmark that predicts tomorrow's price will equal today's close. AI models should beat it to be useful."),
    "model_select": ("Choose a Model", "Selects which saved model's historical test predictions are displayed."),
    "table_pred": ("Test Predictions", "Actual price, predicted price, baseline and current close for the latest test days. Predicted_Return is the expected daily % move."),
    "predict_btn": ("Generate AI Predictions", "Runs the four saved models on the latest data to forecast the next trading day's closing price."),
    "pred_close": ("Predicted Close", "The closing price a model expects for the next trading day."),
    "pred_change": ("Predicted Change (%)", "The expected percentage difference between the next predicted close and the latest close."),
    "table_forecast": ("Next-Day Predictions", "Each model's predicted closing price and expected percentage change for the next trading day."),
    "rf": ("Random Forest", "An ensemble of many decision trees whose predictions are averaged."),
    "lstm": ("LSTM", "Long Short-Term Memory neural network that learns patterns from sequences of past prices."),
}
INFO_DEFAULT = (
    "Information",
    "Hover over any metric, chart, table or control to see a short description here."
)


def tip(key):
    """Container whose hover text appears in the right-hand info panel."""
    return st.container(key=f"tip_{key}")


def inject_info_panel():
    import json
    payload = json.dumps(
        {k: list(v) for k, v in INFO.items()} | {"_default": list(INFO_DEFAULT)}
    )
    script = (
        """<script>
const INFO = %s;
const doc = window.parent.document;
let panel = doc.getElementById('info-panel');
if (!panel) {
  panel = doc.createElement('div');
  panel.id = 'info-panel';
  doc.body.appendChild(panel);
}
function render(k) {
  const d = INFO[k] || INFO._default;
  const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;');
  panel.innerHTML = '<div class="ip-label">ABOUT THIS</div><h4>' +
    esc(d[0]) + '</h4><p>' + esc(d[1]) + '</p>';
}
render('_default');
if (window.parent.__tipHandler) {
  doc.removeEventListener('mouseover', window.parent.__tipHandler);
}
window.parent.__tipHandler = function (e) {
  const el = e.target.closest ? e.target.closest('[class*="st-key-tip_"]') : null;
  if (!el) return;
  const m = (el.getAttribute('class') || '').match(/st-key-tip_([A-Za-z0-9_]+)/);
  if (m) render(m[1]);
};
doc.addEventListener('mouseover', window.parent.__tipHandler);
</script>""" % payload
    )
    if hasattr(st, "iframe"):
        st.iframe(script, height=1)
    else:
        import streamlit.components.v1 as components
        components.html(script, height=0)


# 2. Data loading and reusable functions
@st.cache_data
def load_data(filename):
    path = PROJECT_DIR / filename
    if not path.exists():
        return None
    data = pd.read_csv(path, index_col=0, parse_dates=True)
    data = data.loc[:, ~data.columns.duplicated()]
    return data.sort_index()


def find_file(names):
    for name in names:
        if (PROJECT_DIR / name).is_file():
            return name
    return None


def heading(label, title, description):
    st.markdown(
        f'<div class="section-label">{label}</div>',
        unsafe_allow_html=True
    )
    st.header(title)
    st.caption(description)


def styled(chart, height=310):
    return (
        chart.properties(height=height, background="white")
        .configure_view(stroke=None)
        .configure_legend(labelColor="#475569", labelFontSize=12, symbolType="stroke")
        .configure_axis(
            labelColor="#64748B",
            titleColor="#475569",
            gridColor="#E8EDF5",
            domain=False,
            tickColor="#CBD5E1",
            labelFontSize=11,
            titleFontSize=12
        )
    )


def chart_data(data, columns):
    """Create exactly one Date column for Altair."""
    frame = data[columns].copy()
    frame.index.name = "Date"
    return frame.reset_index()


def choose_period(key, allow_all=False):
    options = ["30 Days", "90 Days", "180 Days", "1 Year"]
    if allow_all:
        options.append("All Data")
    with tip("period"):
        return st.selectbox("Time period", options, index=1, key=key)


def slice_period(data, period):
    count = {
        "30 Days": 30,
        "90 Days": 90,
        "180 Days": 180,
        "1 Year": 252
    }
    return (
        data.copy()
        if period == "All Data"
        else data.tail(count[period]).copy()
    )


def line_chart(data, columns, colors, height=310, tip_key="chart_price", y_title="Price (₹)"):
    """Display time-series data using the original chart format."""
    if not isinstance(data.index, pd.DatetimeIndex):
        st.warning("This chart requires a valid date index.")
        return

    columns = [c for c in columns if c in data.columns]
    if not columns:
        st.warning("No chart columns were found.")
        return

    long = chart_data(data, columns).melt(
        id_vars="Date",
        value_vars=columns,
        var_name="Series",
        value_name="Price"
    )
    long["Price"] = pd.to_numeric(long["Price"], errors="coerce")
    long = long.dropna(subset=["Price"])

    chart = alt.Chart(long).mark_line(strokeWidth=2.5).encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y(
            "Price:Q",
            title=y_title,
            scale=alt.Scale(zero=False)
        ),
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(domain=columns, range=colors),
            legend=alt.Legend(orient="bottom", title=None)
        ),
        strokeDash=alt.StrokeDash(
            "Series:N",
            scale=alt.Scale(
                domain=columns,
                range=[
                    [1, 0] if i < 2 else [6, 4]
                    for i in range(len(columns))
                ]
            ),
            legend=None
        ),
        tooltip=[
            alt.Tooltip("Date:T", format="%d %b %Y"),
            "Series:N",
            alt.Tooltip("Price:Q", format=",.2f")
        ]
    )
    with tip(tip_key):
        st.altair_chart(styled(chart, height), width="stretch")


stock = load_data("tcs_with_indicators.csv")

if stock is None or stock.empty or "Close" not in stock.columns:
    st.error(
        "Missing or invalid tcs_with_indicators.csv "
        "in the project folder."
    )
    st.stop()

if not isinstance(stock.index, pd.DatetimeIndex):
    st.error("The stock CSV must have dates in its first column.")
    st.stop()

stock["Close"] = pd.to_numeric(stock["Close"], errors="coerce")


# 3. Dashboard — original layout
def dashboard_page():
    heading(
        "MARKET OVERVIEW",
        "TCS Stock Dashboard",
        "Historical Tata Consultancy Services stock data."
    )

    latest = stock.iloc[-1]
    a, b, c = st.columns(3)
    with a, tip("close"):
        st.metric("Latest Historical Close", f"₹{latest['Close']:,.2f}")
    with b, tip("records"):
        st.metric("Historical Records", f"{len(stock):,}")
    with c, tip("symbol"):
        st.metric("Stock Symbol", "TCS.NS")

    st.write("")
    left, right = st.columns([3, 1])

    with left:
        st.subheader("Historical Closing Prices")
    with right:
        period = choose_period("dashboard_period", allow_all=True)

    line_chart(
        slice_period(stock, period),
        ["Close"],
        [RED],
        height=320
    )
    st.caption("Historical closing prices are not live market quotes.")

    st.divider()
    st.subheader("Recent Stock Data")
    with tip("table_recent"):
        st.dataframe(
            stock.tail(15).sort_index(ascending=False),
            width="stretch"
        )


# 4. Technical Indicators — original layout
def technical_indicators_page():
    heading(
        "TECHNICAL ANALYSIS",
        "Technical Indicator Analysis",
        "Historical price trends and market momentum."
    )

    required = ["Close", "SMA_20", "EMA_20", "RSI_14"]

    if not all(col in stock.columns for col in required):
        st.error("Some technical indicator columns are missing.")
        return

    valid = stock[required].dropna()
    if valid.empty:
        st.warning("No valid technical indicator data available.")
        return

    latest = valid.iloc[-1]
    cards = st.columns(4)
    labels = ["Closing Price", "20-Day SMA", "20-Day EMA", "RSI (14)"]
    values = [
        f"₹{latest['Close']:,.2f}",
        f"₹{latest['SMA_20']:,.2f}",
        f"₹{latest['EMA_20']:,.2f}",
        f"{latest['RSI_14']:.2f}"
    ]

    for card, label, value, key in zip(
        cards, labels, values, ["close", "sma", "ema", "rsi"]
    ):
        with card, tip(key):
            st.metric(label, value)

    st.write("")
    left, right = st.columns([3, 1])

    with left:
        st.subheader("Price & Moving Averages")
        st.caption("Closing price compared with SMA and EMA.")
    with right:
        period = choose_period("indicator_period")

    recent = slice_period(stock, period)

    line_chart(
        recent,
        ["Close", "SMA_20", "EMA_20"],
        [RED, GREEN, GOLD],
        tip_key="chart_ma"
    )

    st.divider()
    st.subheader("RSI Momentum")

    rsi = chart_data(
        recent.dropna(subset=["RSI_14"]),
        ["RSI_14"]
    )

    curve = alt.Chart(rsi).mark_line(
        color=GREEN,
        strokeWidth=2.5
    ).encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y(
            "RSI_14:Q",
            title="RSI",
            scale=alt.Scale(domain=[0, 100])
        ),
        tooltip=[
            alt.Tooltip("Date:T", format="%d %b %Y"),
            alt.Tooltip("RSI_14:Q", format=".2f")
        ]
    )

    rules = alt.Chart(
        pd.DataFrame({"Level": [30, 70]})
    ).mark_rule(
        color=RED,
        strokeDash=[5, 4]
    ).encode(y="Level:Q")

    with tip("chart_rsi"):
        st.altair_chart(
            styled(curve + rules, 190),
            width="stretch"
        )
    st.caption(
        "Dashed lines indicate RSI levels 30 and 70. "
        "Indicators do not guarantee future price movement."
    )

    if {"MACD", "MACD_Signal"}.issubset(stock.columns):
        st.divider()
        st.subheader("MACD")
        line_chart(
            recent,
            ["MACD", "MACD_Signal"],
            [RED, GOLD],
            tip_key="chart_macd",
            y_title="MACD"
        )

    if {"BB_Upper", "BB_Lower"}.issubset(stock.columns):
        st.divider()
        st.subheader("Bollinger Bands")
        line_chart(
            recent,
            ["Close", "BB_Upper", "BB_Lower"],
            [RED, GREEN, GOLD],
            tip_key="chart_bb"
        )

    st.divider()
    st.subheader("Recent Indicator Data")

    formats = {
        col: ("{:.2f}" if col == "RSI_14" else "₹{:,.2f}")
        for col in required
    }

    with tip("table_indicator"):
        st.dataframe(
            stock[required]
            .tail(10)
            .sort_index(ascending=False)
            .style.format(formats),
            width="stretch"
        )


# 5. Model Comparison
def model_comparison_page():
    heading(
        "MODEL EVALUATION",
        "Machine Learning Model Comparison",
        "Recorded results from historical TCS experiments."
    )

    # Previously recorded experiment figures.
    # Verify against the matching model files before final submission.
    comparison = pd.DataFrame({
        "Method": [
            "Random Forest",
            "Ridge Regression",
            "XGBoost",
            "LSTM",
            "No-Change Baseline"
        ],
        "MAE (₹)": [34.63, 34.91, 34.75, 73.26, 34.20],
        "RMSE (₹)": [49.36, 49.47, 49.24, 105.68, 48.65]
    })

    with tip("table_compare"):
        st.dataframe(
            comparison,
            hide_index=True,
            width="stretch"
        )

    st.subheader("Mean Absolute Error")

    bars = alt.Chart(comparison).mark_bar(
        color=GOLD,
        cornerRadiusEnd=5
    ).encode(
        x=alt.X("MAE (₹):Q", title="MAE (₹)"),
        y=alt.Y("Method:N", title=None, sort=None),
        tooltip=[
            "Method:N",
            alt.Tooltip("MAE (₹):Q", format=".2f")
        ]
    )

    with tip("chart_mae"):
        st.altair_chart(styled(bars, 280), width="stretch")

    st.info(
        "In these recorded experiments, the no-change baseline "
        "had lower MAE and RMSE than the four AI models."
    )

    st.subheader("Research Observation")
    st.write(
        "These experiments demonstrate why AI models should "
        "be evaluated against simple baselines when predicting "
        "financial time series."
    )


# 6. Prediction Results
def prediction_results_page():
    heading(
        "HISTORICAL TESTING",
        "Saved Prediction Results",
        "Explore predictions from the completed experiments."
    )

    files = {
        "Random Forest": [
            "improved_random_forest_results.csv",
            "random_forest_results.csv"
        ],
        "Ridge Regression": [
            "ridge_regression_results.csv"
        ],
        "XGBoost": [
            "xgboost_results.csv"
        ],
        "LSTM": [
            "lstm_results.csv"
        ]
    }

    with tip("model_select"):
        selected = st.selectbox("Choose a model", list(files))
    filename = find_file(files[selected])

    if filename is None:
        st.warning(
            "Results file not found. Expected one of: "
            + ", ".join(files[selected])
        )
        return

    results = load_data(filename)

    if results is None or results.empty:
        st.warning("The selected results file is empty.")
        return

    st.subheader(f"{selected}: Test Predictions")

    # Only display formatting changes:
    # Price columns show ₹ and two decimal places.
    # Other numeric columns show two decimal places.
    price_columns_for_format = [
        "Actual",
        "Predicted",
        "Baseline",
        "Current_Close"
    ]

    numeric_columns = results.select_dtypes(
        include="number"
    ).columns

    formats = {
        col: (
            "₹{:,.2f}"
            if col in price_columns_for_format
            else "{:.2f}"
        )
        for col in numeric_columns
    }

    with tip("table_pred"):
        st.dataframe(
            results.tail(20).style.format(formats),
            width="stretch"
        )

    # Original graph functionality
    numeric = results.select_dtypes(include="number")
    price_columns = [
        col for col in ["Actual", "Predicted", "Baseline"]
        if col in numeric.columns
    ]

    if (
        len(price_columns) >= 2
        and isinstance(results.index, pd.DatetimeIndex)
    ):
        st.subheader("Actual vs Predicted Prices")
        line_chart(
            numeric.tail(90),
            price_columns,
            [RED, GREEN, GOLD][:len(price_columns)],
            tip_key="chart_pred"
        )
    else:
        st.caption(
            "The saved results are displayed above. "
            "A dated CSV with actual and predicted price columns "
            "is needed for the comparison chart."
        )

    st.download_button(
        "Download Prediction Results",
        data=results.to_csv(),
        file_name=filename,
        mime="text/csv"
    )

    st.caption(
        "These are historical test predictions, "
        "not live market forecasts."
    )


# 7. AI Prediction
def ai_prediction_page():
    heading(
        "AI FORECASTING",
        "AI-Powered Stock Prediction",
        "Experimental next-day TCS predictions using four saved models."
    )

    st.info(
        "Predictions use the latest saved historical data, "
        "not live market prices."
    )

    with tip("predict_btn"):
        clicked = st.button("Generate AI Predictions", type="primary")

    if clicked:
        with st.spinner("Running saved prediction models..."):
            try:
                from prediction_engine import generate_prediction
                st.session_state["prediction"] = generate_prediction()
            except Exception as error:
                st.session_state["prediction"] = None
                st.error(f"Prediction failed: {error}")

    result = st.session_state.get("prediction")

    if result is None:
        return

    prediction_date = pd.Timestamp(result["date"])

    if prediction_date != pd.Timestamp(stock.index[-1]):
        st.warning(
            "Saved stock data has changed. "
            "Generate predictions again."
        )
        return

    st.success("Prediction generated successfully!")
    st.write(
        "Latest data date:",
        prediction_date.strftime("%d %B %Y")
    )

    left, right = st.columns(2)
    with left, tip("close"):
        st.metric(
            "Latest Historical Close",
            f"₹{result['latest_close']:,.2f}"
        )
    with right, tip("baseline"):
        st.metric(
            "No-Change Baseline",
            f"₹{result['baseline_close']:,.2f}"
        )

    predictions = result.get("predictions", {})

    if predictions:
        rows = [
            {
                "Model": model,
                "Predicted Close (₹)": values["predicted_close"],
                "Predicted Change (%)": values["predicted_return"]
            }
            for model, values in predictions.items()
        ]

        forecast = pd.DataFrame(rows)

        st.subheader("Next-Day Model Predictions")

        with tip("table_forecast"):
            st.dataframe(
                forecast.style.format({
                    "Predicted Close (₹)": "₹{:,.2f}",
                    "Predicted Change (%)": "{:+.2f}%"
                }),
                hide_index=True,
                width="stretch"
            )

        chart = alt.Chart(forecast).mark_bar(
            cornerRadiusEnd=5
        ).encode(
            x=alt.X("Model:N", title=None),
            y=alt.Y(
                "Predicted Close (₹):Q",
                scale=alt.Scale(zero=False)
            ),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(
                    domain=[
                        "Random Forest",
                        "Ridge Regression",
                        "XGBoost",
                        "LSTM"
                    ],
                    range=[RED, GREEN, GOLD, NAVY]
                ),
                legend=None
            ),
            tooltip=[
                "Model:N",
                alt.Tooltip("Predicted Close (₹):Q", format=",.2f"),
                alt.Tooltip("Predicted Change (%):Q", format="+.2f")
            ]
        )

        with tip("chart_models"):
            st.altair_chart(styled(chart, 300), width="stretch")

    for model, error in result.get("errors", {}).items():
        st.warning(f"{model}: {error}")

    st.caption(
        "These are experimental forecasts. "
        "Historical model errors are shown on the Model Comparison page."
    )


# 8. Original navigation and application layout
st.sidebar.title("AI-Based Stock Prediction Engine")
st.sidebar.caption("AI-powered financial analytics")
st.sidebar.divider()

pages = {
    "Dashboard": dashboard_page,
    "Technical Indicators": technical_indicators_page,
    "Model Comparison": model_comparison_page,
    "Prediction Results": prediction_results_page,
    "AI Prediction": ai_prediction_page
}

page = st.sidebar.radio("NAVIGATION", list(pages))
st.sidebar.divider()

if st.sidebar.button("Refresh Saved Data"):
    st.cache_data.clear()
    st.session_state.pop("prediction", None)
    st.rerun()

st.sidebar.info(
    "Academic Research Prototype\n\n"
    "Stock: TCS.NS\n\n"
    "Historical data and experimental predictions"
)

st.markdown("""
<div class="project-banner">
    <h1>AI-Based Stock Prediction Engine</h1>
    <p>Machine Learning & Deep Learning for Short-Term Stock Price Prediction</p>
</div>
""", unsafe_allow_html=True)

inject_info_panel()
pages[page]()

st.divider()
st.caption(
    "AI-Based Stock Prediction Engine | Academic Research Prototype | "
    "Historical Data & Experimental AI Predictions"
)