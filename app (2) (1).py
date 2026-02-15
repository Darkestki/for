# ==========================================
# 🚜 Tractor Forecast Pro – Compact Dynamic Version
# ==========================================

import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# ------------------------------------------
# 🎨 Page Configuration
# ------------------------------------------
st.set_page_config(
    page_title="Tractor Forecast Pro",
    page_icon="🚜",
    layout="wide"
)

# ------------------------------------------
# 💎 Compact Professional UI Styling
# ------------------------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#f4f9ff,#eef7f1,#ffffff);
}

/* Title */
.main-title {
    text-align:center;
    font-size:36px;
    font-weight:700;
    color:#1f3c88;
}

.sub-title {
    text-align:center;
    font-size:16px;
    color:#555;
    margin-bottom:30px;
}

/* Smaller Cards */
.card {
    background:white;
    padding:15px;
    border-radius:12px;
    box-shadow:0 4px 12px rgba(0,0,0,0.08);
    text-align:center;
    transition:0.3s ease;
    margin-bottom:10px;
}

.card h3 {
    font-size:18px;
    margin-bottom:8px;
}

.card p {
    font-size:14px;
    margin:4px 0;
}

.metric-good {
    border-left:5px solid #2ca02c;
}

.metric-bad {
    border-left:5px solid #d62728;
}

.best-card {
    background:#f8fff8;
    padding:18px;
    border-radius:12px;
    text-align:center;
    border:2px solid #2ca02c;
    margin-top:15px;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg,#1f77b4,#2ca02c);
    color:white;
    border-radius:8px;
    padding:8px 18px;
    font-weight:bold;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# 📂 Load Data
# ------------------------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "Tractor-Sales - Tractor-Sales.csv")

    df = pd.read_csv(file_path)
    df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='%b-%y')
    df = df.set_index('Month-Year')
    return df

# ------------------------------------------
# 🤖 Load Models
# ------------------------------------------
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(base_dir,"exponential_smoothing_model.pkl"),"rb") as f:
        exp_model = pickle.load(f)

    with open(os.path.join(base_dir,"arima_model.pkl"),"rb") as f:
        arima_model = pickle.load(f)

    return exp_model, arima_model

# ------------------------------------------
# 📊 Error Metrics
# ------------------------------------------
def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ------------------------------------------
# 🚀 Load Everything
# ------------------------------------------
df = load_data()
exp_model, arima_model = load_models()

# ------------------------------------------
# 🏷️ Header
# ------------------------------------------
st.markdown('<div class="main-title">🚜 Tractor Sales Forecast Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Dynamic Model Comparison: Exponential Smoothing vs SARIMAX</div>', unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 📉 SECTION 1: Model Performance
# ==========================================

st.subheader("📉 Model Performance")

y_true = df["Number of Tractor Sold"]

exp_pred = exp_model.fittedvalues
arima_pred = arima_model.fittedvalues

min_len_exp = min(len(y_true), len(exp_pred))
min_len_arima = min(len(y_true), len(arima_pred))

y_exp = y_true[-min_len_exp:]
y_arima = y_true[-min_len_arima:]

exp_pred = exp_pred[-min_len_exp:]
arima_pred = arima_pred[-min_len_arima:]

# Metrics
exp_mae = calculate_mae(y_exp, exp_pred)
exp_rmse = calculate_rmse(y_exp, exp_pred)
exp_mape = calculate_mape(y_exp, exp_pred)

arima_mae = calculate_mae(y_arima, arima_pred)
arima_rmse = calculate_rmse(y_arima, arima_pred)
arima_mape = calculate_mape(y_arima, arima_pred)

# Dynamic Metric Selector
metric_choice = st.selectbox(
    "Select Metric to Decide Best Model",
    ["RMSE", "MAE", "MAPE"]
)

if metric_choice == "RMSE":
    better_model = "Exponential Smoothing" if exp_rmse < arima_rmse else "SARIMAX"
elif metric_choice == "MAE":
    better_model = "Exponential Smoothing" if exp_mae < arima_mae else "SARIMAX"
else:
    better_model = "Exponential Smoothing" if exp_mape < arima_mape else "SARIMAX"

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="card {'metric-good' if better_model == 'Exponential Smoothing' else 'metric-bad'}">
        <h3>📈 Exponential</h3>
        <p>MAE: {round(exp_mae,2)}</p>
        <p>RMSE: {round(exp_rmse,2)}</p>
        <p>MAPE: {round(exp_mape,2)}%</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card {'metric-good' if better_model == 'SARIMAX' else 'metric-bad'}">
        <h3>📊 SARIMAX</h3>
        <p>MAE: {round(arima_mae,2)}</p>
        <p>RMSE: {round(arima_rmse,2)}</p>
        <p>MAPE: {round(arima_mape,2)}%</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class="best-card">
    <h4>🏆 Best Model (Based on {metric_choice})</h4>
    <h2>{better_model}</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 🔮 SECTION 2: Forecast Comparison
# ==========================================

st.subheader("🔮 Future Forecast")

colA, colB = st.columns(2)

with colA:
    selected_month = st.selectbox("Select Month",
        ["January","February","March","April","May","June",
         "July","August","September","October","November","December"]
    )

with colB:
    selected_year = st.number_input("Select Year", 2014, 2030, 2024)

if st.button("Generate Forecast"):

    selected_date = pd.to_datetime(f"01-{selected_month}-{selected_year}")
    last_date = df.index[-1]

    months_diff = (selected_date.year - last_date.year) * 12 + \
                  (selected_date.month - last_date.month)

    if months_diff > 0:

        exp_forecast = exp_model.forecast(months_diff).iloc[-1]
        arima_forecast = arima_model.forecast(steps=months_diff).iloc[-1]

        colX, colY = st.columns(2)

        with colX:
            st.markdown(f"""
            <div class="card">
                <h3>📈 Exponential</h3>
                <h2>{round(exp_forecast)}</h2>
            </div>
            """, unsafe_allow_html=True)

        with colY:
            st.markdown(f"""
            <div class="card">
                <h3>📊 SARIMAX</h3>
                <h2>{round(arima_forecast)}</h2>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("⚠ Please select a future date.")
