# ==========================================
# 🚜 Tractor Forecast Pro – Dynamic Version
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
# 💎 Premium Dynamic UI Styling
# ------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#f4f9ff,#eef7f1,#ffffff);
}

.main-title {
    text-align:center;
    font-size:42px;
    font-weight:700;
    color:#1f3c88;
}

.sub-title {
    text-align:center;
    font-size:18px;
    color:#555;
    margin-bottom:40px;
}

.card {
    background:white;
    padding:25px;
    border-radius:18px;
    box-shadow:0 8px 20px rgba(0,0,0,0.08);
    text-align:center;
    transition:0.3s ease;
}

.card:hover {
    transform:translateY(-5px);
}

.metric-good {
    border-left:6px solid #2ca02c;
}

.metric-bad {
    border-left:6px solid #d62728;
}

.stButton>button {
    background: linear-gradient(90deg,#1f77b4,#2ca02c);
    color:white;
    border-radius:10px;
    padding:12px 25px;
    font-weight:bold;
    font-size:16px;
    transition:0.3s ease;
}

.stButton>button:hover {
    transform:scale(1.05);
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
# 📊 Error Metric Functions (Manual)
# ------------------------------------------
def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# ------------------------------------------
# 🚀 Load Everything
# ------------------------------------------
df = load_data()
exp_model, arima_model = load_models()

# ------------------------------------------
# 🏷️ Header Section
# ------------------------------------------
st.markdown('<div class="main-title">🚜 Tractor Sales Forecast Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Dynamic Model Comparison: Exponential Smoothing vs SARIMAX</div>', unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 📉 SECTION 1: Model Performance Metrics
# ==========================================

st.subheader("Model Performance (Training Data)")

y_true = df["Number of Tractor Sold"]

exp_pred = exp_model.fittedvalues
arima_pred = arima_model.fittedvalues

min_len_exp = min(len(y_true), len(exp_pred))
min_len_arima = min(len(y_true), len(arima_pred))

exp_mae = calculate_mae(y_true[-min_len_exp:], exp_pred[-min_len_exp:])
exp_rmse = calculate_rmse(y_true[-min_len_exp:], exp_pred[-min_len_exp:])

arima_mae = calculate_mae(y_true[-min_len_arima:], arima_pred[-min_len_arima:])
arima_rmse = calculate_rmse(y_true[-min_len_arima:], arima_pred[-min_len_arima:])

better_model = "Exponential Smoothing" if exp_rmse < arima_rmse else "ARIMA"

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="card {'metric-good' if exp_rmse < arima_rmse else 'metric-bad'}">
        <h3>📈 Exponential Smoothing</h3>
        <p><b>MAE:</b> {round(exp_mae,2)}</p>
        <p><b>RMSE:</b> {round(exp_rmse,2)}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card {'metric-good' if arima_rmse < exp_rmse else 'metric-bad'}">
        <h3>📊 SARIMAX Model</h3>
        <p><b>MAE:</b> {round(arima_mae,2)}</p>
        <p><b>RMSE:</b> {round(arima_rmse,2)}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class="card" style="margin-top:25px;">
    <h3>🏆 Best Performing Model</h3>
    <h2 style="color:#2ca02c;">{better_model}</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 🔮 SECTION 2: Forecast Comparison
# ==========================================

st.subheader("🔮 Future Forecast Comparison")

months = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

colA, colB = st.columns(2)

with colA:
    selected_month = st.selectbox("Select Month", months)

with colB:
    selected_year = st.number_input("Select Year", min_value=2014, max_value=2025, value=2024)

if st.button("Compare Forecast"):

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
                <h3>📈 Exponential Forecast</h3>
                <h1>{round(exp_forecast)}</h1>
            </div>
            """, unsafe_allow_html=True)

        with colY:
            st.markdown(f"""
            <div class="card">
                <h3>📊 SARIMAX Forecast</h3>
                <h1>{round(arima_forecast)}</h1>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("Please select a future date for forecasting.")
