# ==========================================
# 🚜 Tractor Forecast Pro – No Scroll Version
# ==========================================

import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# ------------------------------------------
# 🎨 Page Config
# ------------------------------------------
st.set_page_config(
    page_title="Tractor Forecast Pro",
    page_icon="🚜",
    layout="wide"
)

# ------------------------------------------
# 💎 Ultra Compact Styling
# ------------------------------------------
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-size: 14px;
}

.stApp {
    background: #f9fbff;
}

.main-title {
    text-align:center;
    font-size:28px;
    font-weight:700;
    color:#1f3c88;
    margin-bottom:5px;
}

.sub-title {
    text-align:center;
    font-size:14px;
    color:#666;
    margin-bottom:15px;
}

/* Compact cards */
.card {
    background:white;
    padding:12px;
    border-radius:10px;
    box-shadow:0 3px 8px rgba(0,0,0,0.05);
    text-align:center;
}

.metric-good {
    border-left:4px solid #2ca02c;
}

.metric-bad {
    border-left:4px solid #d62728;
}

/* Remove extra padding */
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
}

/* Smaller button */
.stButton>button {
    padding:6px 14px;
    font-size:13px;
    border-radius:6px;
    background:#1f77b4;
    color:white;
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
# 📊 Metrics
# ------------------------------------------
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred):
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
st.markdown('<div class="main-title">🚜 Tractor Forecast Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Exponential Smoothing vs SARIMAX</div>', unsafe_allow_html=True)

# ==========================================
# 📊 TOP SECTION (Everything in One Row)
# ==========================================

col1, col2, col3 = st.columns([1,1,1])

y_true = df["Number of Tractor Sold"]

exp_pred = exp_model.fittedvalues
arima_pred = arima_model.fittedvalues

min_len = min(len(y_true), len(exp_pred), len(arima_pred))

y = y_true[-min_len:]
exp_pred = exp_pred[-min_len:]
arima_pred = arima_pred[-min_len:]

exp_rmse = rmse(y, exp_pred)
arima_rmse = rmse(y, arima_pred)

exp_mae = mae(y, exp_pred)
arima_mae = mae(y, arima_pred)

exp_mape = mape(y, exp_pred)
arima_mape = mape(y, arima_pred)

metric_choice = col1.selectbox(
    "Metric",
    ["RMSE", "MAE", "MAPE"]
)

if metric_choice == "RMSE":
    better_model = "Exponential" if exp_rmse < arima_rmse else "SARIMAX"
elif metric_choice == "MAE":
    better_model = "Exponential" if exp_mae < arima_mae else "SARIMAX"
else:
    better_model = "Exponential" if exp_mape < arima_mape else "SARIMAX"

with col2:
    st.markdown(f"""
    <div class="card {'metric-good' if better_model=='Exponential' else 'metric-bad'}">
    <b>Exponential</b><br>
    RMSE: {round(exp_rmse,1)}<br>
    MAE: {round(exp_mae,1)}<br>
    MAPE: {round(exp_mape,1)}%
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card {'metric-good' if better_model=='SARIMAX' else 'metric-bad'}">
    <b>SARIMAX</b><br>
    RMSE: {round(arima_rmse,1)}<br>
    MAE: {round(arima_mae,1)}<br>
    MAPE: {round(arima_mape,1)}%
    </div>
    """, unsafe_allow_html=True)

st.success(f"🏆 Best Model ({metric_choice}): {better_model}")

# ==========================================
# 🔮 FORECAST SECTION (Same Screen)
# ==========================================

colA, colB, colC = st.columns([1,1,1])

selected_month = colA.selectbox("Month",
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
)

selected_year = colB.number_input("Year", 2014, 2030, 2024)

if colC.button("Forecast"):

    selected_date = pd.to_datetime(f"01-{selected_month}-{selected_year}", format="%d-%b-%Y")
    last_date = df.index[-1]

    months_diff = (selected_date.year - last_date.year) * 12 + \
                  (selected_date.month - last_date.month)

    if months_diff > 0:
        exp_forecast = exp_model.forecast(months_diff).iloc[-1]
        arima_forecast = arima_model.forecast(steps=months_diff).iloc[-1]

        colX, colY = st.columns(2)

        colX.metric("Exponential Forecast", round(exp_forecast))
        colY.metric("SARIMAX Forecast", round(arima_forecast))
    else:
        st.warning("Select a future date.")
