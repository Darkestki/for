# ==========================================
# 🚜 3D Professional Tractor Forecast App
# ==========================================

import streamlit as st
import pandas as pd
import pickle
import os

# ------------------------------------------
# 🎨 Page Configuration
# ------------------------------------------
st.set_page_config(
    page_title="Tractor Forecast Pro",
    page_icon="🚜",
    layout="centered"
)

# ------------------------------------------
# 💎 Dynamic Animated Light UI Styling
# ------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, 
        #f9fbfd, 
        #e3f2fd, 
        #e8f5e9, 
        #fff3e0,
        #fce4ec,
        #e1f5fe);
    background-size: 500% 500%;
    animation: gradientMove 15s ease infinite;
    color: #1a1a1a;
}

@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    25% {background-position: 50% 100%;}
    50% {background-position: 100% 50%;}
    75% {background-position: 50% 0%;}
    100% {background-position: 0% 50%;}
}

.glass-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(240,248,255,0.7));
    backdrop-filter: blur(12px);
    border-radius: 25px;
    padding: 25px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.08);
    text-align: center;
    margin-top: 20px;
    border: 1px solid rgba(0,0,0,0.05);
}

.stButton>button {
    background: linear-gradient(270deg, #4CAF50, #2196F3, #FF9800, #9C27B0);
    background-size: 600% 600%;
    animation: buttonFlow 6s ease infinite;
    color: white;
    border-radius: 12px;
    padding: 12px 28px;
    font-size: 17px;
    font-weight: bold;
    border: none;
}

@keyframes buttonFlow {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
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

    exp_path = os.path.join(base_dir, "exponential_smoothing_model.pkl")
    arima_path = os.path.join(base_dir, "arima_model.pkl")

    with open(exp_path, "rb") as f:
        exp_model = pickle.load(f)

    with open(arima_path, "rb") as f:
        arima_model = pickle.load(f)

    return exp_model, arima_model

df = load_data()
exp_model, arima_model = load_models()

# ------------------------------------------
# 🏷️ Title
# ------------------------------------------
st.markdown("<h1 style='text-align:center;'>🚜 Tractor Sales Forecast Pro</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>📊 Exponential vs ARIMA Model Comparison</h4>", unsafe_allow_html=True)

# ------------------------------------------
# 📅 Month-Year Selection
# ------------------------------------------
months = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

col1, col2 = st.columns(2)

with col1:
    selected_month = st.selectbox("📆 Select Month", months)

with col2:
    selected_year = st.number_input("📅 Select Year", min_value=2014, max_value=2025, value=2022)

# ------------------------------------------
# 🔮 Prediction Logic
# ------------------------------------------
if st.button("Get Forecast Comparison"):

    selected_date = pd.to_datetime(f"01-{selected_month}-{selected_year}")

    if selected_date in df.index:
        actual_value = df.loc[selected_date]["Number of Tractor Sold"]

        st.markdown(f"""
        <div class="glass-card">
            <h2>📊 Historical Sales</h2>
            <h1 style="font-size:48px;">🚜 {round(actual_value)} Units</h1>
            <h4>Actual Sales for {selected_month} {selected_year}</h4>
        </div>
        """, unsafe_allow_html=True)

    else:
        last_training_date = df.index[-1]

        months_diff = (selected_date.year - last_training_date.year) * 12 + \
                      (selected_date.month - last_training_date.month)

        if months_diff > 0:

            # Exponential Forecast
            exp_forecast = exp_model.forecast(months_diff)
            exp_value = exp_forecast.iloc[-1]

            # ARIMA Forecast
            arima_forecast = arima_model.forecast(steps=months_diff)
            arima_value = arima_forecast.iloc[-1]

            colA, colB = st.columns(2)

            with colA:
                st.markdown(f"""
                <div class="glass-card">
                    <h3>📈 Exponential Smoothing</h3>
                    <h1>🚜 {round(exp_value)} Units</h1>
                </div>
                """, unsafe_allow_html=True)

            with colB:
                st.markdown(f"""
                <div class="glass-card">
                    <h3>📊 ARIMA Model</h3>
                    <h1>🚜 {round(arima_value)} Units</h1>
                </div>
                """, unsafe_allow_html=True)

            difference = abs(exp_value - arima_value)

            st.markdown(f"""
            <div class="glass-card">
                <h3>📌 Model Difference</h3>
                <h2>{round(difference)} Units Difference</h2>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.warning("⚠ Data not available for selected period.")
