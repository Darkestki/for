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
    page_title="🚜 Tractor Forecast Pro",
    page_icon="🚜",
    layout="centered"
)

# ------------------------------------------
# 💎 3D Animated Glass UI Styling
# ------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #1e3c72, #2a5298, #16222A, #3A6073);
    background-size: 400% 400%;
    animation: gradient 10s ease infinite;
    color: white;
}

@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.glass-card {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    text-align: center;
    margin-top: 20px;
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
# 🤖 Load Model
# ------------------------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "exponential_smoothing_model.pkl")

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model

df = load_data()
model = load_model()

# ------------------------------------------
# 🏷️ Title
# ------------------------------------------
st.markdown("<h1 style='text-align:center;'>🚜 Tractor Sales Forecast Pro</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>📅 Forecast Between 2014 – 2025</h4>", unsafe_allow_html=True)

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
if st.button("🔮 Get Result"):

    selected_date = pd.to_datetime(f"01-{selected_month}-{selected_year}")

    # If date exists in historical data
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
        # Forecast future value
        last_training_date = df.index[-1]

        months_diff = (selected_date.year - last_training_date.year) * 12 + \
                      (selected_date.month - last_training_date.month)

        if months_diff > 0:
            forecast = model.forecast(months_diff)
            predicted_value = forecast.iloc[-1]

            st.markdown(f"""
            <div class="glass-card">
                <h2>📊 Forecasted Sales</h2>
                <h1 style="font-size:48px;">🚜 {round(predicted_value)} Units</h1>
                <h4>Predicted Sales for {selected_month} {selected_year}</h4>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠ Data not available for selected period.")

# ------------------------------------------
# Footer
# ------------------------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<center>✨ Professional Forecast Dashboard | 2014 – 2025</center>", unsafe_allow_html=True)
