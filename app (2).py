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

/* Animated Gradient Background */
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

/* Glass Card */
.glass-card {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    text-align: center;
    margin-top: 20px;
    transition: transform 0.3s ease;
}

.glass-card:hover {
    transform: scale(1.03);
}

/* Button Styling */
.stButton>button {
    background: linear-gradient(90deg, #ff8a00, #e52e71);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    padding: 10px 25px;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #e52e71, #ff8a00);
    transform: scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# 🤖 Load Model
# ------------------------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "exponential_smoothing_model.pkl")

    if not os.path.exists(model_path):
        st.error("❌ Model file not found.")
        st.stop()

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model

model = load_model()

# ------------------------------------------
# 🏷️ Title Section
# ------------------------------------------
st.markdown("<h1 style='text-align:center;'>🚜 Tractor Sales Forecast Pro</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>📅 Select Month & Year to Predict Sales</h4>", unsafe_allow_html=True)

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
    selected_year = st.number_input("📅 Select Year", min_value=2020, max_value=2035, value=2025)

# ------------------------------------------
# 🔮 Prediction
# ------------------------------------------
if st.button("🔮 Predict Sales"):

    selected_date = pd.to_datetime(f"01-{selected_month}-{selected_year}")
    last_training_date = model.data.dates[-1]

    months_diff = (selected_date.year - last_training_date.year) * 12 + \
                  (selected_date.month - last_training_date.month)

    if months_diff <= 0:
        st.warning("⚠ Please select a future Month-Year.")
    else:
        forecast = model.forecast(months_diff)
        predicted_value = forecast.iloc[-1]

        st.markdown(f"""
        <div class="glass-card">
            <h2>📊 Forecast Result</h2>
            <h1 style="font-size:48px;">🚜 {round(predicted_value)} Units</h1>
            <h4>Predicted Sales for {selected_month} {selected_year}</h4>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------
# Footer
# ------------------------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<center>✨ Powered by Exponential Smoothing | Professional Forecast Dashboard</center>", unsafe_allow_html=True)
