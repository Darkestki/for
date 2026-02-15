# ==========================================
# 🚜 Tractor Sales Month-Year Forecast App
# ==========================================

import streamlit as st
import pandas as pd
import pickle
import os

# ------------------------------------------
# 🎨 Page Configuration
# ------------------------------------------
st.set_page_config(
    page_title="🚜 Tractor Forecast",
    page_icon="🚜",
    layout="centered"
)

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
# 🏷️ Title
# ------------------------------------------
st.title("🚜 Tractor Sales Forecast")
st.markdown("### 📅 Predict Sales for Selected Month & Year")

# ------------------------------------------
# 📅 Month-Year Selection
# ------------------------------------------
months = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

col1, col2 = st.columns(2)

with col1:
    selected_month = st.selectbox("Select Month", months)

with col2:
    selected_year = st.number_input("Select Year", min_value=2020, max_value=2035, value=2025)

# ------------------------------------------
# 🔮 Predict Button
# ------------------------------------------
if st.button("🔮 Predict Sales"):

    # Convert selected date to pandas datetime
    selected_date = pd.to_datetime(f"01-{selected_month}-{selected_year}")

    # Calculate number of months from last training date
    last_training_date = model.data.dates[-1]
    months_diff = (selected_date.year - last_training_date.year) * 12 + \
                  (selected_date.month - last_training_date.month)

    if months_diff <= 0:
        st.warning("⚠ Please select a future month-year.")
    else:
        forecast = model.forecast(months_diff)
        predicted_value = forecast.iloc[-1]

        st.success("✅ Forecast Generated Successfully!")

        st.markdown("## 📊 Forecast Result")
        st.metric(
            label=f"🚜 Predicted Sales for {selected_month} {selected_year}",
            value=f"{round(predicted_value)} Units"
        )

# ------------------------------------------
# Footer
# ------------------------------------------
st.markdown("---")
st.markdown("📈 Powered by Exponential Smoothing Model")
