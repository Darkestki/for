# ==========================================
# 🚜 Tractor Sales Forecasting Dashboard
# ==========================================

import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import os

# ------------------------------------------
# 🎨 Page Configuration
# ------------------------------------------
st.set_page_config(
    page_title="🚜 Tractor Sales Forecast",
    page_icon="🚜",
    layout="centered"
)

# ------------------------------------------
# 📂 Load Dataset (Cloud Safe)
# ------------------------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "tractor_sales.csv")

    if not os.path.exists(file_path):
        st.error("❌ CSV file not found. Please upload 'tractor_sales.csv' in same folder.")
        st.stop()

    df = pd.read_csv(file_path)
    df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='%b-%y')
    df = df.set_index('Month-Year')
    return df


# ------------------------------------------
# 🤖 Load Trained Model
# ------------------------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "exponential_smoothing_model.pkl")

    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please upload 'exponential_smoothing_model.pkl'.")
        st.stop()

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model


# Load Data & Model
df = load_data()
model = load_model()

# ------------------------------------------
# 🏷️ Title Section
# ------------------------------------------
st.title("🚜 Tractor Sales Forecasting App")
st.markdown("### 📊 Month-Year Wise Sales Prediction")
st.write("This dashboard forecasts future tractor sales using an Exponential Smoothing model.")

# ------------------------------------------
# 🎛 Sidebar Forecast Control
# ------------------------------------------
st.sidebar.header("⚙ Forecast Settings")

forecast_months = st.sidebar.slider(
    "📅 Select number of months to forecast:",
    min_value=1,
    max_value=36,
    value=12
)

# ------------------------------------------
# 🔮 Generate Forecast
# ------------------------------------------
forecast = model.forecast(forecast_months)
forecast.index = pd.to_datetime(forecast.index)

# Create Forecast DataFrame
forecast_df = forecast.to_frame(name="Forecasted Sales")
forecast_df["Month-Year"] = forecast_df.index.strftime("%b-%Y")
forecast_df = forecast_df.reset_index(drop=True)

# ------------------------------------------
# 📈 Visualization
# ------------------------------------------
fig = go.Figure()

# Historical Data
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["Number of Tractor Sold"],
    mode="lines",
    name="📘 Historical Sales"
))

# Forecast Data
fig.add_trace(go.Scatter(
    x=forecast.index,
    y=forecast,
    mode="lines",
    name="🔴 Forecasted Sales",
    line=dict(dash="dot")
))

fig.update_layout(
    title="🚜 Tractor Sales: Historical vs Forecast",
    xaxis_title="Month-Year",
    yaxis_title="Number of Tractors Sold",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# 📋 Forecast Table
# ------------------------------------------
st.subheader("📅 Forecast Details (Month-Year Wise)")
st.dataframe(
    forecast_df[["Month-Year", "Forecasted Sales"]].round(0),
    use_container_width=True
)

# ------------------------------------------
# 📌 Footer
# ------------------------------------------
st.markdown("---")
st.markdown("✅ Built with Streamlit | 📈 Time Series Forecasting | 🤖 Exponential Smoothing")
