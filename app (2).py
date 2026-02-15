# =====================================
# 🚜 Tractor Sales Forecasting App
# =====================================

import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import os

# -------------------------------------
# 🎨 Page Configuration
# -------------------------------------
st.set_page_config(
    page_title="🚜 Tractor Sales Forecast",
    page_icon="🚜",
    layout="centered"
)

# -------------------------------------
# 📂 Load Dataset
# -------------------------------------
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "tractor_sales.csv")
    df = pd.read_csv(file_path)
    df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='%b-%y')
    df = df.set_index('Month-Year')
    return df

# -------------------------------------
# 🤖 Load Trained Model
# -------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "exponential_smoothing_model.pkl")
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

# Load data & model
df = load_data()
model = load_model()

# -------------------------------------
# 🏷️ App Title
# -------------------------------------
st.title("🚜 Tractor Sales Forecasting Dashboard")
st.markdown("### 📊 Predict Future Tractor Sales Month-Year Wise")
st.write("This app uses an Exponential Smoothing model to forecast future tractor sales based on historical data.")

# -------------------------------------
# 🎛 Sidebar Controls
# -------------------------------------
st.sidebar.header("⚙ Forecast Settings")

forecast_months = st.sidebar.slider(
    "📅 Select number of months to forecast:",
    min_value=1,
    max_value=36,
    value=12
)

# -------------------------------------
# 🔮 Generate Forecast
# -------------------------------------
forecast = model.forecast(forecast_months)

# Convert forecast index to Month-Year format
forecast.index = pd.to_datetime(forecast.index)
forecast_df = forecast.to_frame(name="Forecasted Sales")
forecast_df["Month-Year"] = forecast_df.index.strftime("%b-%Y")
forecast_df = forecast_df.reset_index(drop=True)

# -------------------------------------
# 📈 Plot Chart
# -------------------------------------
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
    title="🚜 Tractor Sales Forecast",
    xaxis_title="Month-Year",
    yaxis_title="Number of Tractors Sold",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------
# 📋 Forecast Table
# -------------------------------------
st.subheader("📅 Month-Year Wise Forecast Details")

st.dataframe(
    forecast_df[["Month-Year", "Forecasted Sales"]]
    .round(0),
    use_container_width=True
)

# -------------------------------------
# 📌 Footer
# -------------------------------------
st.markdown("---")
st.markdown("✅ Developed using Streamlit | 📊 Time Series Forecasting | 🤖 Exponential Smoothing Model")
