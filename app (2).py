
import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from statsmodels.tsa.api import ExponentialSmoothing

# --- Load Data and Model ---
@st.cache_data
def load_data():
    # Load the original dataset to display historical data
    df = pd.read_csv('Tractor-Sales - Tractor-Sales.csv')
    df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='%b-%y')
    df = df.set_index('Month-Year')
    return df

@st.cache_resource
def load_model():
    # Load the pre-trained Exponential Smoothing model
    with open('exponential_smoothing_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

df1 = load_data()
exp_smoothing_model = load_model()

# --- Streamlit Application ---
st.set_page_config(page_title="Tractor Sales Forecast App", layout="centered")

st.title("🚜 Tractor Sales Forecasting App")
st.write("This application forecasts tractor sales using an Exponential Smoothing model.")
st.write("The model was trained on historical data and is now used to predict future sales.")

# User Input for Forecast Horizon
st.sidebar.header("Forecast Settings")
forecast_months = st.sidebar.slider(
    "Number of months to forecast:",
    min_value=1,
    max_value=36,
    value=12,
    step=1
)

# Generate Forecast
# The model's forecast method automatically extends the index
forecast = exp_smoothing_model.forecast(forecast_months)

# --- Visualization with Plotly ---
fig = go.Figure()

# Add original data trace
fig.add_trace(go.Scatter(
    x=df1.index,
    y=df1['Number of Tractor Sold'],
    mode='lines',
    name='Historical Sales',
    line=dict(color='blue')
))

# Add forecast trace
fig.add_trace(go.Scatter(
    x=forecast.index,
    y=forecast,
    mode='lines',
    name=f'Forecasted Sales ({forecast_months} months)',
    line=dict(color='red', dash='dot')
))

# Update layout for title, labels, and legend
fig.update_layout(
    title='Tractor Sales: Historical Data vs. Forecast',
    xaxis_title='Date',
    yaxis_title='Number of Tractors Sold',
    hovermode='x unified',
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.5)', borderwidth=1)
)

# Display the Plotly chart in Streamlit
st.plotly_chart(fig, use_container_width=True)

st.sidebar.info("Adjust the slider to change the forecast horizon.")

# Display forecast in a table (optional)
if st.sidebar.checkbox("Show Forecast Details"):
    st.subheader("Forecasted Sales Details")
    st.dataframe(forecast.to_frame(name='Forecasted Sales').round(0))
