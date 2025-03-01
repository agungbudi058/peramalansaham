import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go  # Import Plotly's graph_objs module

# Page title
st.title("Time Series Forecasting with Prophet and ARIMA")

# File upload
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Read the Excel file
    df = pd.read_excel(uploaded_file)
    st.write("### Raw Data")
    st.write(df)

    # Check if the required columns exist
    if 'ds' not in df.columns or 'y' not in df.columns:
        st.error("The Excel file must contain 'ds' (date) and 'y' (value) columns.")
    else:
        # Convert 'ds' to datetime
        df['ds'] = pd.to_datetime(df['ds'])

        # Model selection
        model_type = st.selectbox("Select a forecasting model", ["Prophet", "ARIMA"])

        # Forecast periods
        periods = st.number_input("Enter the number of periods to forecast", min_value=1, value=60)

        if model_type == "Prophet":
            # Prophet model
            st.write("### Prophet Forecast")
            model = Prophet()
            model.fit(df)
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

            # Plot the forecast
            fig = plot_plotly(model, forecast)
            st.plotly_chart(fig)

        elif model_type == "ARIMA":
            # ARIMA model
            st.write("### ARIMA Forecast")
            model = ARIMA(df['y'], order=(2, 1, 2))  # Example order (p, d, q)
            results = model.fit()
            forecast = results.forecast(steps=periods)

            # Create a DataFrame for the forecast
            last_date = df['ds'].iloc[-1]
            forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq='D')[1:]
            forecast_df = pd.DataFrame({'ds': forecast_dates, 'y': forecast})

            st.write(forecast_df)

            # Plot the forecast
            plt.figure(figsize=(10, 6))
            plt.plot(df['ds'], df['y'], label="Actual Data")
            plt.plot(forecast_df['ds'], forecast_df['y'], label="Forecast", linestyle="--")
            plt.legend()
            plt.title("ARIMA Forecast")
            st.pyplot(plt)
