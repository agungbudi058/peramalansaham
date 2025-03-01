import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima  # For auto ARIMA
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

# Page title
st.title("Time Series Forecasting with Prophet and Auto ARIMA")

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
        model_type = st.selectbox("Select a forecasting model", ["Prophet", "Auto ARIMA"])

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

        elif model_type == "Auto ARIMA":
            # Auto ARIMA model
            st.write("### Auto ARIMA Forecast")
            
            # Automatically find the best ARIMA parameters
            with st.spinner("Searching for best ARIMA parameters..."):
                arima_model = auto_arima(
                    df['y'],
                    seasonal=False,  # Disable SARIMA (set seasonal=True for SARIMA)
                    trace=True,
                    error_action='ignore',
                    suppress_warnings=True
                )
            
            st.write(f"**Best ARIMA Order (p, d, q):** {arima_model.order}")
            
            # Fit the model
            arima_model.fit(df['y'])
            
            # Forecast
            forecast = arima_model.predict(n_periods=periods)
            
            # Create forecast dates
            last_date = df['ds'].iloc[-1]
            forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq='D')[1:]
            forecast_df = pd.DataFrame({'ds': forecast_dates, 'y': forecast})
            
            # Plot results
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['ds'], df['y'], label="Actual Data")
            ax.plot(forecast_df['ds'], forecast_df['y'], label="Forecast", linestyle="--")
            ax.set_title("Auto ARIMA Forecast")
            ax.legend()
            st.pyplot(fig)

        # =========================================================================
        # Analysis Section
        # =========================================================================
        st.write("### Advanced Analysis")
        
        # 1. Weekly Analysis
        df['day_of_week'] = df['ds'].dt.day_name()
        day_avg = df.groupby('day_of_week')['y'].mean().reset_index()
        highest_day = day_avg.loc[day_avg['y'].idxmax()]
        lowest_day = day_avg.loc[day_avg['y'].idxmin()]
        
        # 2. Monthly Analysis (Last Year)
        df['month'] = df['ds'].dt.month_name()
        last_year_data = df[df['ds'] >= (df['ds'].max() - pd.DateOffset(years=1))]
        month_avg = last_year_data.groupby('month')['y'].mean().reset_index()
        highest_month = month_avg.loc[month_avg['y'].idxmax()]
        lowest_month = month_avg.loc[month_avg['y'].idxmin()]
        
        # 3. Trend Analysis
        first_value = last_year_data['y'].iloc[0]
        last_value = last_year_data['y'].iloc[-1]
        trend = "up" if last_value > first_value else "down"
        
        # =========================================================================
        # Complete Analysis Report
        # =========================================================================
        st.write("### Complete Analysis Report")
        
        st.write(f"""
        **Temporal Patterns Analysis**
        
        1. **Weekly Performance:**
           - Peak day: {highest_day['day_of_week']} (Avg: {highest_day['y']:.2f})
           - Lowest day: {lowest_day['day_of_week']} (Avg: {lowest_day['y']:.2f})
        
        2. **Monthly Performance (Last 12 Months):**
           - Strongest month: {highest_month['month']} (Avg: {highest_month['y']:.2f})
           - Weakest month: {lowest_month['month']} (Avg: {lowest_month['y']:.2f})
        
        3. **Trend Analysis:**
           - Overall trend direction: {trend.capitalize()}ward
           - First value: {first_value:.2f}
           - Last value: {last_value:.2f}
           - Change: {(last_value - first_value):.2f} ({abs((last_value - first_value)/first_value*100):.2f}%)
        
        **Model Implementation**
        4. **Forecasting Model:**
           - Selected model: {model_type}
           - Forecast horizon: {periods} periods
           - ARIMA order (p,d,q): {arima_model.order if model_type == "Auto ARIMA" else "N/A"}
        
        **Recommendations**
        5. **Strategic Suggestions:**
           - Capitalize on {highest_day['day_of_week']} performance through targeted promotions
           - Investigate root causes for {lowest_month['month']} underperformance
           - Allocate resources for {trend}ward trend continuation
        """)

        # =========================================================================
        # Raw Data Download
        # =========================================================================
        st.write("### Data Export")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Processed Data (CSV)",
            data=csv,
            file_name="processed_forecast_data.csv",
            mime="text/csv"
        )
