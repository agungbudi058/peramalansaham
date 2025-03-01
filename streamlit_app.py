import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

# Page title
st.title("Time Series Forecasting with Prophet")

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

        # Forecast periods
        periods = st.number_input("Enter the number of periods to forecast", min_value=1, value=60)

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

        # Perform analysis
        st.write("### Analysis Results")

        # 1. What day has the highest and lowest average value (weekly)?
        df['day_of_week'] = df['ds'].dt.day_name()
        day_avg = df.groupby('day_of_week')['y'].mean().reset_index()
        highest_day = day_avg.loc[day_avg['y'].idxmax()]
        lowest_day = day_avg.loc[day_avg['y'].idxmin()]
        st.write(f"1. Day with the **highest** average value: **{highest_day['day_of_week']}** (Average: **{highest_day['y']:.2f}**)")
        st.write(f"2. Day with the **lowest** average value: **{lowest_day['day_of_week']}** (Average: **{lowest_day['y']:.2f}**)")

        # 2. What month has the highest and lowest average value in the last year?
        df['month'] = df['ds'].dt.month_name()
        last_year_data = df[df['ds'] >= (df['ds'].max() - pd.DateOffset(years=1))]
        month_avg = last_year_data.groupby('month')['y'].mean().reset_index()
        highest_month = month_avg.loc[month_avg['y'].idxmax()]
        lowest_month = month_avg.loc[month_avg['y'].idxmin()]
        st.write(f"3. Month with the **highest** average value in the last year: **{highest_month['month']}** (Average: **{highest_month['y']:.2f}**)")
        st.write(f"4. Month with the **lowest** average value in the last year: **{lowest_month['month']}** (Average: **{lowest_month['y']:.2f}**)")

        # 3. Trend last year: up or down?
        first_value = last_year_data['y'].iloc[0]
        last_value = last_year_data['y'].iloc[-1]
        trend = "up" if last_value > first_value else "down"
        st.write(f"5. Trend over the last year: **{trend}**")

        # 4. Complete Analysis Report
        st.write("### Complete Analysis Report")
        st.write(f"""
        **1. Day with the Highest Average Value:**
        - The day of the week with the highest average value is **{highest_day['day_of_week']}**, with an average value of **{highest_day['y']:.2f}**.

        **2. Day with the Lowest Average Value:**
        - The day of the week with the lowest average value is **{lowest_day['day_of_week']}**, with an average value of **{lowest_day['y']:.2f}**.

        **3. Month with the Highest Average Value in the Last Year:**
        - The month with the highest average value in the last year is **{highest_month['month']}**, with an average value of **{highest_month['y']:.2f}**.

        **4. Month with the Lowest Average Value in the Last Year:**
        - The month with the lowest average value in the last year is **{lowest_month['month']}**, with an average value of **{lowest_month['y']:.2f}**.

        **5. Trend Over the Last Year:**
        - The trend over the last year is **{trend}**.

        **6. Forecasting Model:**
        - The selected forecasting model is **Prophet**.
        - The forecasted values for the next **{periods}** periods are displayed above.
        """)
