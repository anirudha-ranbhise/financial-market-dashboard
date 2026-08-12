import streamlit as st
import pandas as pd
from fetch_data import get_live_data

# 1. Page Configuration (Makes it look wide and professional)
st.set_page_config(page_title="Market Dashboard", layout="wide")
st.title("📈 Financial Market Dashboard")
st.markdown("Build your own trading signals using Python and MACD.")

# 2. The Sidebar (For user inputs)
st.sidebar.header("Dashboard Settings")
# We default to SI=F (Silver) but let the user type anything!
ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., SI=F, AAPL, ^NSEI)", "SI=F")
time_period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

# 3. Fetching the Data
# We are importing and using the exact function you just wrote!
st.write(f"Fetching live data for **{ticker}**...")
data = get_live_data(ticker, time_period)

if data is not None:
    # 4. Displaying the UI
    
    # Let users peek at the raw numbers if they want
    with st.expander("View Raw Data Table"):
        st.dataframe(data.tail(10))

    # Plot 1: The Closing Price
    st.subheader("Price History")
    st.line_chart(data['Close'])

    # Plot 2: The MACD Indicator
    st.subheader("MACD Indicator")
    st.markdown("When the MACD line crosses **above** the Signal line, it indicates bullish momentum.")
    
    # Streamlit easily plots multiple columns if we pass them together
    macd_chart_data = data[['MACD', 'Signal_Line']]
    st.line_chart(macd_chart_data)
    
else:
    st.error("Whoops! No data found. Try checking the ticker symbol.")