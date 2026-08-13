import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from fetch_data import get_live_data

# 1. Page Configuration & Custom CSS
# Hiding the default Streamlit headers and footers makes it look like a pro app
st.set_page_config(page_title="Market Dashboard", page_icon="📈", layout="wide")
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    /* Adds a subtle padding to the top */
    .block-container {padding-top: 2rem;}
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Premium Market Dashboard")
st.markdown("Advanced technical analysis and MACD momentum tracking.")
st.markdown("---") # Adds a clean horizontal divider

# 2. The Sidebar 
st.sidebar.header("⚙️ Control Panel")
ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., SI=F, GC=F, ^NSEI)", "SI=F")
time_period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

# 3. Fetching the Data
data = get_live_data(ticker, time_period)

if data is not None:
    # 4. AESTHETIC UPGRADE: Top Level Metrics
    # Grabbing the absolute latest data points for the cards
    latest_close = data['Close'].iloc[-1]
    previous_close = data['Close'].iloc[-2]
    price_change = latest_close - previous_close
    
    latest_macd = data['MACD'].iloc[-1]
    latest_signal = data['Signal_Line'].iloc[-1]
    
    # Using Streamlit columns to create a sleek top row of data cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Current Price", value=f"{latest_close:.2f}", delta=f"{price_change:.2f}")
    with col2:
        st.metric(label="MACD Value", value=f"{latest_macd:.2f}")
    with col3:
        st.metric(label="Signal Line", value=f"{latest_signal:.2f}")
        
    st.markdown("---")

    # 5. AESTHETIC UPGRADE: Plotly Interactive Charts
    st.subheader(f"Price History: {ticker.upper()}")
    
    # Creating a professional price chart
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=data.index, y=data['Close'], 
                                   mode='lines', name='Close Price', 
                                   line=dict(color='#00ff88', width=2))) # Sleek neon green
    fig_price.update_layout(template='plotly_dark', margin=dict(l=0, r=0, t=30, b=0), height=400)
    st.plotly_chart(fig_price, use_container_width=True)

    # Creating a professional MACD chart
    st.subheader("MACD & Momentum")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD'], 
                                  mode='lines', name='MACD', 
                                  line=dict(color='#00d4ff', width=2))) # Bright blue
    fig_macd.add_trace(go.Scatter(x=data.index, y=data['Signal_Line'], 
                                  mode='lines', name='Signal Line', 
                                  line=dict(color='#ffaa00', width=2))) # Pale orange contrast
    fig_macd.update_layout(template='plotly_dark', margin=dict(l=0, r=0, t=30, b=0), height=400)
    st.plotly_chart(fig_macd, use_container_width=True)
    
else:
    st.error("Whoops! No data found. Try checking the ticker symbol.")
