import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# ---------------------------------------------------------
# THE FIX: We moved the data function directly into app.py
# We also added @st.cache_data to make the app lightning fast!
# ---------------------------------------------------------
@st.cache_data 
def get_live_data(symbol, time_period="1y"):
    asset = yf.Ticker(symbol)
    data = asset.history(period=time_period)
    
    if data.empty:
        return None
        
    clean_data = data[['Close', 'Volume']].copy()
    
    # The MACD Math explicitly defined here
    clean_data['EMA_12'] = clean_data['Close'].ewm(span=12, adjust=False).mean()
    clean_data['EMA_26'] = clean_data['Close'].ewm(span=26, adjust=False).mean()
    clean_data['MACD'] = clean_data['EMA_12'] - clean_data['EMA_26']
    clean_data['Signal_Line'] = clean_data['MACD'].ewm(span=9, adjust=False).mean()
    
    return clean_data
# ---------------------------------------------------------

# Page UI Setup
st.set_page_config(page_title="Market Dashboard", page_icon="📈", layout="wide")
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 2rem;}
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Premium Market Dashboard")
st.markdown("Advanced technical analysis and MACD momentum tracking.")
st.markdown("---") 

# Sidebar
st.sidebar.header("⚙️ Control Panel")
ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., SI=F, GC=F, ^NSEI)", "SI=F")
time_period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

# Fetching the data using the function at the top of this file
data = get_live_data(ticker, time_period)

if data is not None:
    # Top Level Metrics
    latest_close = data['Close'].iloc[-1]
    previous_close = data['Close'].iloc[-2]
    price_change = latest_close - previous_close
    
    latest_macd = data['MACD'].iloc[-1]
    latest_signal = data['Signal_Line'].iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Current Price", value=f"{latest_close:.2f}", delta=f"{price_change:.2f}")
    with col2:
        st.metric(label="MACD Value", value=f"{latest_macd:.2f}")
    with col3:
        st.metric(label="Signal Line", value=f"{latest_signal:.2f}")
        
    st.markdown("---")

    # Price Chart
    st.subheader(f"Price History: {ticker.upper()}")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=data.index, y=data['Close'], 
                                   mode='lines', name='Close Price', 
                                   line=dict(color='#00ff88', width=2)))
    fig_price.update_layout(template='plotly_dark', margin=dict(l=0, r=0, t=30, b=0), height=400)
    st.plotly_chart(fig_price, use_container_width=True)

    # MACD Chart
    st.subheader("MACD & Momentum")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD'], 
                                  mode='lines', name='MACD', 
                                  line=dict(color='#00d4ff', width=2)))
    fig_macd.add_trace(go.Scatter(x=data.index, y=data['Signal_Line'], 
                                  mode='lines', name='Signal Line', 
                                  line=dict(color='#ffaa00', width=2)))
    fig_macd.update_layout(template='plotly_dark', margin=dict(l=0, r=0, t=30, b=0), height=400)
    st.plotly_chart(fig_macd, use_container_width=True)
    
else:
    st.error("Whoops! No data found. Try checking the ticker symbol.")
# ---------------------------------------------------------
# Upgraded Sidebar with Popular Presets + Custom Option
# ---------------------------------------------------------
st.sidebar.header("⚙️ Control Panel")

# 1. Preset dictionary: Friendly Name -> Yahoo Ticker
stock_presets = {
    "Silver Futures": "SI=F",
    "Gold Futures": "GC=F",
    "Nifty 50 (India)": "^NSEI",
    "Reliance Industries": "RELIANCE.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Infosys": "INFY.NS",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Bitcoin (USD)": "BTC-USD",
    "Enter Custom Ticker...": "CUSTOM"
}

# 2. Dropdown for quick selection
selected_option = st.sidebar.selectbox("Select Asset / Market", list(stock_presets.keys()))

# 3. Logic to determine the active ticker
if selected_option == "Enter Custom Ticker...":
    ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., TSLA, TCS.NS)", "NVDA")
else:
    ticker = stock_presets[selected_option]

time_period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
