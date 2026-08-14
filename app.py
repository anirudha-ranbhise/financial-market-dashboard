import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import xgboost as xgb

@st.cache_data 
def get_live_data(symbol, time_period="2y"): # Expanded to 2y to give the ML model more training data
    asset = yf.Ticker(symbol)
    data = asset.history(period=time_period)
    
    if data.empty:
        return None
        
    clean_data = data[['Close', 'Volume']].copy()
    
    # 1. Technical Indicators (Features)
    clean_data['EMA_12'] = clean_data['Close'].ewm(span=12, adjust=False).mean()
    clean_data['EMA_26'] = clean_data['Close'].ewm(span=26, adjust=False).mean()
    clean_data['MACD'] = clean_data['EMA_12'] - clean_data['EMA_26']
    clean_data['Signal_Line'] = clean_data['MACD'].ewm(span=9, adjust=False).mean()
    
    delta = clean_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    clean_data['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. ML Feature Engineering: Daily Returns and Volatility
    clean_data['Daily_Return'] = clean_data['Close'].pct_change()
    
    # 3. Target Definition: 1 if tomorrow's close is higher than today's, else 0
    clean_data['Target'] = np.where(clean_data['Close'].shift(-1) > clean_data['Close'], 1, 0)
    
    # Drop rows with NaN values created by our calculations
    clean_data = clean_data.dropna()
    
    return clean_data

def train_and_predict(data):
    # Features we want the model to learn from
    features = ['MACD', 'Signal_Line', 'RSI', 'Daily_Return', 'Volume']
    
    X = data[features]
    y = data['Target']
    
    # Split data: Train on everything except the very last day
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]
    
    # The data point we want to predict (today's data to predict tomorrow)
    X_latest = X.iloc[[-1]]
    
    # Initialize and train the XGBoost Classifier
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # Make prediction and get confidence probability
    prediction = model.predict(X_latest)[0]
    probability = model.predict_proba(X_latest)[0]
    confidence = probability[1] if prediction == 1 else probability[0]
    
    return prediction, confidence

# --- Page UI Setup ---
st.set_page_config(page_title="Market Dashboard", page_icon="📈", layout="wide")
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .block-container {padding-top: 2rem;}
    </style>
    """, unsafe_allow_html=True)

st.title("📈 AI-Powered Market Dashboard")
st.markdown("Professional technical analysis suite featuring XGBoost trend prediction.")
st.markdown("---") 

# --- Sidebar ---
st.sidebar.header("⚙️ Control Panel")
stock_presets = {
    "Silver Futures": "SI=F", "Gold Futures": "GC=F", "Nifty 50 (India)": "^NSEI",
    "Reliance Industries": "RELIANCE.NS", "Tata Motors": "TATAMOTORS.NS", 
    "Infosys": "INFY.NS", "Apple": "AAPL", "Microsoft": "MSFT",
    "Bitcoin (USD)": "BTC-USD", "Enter Custom Ticker...": "CUSTOM"
}

selected_option = st.sidebar.selectbox("Select Asset", list(stock_presets.keys()))
ticker = st.sidebar.text_input("Enter Ticker", "NVDA") if selected_option == "Enter Custom Ticker..." else stock_presets[selected_option]

# --- ML Trigger ---
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 XGBoost Predictor")
st.sidebar.markdown("Train a machine learning model on historical indicator data to predict tomorrow's trend.")
run_ml = st.sidebar.button("Predict Tomorrow's Trend", type="primary")

# --- Fetching Data ---
data = get_live_data(ticker)

if data is not None:
    latest_close = data['Close'].iloc[-1]
    price_change = latest_close - data['Close'].iloc[-2]
    latest_macd = data['MACD'].iloc[-1]
    latest_rsi = data['RSI'].iloc[-1]
    
    # Add a 4th column for our ML result
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Current Price", value=f"{latest_close:.2f}", delta=f"{price_change:.2f}")
    with col2:
        st.metric(label="MACD Value", value=f"{latest_macd:.2f}")
    with col3:
        rsi_delta = "Overbought" if latest_rsi > 70 else "Oversold" if latest_rsi < 30 else "Neutral"
        rsi_color = "normal" if latest_rsi > 70 else "inverse" if latest_rsi < 30 else "off"
        st.metric(label="14-Day RSI", value=f"{latest_rsi:.2f}", delta=rsi_delta, delta_color=rsi_color)
    with col4:
        # Display ML Results if the button was clicked
        if run_ml:
            with st.spinner("Training model..."):
                prediction, confidence = train_and_predict(data)
                pred_label = "▲ UP" if prediction == 1 else "▼ DOWN"
                pred_color = "normal" if prediction == 1 else "inverse"
                st.metric(label="ML Prediction (Next Day)", value=pred_label, delta=f"Conf: {confidence*100:.1f}%", delta_color=pred_color)
        else:
            st.metric(label="ML Prediction (Next Day)", value="Waiting...", delta="Click 'Predict' in sidebar", delta_color="off")
        
    st.markdown("---")

    # --- Charts ---
    st.subheader(f"Price History: {ticker.upper()}")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='#00ff88', width=2)))
    fig_price.update_layout(template='plotly_dark', margin=dict(l=0, r=0, t=30, b=0), height=350)
    st.plotly_chart(fig_price, use_container_width=True)

    ind_col1, ind_col2 = st.columns(2)
    with ind_col1:
        st.subheader("MACD")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD', line=dict(color='#00d4ff', width=2)))
        fig_macd.add_trace(go.Scatter(x=data.index, y=data['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='#ffaa00', width=2)))
        fig_macd.update_layout(template='plotly_dark', margin=dict(l=0, r=0, t=30, b=0), height=300)
        st.plotly_chart(fig_macd, use_container_width=True)
        
    with ind_col2:
        st.subheader("RSI (Relative Strength Index)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='#ff00ff', width=2)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_rsi.update_yaxes(range=[0, 100])
        fig_rsi.update_layout(template='plotly_dark', margin=dict(l=0, r=0, t=30, b=0), height=300)
        st.plotly_chart(fig_rsi, use_container_width=True)
    
else:
    st.error("Whoops! No data found. Try checking the ticker symbol.")
