import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.metrics import accuracy_score 

@st.cache_data 
def get_live_data(symbol, time_period="5y"): # UPGRADE: Increased to 5 years for way more training data!
    asset = yf.Ticker(symbol)
    data = asset.history(period=time_period)
    
    if data.empty:
        return None
        
    clean_data = data[['Close', 'Volume']].copy()
    
    # --- Base Technical Indicators ---
    clean_data['EMA_12'] = clean_data['Close'].ewm(span=12, adjust=False).mean()
    clean_data['EMA_26'] = clean_data['Close'].ewm(span=26, adjust=False).mean()
    clean_data['MACD'] = clean_data['EMA_12'] - clean_data['EMA_26']
    clean_data['Signal_Line'] = clean_data['MACD'].ewm(span=9, adjust=False).mean()
    
    delta = clean_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    clean_data['RSI'] = 100 - (100 / (1 + rs))
    clean_data['Daily_Return'] = clean_data['Close'].pct_change()
    
    # --- ACCURACY UPGRADE: Advanced Feature Engineering ---
    # 1. Simple Moving Averages (Trend identification)
    clean_data['SMA_20'] = clean_data['Close'].rolling(window=20).mean()
    clean_data['SMA_50'] = clean_data['Close'].rolling(window=50).mean()
    
    # 2. Volatility (Risk measurement)
    clean_data['Volatility'] = clean_data['Close'].rolling(window=14).std()
    
    # 3. Lag Features (Giving the model a memory of the past 2 days)
    clean_data['RSI_Lag1'] = clean_data['RSI'].shift(1)
    clean_data['MACD_Lag1'] = clean_data['MACD'].shift(1)
    clean_data['Return_Lag1'] = clean_data['Daily_Return'].shift(1)
    
    # --- Target Definition ---
    clean_data['Target'] = np.where(clean_data['Close'].shift(-1) > clean_data['Close'], 1, 0)
    
    # Drop rows with NaN values created by our rolling windows and lags
    clean_data = clean_data.dropna()
    
    return clean_data

def train_and_predict(data):
    # UPGRADE: The model now has a much wider view of the market mechanics
    features = ['MACD', 'Signal_Line', 'RSI', 'Daily_Return', 'Volume', 
                'SMA_20', 'SMA_50', 'Volatility', 'RSI_Lag1', 'MACD_Lag1', 'Return_Lag1']
    
    X = data[features]
    y = data['Target']
    
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]
    X_latest = X.iloc[[-1]]
    
    # UPGRADE: Hyperparameter Tuning for Time-Series
    # We restrict max_depth to prevent overfitting and use colsample to add randomness
    model = xgb.XGBClassifier(
        n_estimators=200,        # More trees
        learning_rate=0.05,      # Learning slower
        max_depth=4,             # Shallower trees to prevent overfitting noisy stock data
        subsample=0.8,           # Train on 80% of data per tree
        colsample_bytree=0.8,    # Use 80% of features per tree
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    prediction = model.predict(X_latest)[0]
    probability = model.predict_proba(X_latest)[0]
    confidence = probability[1] if prediction == 1 else probability[0]
    
    return prediction, confidence
    def backtest_model(data, test_days=100):
    # Notice how EVERYTHING below is pushed to the right by 4 spaces!
    if len(data) < test_days + 100:
        return None
        
    features = ['MACD', 'Signal_Line', 'RSI', 'Daily_Return', 'Volume', 
                'SMA_20', 'SMA_50', 'Volatility', 'RSI_Lag1', 'MACD_Lag1', 'Return_Lag1']
    
    X = data[features]
    y = data['Target']
    
    X_train = X.iloc[:-test_days]
    y_train = y.iloc[:-test_days]
    
    X_test = X.iloc[-test_days:]
    y_test = y.iloc[-test_days:]
    
    model = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, 
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy
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
# --- Dynamic Data Pipeline ---
@st.cache_data
def load_nse_tickers():
    try:
        # Read the official CSV file
        df = pd.read_csv('EQUITY_L.csv')
        
        # Yahoo Finance requires '.NS' at the end of Indian stock symbols
        symbols = df['SYMBOL'].astype(str) + ".NS"
        names = df['NAME OF COMPANY']
        
        # Zip them together into a dictionary { "Company Name": "SYMBOL.NS" }
        ticker_dict = dict(zip(names, symbols))
        
        # Add a few global fallback options at the top
        ticker_dict["Bitcoin (USD)"] = "BTC-USD"
        ticker_dict["Gold Futures"] = "GC=F"
        ticker_dict["Enter Custom Ticker..."] = "CUSTOM"
        
        return ticker_dict
    except FileNotFoundError:
        # If the app can't find the CSV, it safely falls back to a basic list
        return {
            "Reliance Industries": "RELIANCE.NS", 
            "TCS": "TCS.NS",
            "Enter Custom Ticker...": "CUSTOM"
        }

# Load the dynamic dictionary
stock_presets = load_nse_tickers()

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
            with st.spinner("Training & Backtesting..."):
                prediction, confidence = train_and_predict(data)
                historical_accuracy = backtest_model(data, test_days=100)
                
                pred_label = "▲ UP" if prediction == 1 else "▼ DOWN"
                pred_color = "normal" if prediction == 1 else "inverse"
                
                # We show the Prediction, Confidence, AND the new Backtested Accuracy
                st.metric(label="ML Prediction (Next Day)", 
                          value=pred_label, 
                          delta=f"Conf: {confidence*100:.1f}%", 
                          delta_color=pred_color)
                
                if historical_accuracy:
                    st.caption(f"🧪 **Model Accuracy:** {historical_accuracy*100:.1f}% (Over last 100 days)")
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
