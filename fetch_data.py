import yfinance as yf
import pandas as pd

def get_live_data(symbol, time_period="1y"):
    asset = yf.Ticker(symbol)
    data = asset.history(period=time_period)
    
    if data.empty:
        return None
        
    clean_data = data[['Close', 'Volume']].copy()
    
    # ---------------------------------------------------------
    # The crucial math step that creates the missing columns!
    # ---------------------------------------------------------
    clean_data['EMA_12'] = clean_data['Close'].ewm(span=12, adjust=False).mean()
    clean_data['EMA_26'] = clean_data['Close'].ewm(span=26, adjust=False).mean()
    clean_data['MACD'] = clean_data['EMA_12'] - clean_data['EMA_26']
    clean_data['Signal_Line'] = clean_data['MACD'].ewm(span=9, adjust=False).mean()
    
    return clean_data
