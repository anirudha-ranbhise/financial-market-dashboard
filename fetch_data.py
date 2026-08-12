import yfinance as yf
import pandas as pd

def get_live_data(symbol, time_period="1y"):
    print(f"Fetching market data for {symbol}...\n")
    
    # Create a Ticker object and get historical data
    asset = yf.Ticker(symbol)
    data = asset.history(period=time_period)
    
    if data.empty:
        print("Whoops! No data found. Check your ticker symbol.")
        return None
        
    # Keep it clean: we just need the closing price and volume for our indicators
    clean_data = data[['Close', 'Volume']]
    
    print(f"--- Most Recent 5 Days of Data for {symbol} ---")
    print(clean_data.tail())
    
    return clean_data

if __name__ == "__main__":
    # We are using Silver Futures (SI=F) as a test, but you can easily swap this 
    # to 'GC=F' for Gold, or '^NSEI' for the Nifty 50!
    my_ticker = "SI=F"
    df = get_live_data(my_ticker, time_period="1y")
    
    if df is not None:
        print("\nBoom! Data fetched successfully. Ready for the math.")