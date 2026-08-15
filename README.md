# 📈 AI-Powered Market Dashboard

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![XGBoost](https://img.shields.io/badge/Machine%20Learning-XGBoost-green.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

An advanced, full-stack financial market dashboard that combines real-time data engineering, custom technical indicators, and machine learning to forecast market trends. 

This application fetches live market data for global assets and NSE Indian stocks, calculates technical momentum indicators (MACD, RSI, SMA) from scratch, and utilizes an XGBoost classification model to predict the next day's market movement based on historical patterns.

---

## 🚀 Key Features

*   **Live Data Pipeline:** Integrates with the `yfinance` API to fetch real-time market data for thousands of global stocks, commodities, crypto, and market indices.
*   **Predictive Machine Learning:** Features an integrated **XGBoost Classifier** that trains instantly on 5 years of historical data to predict tomorrow's trend (UP/DOWN) and outputs a confidence probability.
*   **Automated Backtesting:** Includes a backtesting engine that evaluates the ML model against the last 100 trading days, providing a transparent historical accuracy score.
*   **Algorithmic Indicators:** Custom Python/Pandas logic calculates Moving Averages, MACD (Moving Average Convergence Divergence), and RSI (Relative Strength Index) without relying on pre-built indicator libraries.
*   **Dynamic UI & Seamless Charting:** Built with Streamlit and Plotly for a highly responsive, dark-themed interface. Features intelligent gap-handling for seamless charts, completely ignoring after-hours and weekend market closures.
*   **Massive NSE Integration:** Dynamically loads all National Stock Exchange (NSE) tickers via an external CSV pipeline, making the app a universal screener.

---

## 🛠️ Tech Stack

*   **Frontend / UI:** [Streamlit](https://streamlit.io/)
*   **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Data Source:** [yfinance](https://pypi.org/project/yfinance/)
*   **Machine Learning:** [XGBoost](https://xgboost.readthedocs.io/), [Scikit-Learn](https://scikit-learn.org/)
*   **Data Visualization:** [Plotly Graph Objects](https://plotly.com/python/)

---

## ⚙️ Installation & Setup

To run this project locally on your machine, follow these steps:

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/market-dashboard.git
cd market-dashboard
```

**2. Create a virtual environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```
*(Ensure your `requirements.txt` includes: `streamlit`, `pandas`, `numpy`, `yfinance`, `plotly`, `xgboost`, `scikit-learn`)*

**4. Add the NSE Equity List (Optional but recommended)**
* Download the official `EQUITY_L.csv` file from the NSE website.
* Place the file directly in the root directory of this project so the dynamic search pipeline can map Indian stocks automatically.

**5. Run the Application**
```bash
streamlit run app.py
```

---

## 📊 Usage Guide

1. **Select an Asset:** Use the sidebar dropdown to select from popular presets (Indian stocks, US Tech, Commodities, Crypto) or choose "Enter Custom Ticker..." to search manually (e.g., `TSLA`, `RELIANCE.NS`).
2. **Adjust Timeframes:** Select a visual time period (e.g., 1 Month, 1 Year, 5 Years). The backend will independently fetch the necessary data to ensure ML accuracy remains high regardless of the frontend view.
3. **Run AI Predictor:** Click **"Predict Tomorrow's Trend"** in the sidebar to trigger the XGBoost model. The app will train the model, run a 100-day backtest, and display the prediction metrics in the top dashboard.
4. **Customize Charts:** Toggle MACD and RSI charts on/off and adjust the Fast/Slow Simple Moving Averages using the interactive sliders.

---

## 🧠 Machine Learning Architecture

The prediction engine relies on **Feature Engineering** to give the model a "memory" of market mechanics. 
Features fed into the XGBoost Classifier include:
*   `MACD` & `Signal_Line` (Momentum)
*   `RSI` & `RSI_Lag1` (Overbought/Oversold velocity)
*   `SMA_20` & `SMA_50` (Trend identification)
*   `Volatility` (14-day rolling standard deviation)
*   `Daily_Return` & `Volume`

The model utilizes shallow decision trees (`max_depth=4`) and data subsetting (`subsample=0.8`) to aggressively prevent overfitting on noisy financial time-series data.
