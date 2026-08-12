# financial-market-dashboard
# 📈 Financial Market Dashboard

A real-time financial market dashboard built with Python. This application fetches live market data using the Yahoo Finance API and calculates technical momentum indicators like the MACD (Moving Average Convergence Divergence) to identify potential market trends.

## Features
* **Live Data Integration:** Fetches real-time stock, commodity, and index data via `yfinance`.
* **Technical Analysis:** Automatically calculates the MACD and Signal lines using 12-day and 26-day Exponential Moving Averages (EMA).
* **Interactive UI:** Features a responsive front-end built with Streamlit for dynamic ticker selection and time-series visualization.

## Tech Stack
* **Language:** Python
* **Libraries:** Streamlit, Pandas, yfinance
* **Deployment:** Streamlit Community Cloud

## How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python -m streamlit run app.py`
