# scoring.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the trained model and scaler
model_path = 'model/model.pkl'
scaler_path = 'model/scaler.pkl'

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    numeric_transformer = pickle.load(scaler_file)

commodity_tickers = ['GC=F', 'SI=F', 'CL=F', 'NG=F']  # Gold, Silver, Oil, Natural Gas

# ***********Change: Removed user input prompt for ticker and added `company_name` and `ticker_symbol` as function parameters.
def make_prediction(company_name, ticker_symbol):
    # ***********Change: Fetch financial data using the ticker symbol and commodities list
    financial_data = fetch_financial_data(ticker_symbol, commodity_tickers)

    # Compute technical indicators for the fetched financial data
    financial_data = compute_technical_indicators(financial_data)
    
    # Fetch news sentiment for the company name
    sentiment_score, sentiment_volatility = fetch_news_sentiment(company_name)

    # Prepare data for scoring, adding the sentiment data as inputs
    scoring_data = prepare_scoring_data(financial_data, sentiment_score, sentiment_volatility)

    # Make predictions using the scoring data
    predictions = make_predictions(scoring_data)
    
    # ***********Change: Return the latest prediction result only (upward or downward)
    latest_prediction = "upward" if predictions.iloc[-1]['Predicted_Direction'] == 1 else "downward"
    return latest_prediction


# ***********Change: Modified `fetch_financial_data` to take `ticker` and `commodities` as parameters
def fetch_financial_data(ticker, commodities):
    # Include user-selected ticker along with commodities
    tickers = [ticker] + commodities
    data = yf.download(tickers, period='1mo', interval='1d')
    
    # Resample to weekly frequency and use 'Close' prices only
    data = data['Close'].resample('W-MON').last()
    
    # Flatten multi-index columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns.values]
    
    # Rename columns for consistency
    data.columns = ['User_Ticker'] + ['Gold', 'Silver', 'Oil', 'NaturalGas']
    
    # Reset index to remove time from Date field
    data.index = data.index.strftime('%Y-%m-%d')
    return data

# Compute technical indicators for the user's ticker without moving averages
def compute_technical_indicators(data):
    data['User_Ticker_pct_change'] = data['User_Ticker'].pct_change() * 100
    data['RSI'] = data['User_Ticker'].rolling(window=3).apply(lambda x: 100 - (100 / (1 + (x.diff().apply(lambda y: y if y > 0 else 0).sum() / x.diff().apply(lambda y: -y if y < 0 else 0).sum()))))
    data['ATR'] = data['User_Ticker'].rolling(window=3).apply(lambda x: x.max() - x.min(), raw=False)
    return data.dropna()

# Fetch news sentiment
def fetch_news_sentiment(query, count=10):
    subscription_key = st.secrets["bing_api_key"] 
    search_url = "https://api.bing.microsoft.com/v7.0/news/search"
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": query, "count": count, "mkt": "en-US", "freshness": "Week"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    news_data = response.json()
    sia = SentimentIntensityAnalyzer()
    
    # Compute sentiment scores for each article description
    sentiment_scores = [sia.polarity_scores(article['description'])['compound'] for article in news_data['value']]
    weekly_sentiment_score = np.mean(sentiment_scores)
    sentiment_volatility = np.std(sentiment_scores)
    
    return weekly_sentiment_score, sentiment_volatility

# Prepare data for scoring with exactly 2 lags, ensuring only necessary columns are created
def prepare_scoring_data(data, sentiment_score, sentiment_volatility):
    # Add sentiment features with lag naming
    data['Weekly_Sentiment_Score_lag_1'] = sentiment_score
    data['Sentiment_Volatility_lag_1'] = sentiment_volatility
    data['Weekly_Price_Change_%_lag_1'] = data['User_Ticker_pct_change']
    data['RSI_lag_1'] = data['RSI']
    data['ATR_lag_1'] = data['ATR']
    
    # Create lag columns for lag 2 by repeating the values from lag 1
    data['Weekly_Sentiment_Score_lag_2'] = data['Weekly_Sentiment_Score_lag_1']
    data['Sentiment_Volatility_lag_2'] = data['Sentiment_Volatility_lag_1']
    data['Weekly_Price_Change_%_lag_2'] = data['Weekly_Price_Change_%_lag_1']
    data['RSI_lag_2'] = data['RSI_lag_1']
    data['ATR_lag_2'] = data['ATR_lag_1']
    
    # Retain only the columns expected in the model
    expected_features = [
        'Weekly_Sentiment_Score_lag_1', 'Weekly_Sentiment_Score_lag_2',
        'Sentiment_Volatility_lag_1', 'Sentiment_Volatility_lag_2',
        'Weekly_Price_Change_%_lag_1', 'Weekly_Price_Change_%_lag_2',
        'RSI_lag_1', 'RSI_lag_2',
        'ATR_lag_1', 'ATR_lag_2'
    ]
    return data[expected_features]

# ***********Change: Updated `make_predictions` to work as a function for feature scoring
def make_predictions(data):
    # Extract features for scoring and convert to NumPy array to remove feature names
    X = data.to_numpy()

    # Predict using the trained model
    predictions = model.predict(X)
    
    # Append predictions to the DataFrame for reference
    data = data.copy()  # Ensure we're working with a copy to avoid SettingWithCopyWarning
    data.loc[:, 'Predicted_Direction'] = predictions  # Use .loc to set values

    # Return only the prediction column
    return data[['Predicted_Direction']]
