import streamlit as st
import pandas as pd
from utils.scraper import fetch_stock_data, fetch_newsapi_headlines
from utils.sentiment import analyze_sentiment
from utils.predictor import prepare_features, train_lstm, predict_lstm
import numpy as np

st.set_page_config(page_title="AI Stock Price Predictor", layout="centered")
st.title("📈 AI-Based Stock Price Predictor")

st.write("Enter a stock ticker to predict its next movement using price and news sentiment.")

ticker = st.text_input("Stock Ticker (e.g., AAPL, TSLA)", value="AAPL")
period = st.selectbox("History Period", ["1mo", "3mo", "6mo", "1y"], index=0)

if st.button("Fetch & Predict"):
    # 1. Fetch stock data
    st.subheader("Stock Price History")
    price_df = fetch_stock_data(ticker, period=period)
    if price_df.empty:
        st.error("No data found for this ticker.")
    else:
        st.line_chart(price_df['Close'])

        # 2. Fetch news headlines
        st.subheader("Recent News Headlines")
        headlines = fetch_newsapi_headlines(ticker)
        if not headlines:
            st.warning("No news found or NewsAPI key missing.")
        else:
            for h in headlines:
                st.write(f"- {h}")

        # 3. Sentiment analysis
        st.subheader("News Sentiment Scores")
        if headlines:
            sentiments = analyze_sentiment(headlines)
            sentiment_scores = [s['score'] if s['label'] == 'POSITIVE' else -s['score'] for s in sentiments]
            avg_sentiment = np.mean(sentiment_scores)
            st.write(f"Average Sentiment Score: {avg_sentiment:.2f}")
        else:
            sentiment_scores = [0]
            avg_sentiment = 0

        # 4. Prepare sentiment series for feature engineering
        # For demo, assign today's sentiment to all recent days
        sentiment_series = pd.Series([avg_sentiment]*len(price_df), index=price_df.index)

        # 5. Prepare features and train LSTM
        st.subheader("Training LSTM Model...")
        X, y = prepare_features(price_df, sentiment_series)
        if len(X) < 5:
            st.warning("Not enough data to train the model.")
        else:
            model = train_lstm(X, y, input_size=2, epochs=20)
            # 6. Predict next movement
            last_X = X[-1:]
            pred = predict_lstm(model, last_X)[0]
            st.subheader("Prediction for Next Day")
            if pred > 0.5:
                st.success(f"Predicted Movement: 📈 UP ({pred:.2f})")
            else:
                st.error(f"Predicted Movement: 📉 DOWN ({pred:.2f})")

st.caption("Developed with Streamlit, yfinance, HuggingFace Transformers, and PyTorch.") 